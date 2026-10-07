"""
red_electrica.py
================
Todo lo que tiene que ver con la red eléctrica en pandapower:

    * Cargar el caso (config.CASO_RED) y extraer los datos de los generadores
      síncronos, cargas, líneas y transformadores.
    * Resolver el FLUJO DE CARGA AC dado un escenario (cargas) y un despacho
      (P y V de los generadores).  -> RedElectrica.resolver_flujo()
    * Calcular el COSTO de un despacho dado los precios.  -> costo_despacho()
    * Despacho de ORDEN DE MÉRITO "placa de cobre" (ignora la red): es la
      referencia económica de la recompensa.  -> despacho_orden_merito()
    * OPF AC EXACTO de pandapower (pp.runopp): la referencia "ideal" contra la
      que se compara al agente en la evaluación.  -> resolver_opf()

Convención de generadores (se usa en TODO el proyecto):
    índice 0      -> generador de la red externa (net.ext_grid): es la barra
                     SLACK. Su potencia activa NO la decide el agente: sale del
                     balance del flujo de carga (cubre carga + pérdidas − resto).
    índices 1..N  -> generadores PV (net.gen) en el orden de pandapower. El
                     agente decide su potencia activa P.
    El agente decide la consigna de tensión V de TODOS (slack incluida).
"""

from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandapower as pp
import pandapower.networks as pn

import config as C

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# pandapower informa por logging cada ajuste menor del OPF; se silencia.
logging.getLogger("pandapower").setLevel(logging.ERROR)


@dataclass
class ResultadoFlujo:
    """Resultado de un flujo de carga, ya en las unidades que usa el proyecto."""
    convergio: bool
    vm_pu: np.ndarray = field(default_factory=lambda: np.zeros(0))          # tensión por barra
    cargabilidad_lineas_pct: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cargabilidad_trafos_pct: np.ndarray = field(default_factory=lambda: np.zeros(0))
    p_gen_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))       # [slack, PV...]
    q_gen_mvar: np.ndarray = field(default_factory=lambda: np.zeros(0))     # [slack, PV...]
    perdidas_mw: float = 0.0

    @property
    def cargabilidad_ramas_pct(self) -> np.ndarray:
        """Líneas y transformadores juntos (lo que se monitorea)."""
        return np.concatenate([self.cargabilidad_lineas_pct, self.cargabilidad_trafos_pct])


class RedElectrica:
    """Envuelve una red pandapower y resuelve flujos de carga sobre ella."""

    def __init__(self, caso: str = C.CASO_RED, motor: str = C.MOTOR_FLUJO):
        self.caso = caso
        self.motor = motor
        self.net = getattr(pn, caso)()
        net = self.net

        if len(net.ext_grid) != 1:
            raise ValueError("Se espera exactamente una red externa (slack).")

        # ---------------- Generadores síncronos ----------------
        # Orden: [slack, gen_0, gen_1, ...]
        self.n_pv = len(net.gen)
        self.n_gen = self.n_pv + 1
        eg, g = net.ext_grid, net.gen
        self.bus_gen = np.r_[eg.bus.values, g.bus.values].astype(int)
        self.p_min = np.r_[eg.min_p_mw.values, g.min_p_mw.values].astype(float)
        self.p_max = np.r_[eg.max_p_mw.values, g.max_p_mw.values].astype(float)
        self.q_min = np.r_[eg.min_q_mvar.values, g.min_q_mvar.values].astype(float)
        self.q_max = np.r_[eg.max_q_mvar.values, g.max_q_mvar.values].astype(float)
        # Nombres legibles: "G1 (B1, slack)". Las barras se muestran en base 1.
        self.nombres_gen = [
            f"G{i + 1} (B{b + 1}{', slack' if i == 0 else ''})"
            for i, b in enumerate(self.bus_gen)
        ]

        # ---------------- Cargas ----------------
        self.n_cargas = len(net.load)
        self.p_carga_base = net.load.p_mw.values.astype(float).copy()
        self.q_carga_base = net.load.q_mvar.values.astype(float).copy()
        self.bus_carga = net.load.bus.values.astype(int)

        # ---------------- Barras y ramas ----------------
        self.n_barras = len(net.bus)
        self.indices_barras = net.bus.index.values
        self.n_lineas = len(net.line)
        self.n_trafos = len(net.trafo)
        # Corriente nominal efectiva de cada línea (igual que pandapower:
        # max_i_ka * df * parallel).
        self.i_nom_linea_ka = (net.line.max_i_ka * net.line.df * net.line.parallel).values.astype(float)
        if self.n_trafos:
            sn = net.trafo.sn_mva.values * net.trafo.parallel.values
            self.i_nom_trafo_hv_ka = sn / (np.sqrt(3) * net.trafo.vn_hv_kv.values)
            self.i_nom_trafo_lv_ka = sn / (np.sqrt(3) * net.trafo.vn_lv_kv.values)
        self.nombres_lineas = [
            f"L{i + 1} ({f + 1}-{t + 1})"
            for i, (f, t) in enumerate(zip(net.line.from_bus.values, net.line.to_bus.values))
        ]
        self.nombres_trafos = [
            f"T{i + 1} ({h + 1}-{l + 1})"
            for i, (h, l) in enumerate(zip(net.trafo.hv_bus.values, net.trafo.lv_bus.values))
        ]
        self.nombres_ramas = self.nombres_lineas + self.nombres_trafos

        # Los límites de tensión de barras se unifican a los de config para que
        # agente, recompensa y OPF usen exactamente la misma banda.
        net.bus["min_vm_pu"] = C.TENSION_MIN_PU
        net.bus["max_vm_pu"] = C.TENSION_MAX_PU

        # Copia independiente para el OPF (pp.runopp modifica la red).
        self._net_opf = None

        if motor == "lightsim":
            self._iniciar_lightsim()
        elif motor != "pandapower":
            raise ValueError(f"Motor de flujo desconocido: {motor}")

    # ------------------------------------------------------------------
    # Motor lightsim2grid
    # ------------------------------------------------------------------
    def _iniciar_lightsim(self):
        """Construye el modelo de lightsim2grid a partir de la red pandapower.

        lightsim2grid traduce la red pandapower a su propio modelo en C++ una
        sola vez; después solo se cambian consignas y se resuelve. En el modelo
        resultante los generadores quedan como [gen_0 ... gen_{N-1}, ext_grid],
        es decir, la slack queda AL FINAL (se verifica abajo).
        """
        from lightsim2grid.gridmodel import init_from_pandapower

        pp.runpp(self.net)  # flujo inicial: lightsim toma de aquí el estado
        with warnings.catch_warnings():
            # Aviso informativo: la slack es la ext_grid (no un gen marcado
            # como slack); lightsim la toma correctamente igual.
            warnings.simplefilter("ignore")
            self._ls = init_from_pandapower(self.net)
        gens_ls = list(self._ls.get_generators())
        if len(gens_ls) != self.n_gen or not gens_ls[-1].is_slack:
            raise RuntimeError("Orden inesperado de generadores en lightsim2grid.")
        # Mapa: índice del proyecto -> índice en lightsim
        self._ls_idx = np.r_[self.n_pv, np.arange(self.n_pv)]
        self._V_plano = np.ones(self._ls.total_bus(), dtype=complex)

    def _flujo_lightsim(self, p_carga, q_carga, p_pv, v_gen) -> ResultadoFlujo:
        ls = self._ls
        for i in range(self.n_cargas):
            ls.change_p_load(i, float(p_carga[i]))
            ls.change_q_load(i, float(q_carga[i]))
        for k in range(self.n_pv):
            ls.change_p_gen(k, float(p_pv[k]))
        for k in range(self.n_gen):
            ls.change_v_gen(int(self._ls_idx[k]), float(v_gen[k]))

        # Arranque PLANO siempre (|V| = 1 p.u., ángulo 0). Se descartó el
        # arranque en caliente (usar la solución anterior) porque hace que el
        # resultado dependa levemente del flujo previo, y el ambiente debe ser
        # determinista: mismo escenario + misma acción = misma recompensa.
        try:
            V = self._ls.ac_pf(self._V_plano.copy(), C.MAX_ITER_FLUJO, C.TOLERANCIA_FLUJO)
        except Exception:
            V = np.zeros(0)
        if not len(V) or not np.all(np.isfinite(V)):
            return ResultadoFlujo(convergio=False)

        vm = np.abs(V)
        _, _, _, a1 = ls.get_line_res1()
        _, _, _, a2 = ls.get_line_res2()
        carg_l = np.maximum(a1, a2) / self.i_nom_linea_ka * 100.0
        if self.n_trafos:
            _, _, _, ahv = ls.get_trafo_res1()
            _, _, _, alv = ls.get_trafo_res2()
            carg_t = np.maximum(ahv / self.i_nom_trafo_hv_ka, alv / self.i_nom_trafo_lv_ka) * 100.0
        else:
            carg_t = np.zeros(0)
        pg, qg, _ = ls.get_gen_res()
        p_gen = np.asarray(pg)[self._ls_idx]
        q_gen = np.asarray(qg)[self._ls_idx]
        perdidas = float(p_gen.sum() - p_carga.sum())
        return ResultadoFlujo(True, vm, carg_l, carg_t, p_gen, q_gen, perdidas)

    # ------------------------------------------------------------------
    # Motor pandapower clásico
    # ------------------------------------------------------------------
    def _flujo_pandapower(self, p_carga, q_carga, p_pv, v_gen) -> ResultadoFlujo:
        net = self.net
        net.load["p_mw"] = p_carga
        net.load["q_mvar"] = q_carga
        net.gen["p_mw"] = p_pv
        net.ext_grid["vm_pu"] = v_gen[0]
        net.gen["vm_pu"] = v_gen[1:]
        try:
            pp.runpp(net, max_iteration=C.MAX_ITER_FLUJO, tolerance_mva=C.TOLERANCIA_FLUJO)
        except Exception:
            return ResultadoFlujo(convergio=False)
        if not net.converged:
            return ResultadoFlujo(convergio=False)
        vm = net.res_bus.vm_pu.values.copy()
        carg_l = net.res_line.loading_percent.values.copy()
        carg_t = net.res_trafo.loading_percent.values.copy() if self.n_trafos else np.zeros(0)
        p_gen = np.r_[net.res_ext_grid.p_mw.values, net.res_gen.p_mw.values]
        q_gen = np.r_[net.res_ext_grid.q_mvar.values, net.res_gen.q_mvar.values]
        perdidas = float(p_gen.sum() - p_carga.sum())
        return ResultadoFlujo(True, vm, carg_l, carg_t, p_gen, q_gen, perdidas)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def resolver_flujo(self, p_carga, q_carga, p_pv, v_gen) -> ResultadoFlujo:
        """Resuelve el flujo de carga AC.

        Parámetros
        ----------
        p_carga, q_carga : (n_cargas,) MW / MVAr de cada carga.
        p_pv             : (n_pv,) MW de los generadores PV (sin la slack).
        v_gen            : (n_gen,) consigna de tensión p.u. [slack, PV...].
        """
        p_carga = np.asarray(p_carga, float)
        q_carga = np.asarray(q_carga, float)
        p_pv = np.asarray(p_pv, float)
        v_gen = np.asarray(v_gen, float)
        if self.motor == "lightsim":
            return self._flujo_lightsim(p_carga, q_carga, p_pv, v_gen)
        return self._flujo_pandapower(p_carga, q_carga, p_pv, v_gen)

    # ------------------------------------------------------------------
    # Topología (para dibujar la red en la interfaz)
    # ------------------------------------------------------------------
    def coordenadas_barras(self) -> np.ndarray:
        """(n_barras, 2) coordenadas x, y. Usa las geográficas del caso si
        existen; si no, un dibujo automático (networkx, Kamada-Kawai)."""
        import json as _json
        net = self.net
        if "geo" in net.bus and net.bus.geo.notna().all():
            return np.array([_json.loads(g)["coordinates"][:2] for g in net.bus.geo], float)
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(net.bus.index)
        g.add_edges_from(zip(net.line.from_bus, net.line.to_bus))
        g.add_edges_from(zip(net.trafo.hv_bus, net.trafo.lv_bus))
        pos = nx.kamada_kawai_layout(g)
        return np.array([pos[b] for b in net.bus.index], float)

    def ramas(self) -> list[tuple[int, int]]:
        """Barras (posición 0..n-1) de los extremos de cada línea y trafo, en el
        mismo orden que cargabilidad_ramas_pct."""
        pos = {b: i for i, b in enumerate(self.net.bus.index)}
        net = self.net
        return ([(pos[f], pos[t]) for f, t in zip(net.line.from_bus, net.line.to_bus)]
                + [(pos[h], pos[l]) for h, l in zip(net.trafo.hv_bus, net.trafo.lv_bus)])

    # ------------------------------------------------------------------
    # Economía
    # ------------------------------------------------------------------
    @staticmethod
    def costo_despacho(p_gen_mw, precios) -> float:
        """Costo horario del despacho (USD/h) = sum_i precio_i [USD/MWh] * P_i [MW]."""
        return float(np.dot(np.asarray(p_gen_mw, float), np.asarray(precios, float)))

    def despacho_orden_merito(self, demanda_mw: float, precios):
        """Despacho económico de "placa de cobre" (sin red, sin pérdidas).

        Todos arrancan en su Pmin y la demanda restante se asigna al más barato
        hasta su Pmax, luego al siguiente, etc. Es el despacho más barato
        posible si la red no existiera, por lo que su costo es una cota
        inferior (aprox.) del costo de cualquier despacho real. La recompensa
        mide el costo del agente RELATIVO a esta referencia, lo que la hace
        comparable entre escenarios con demandas y precios muy distintos.

        Devuelve (p_gen_mw, costo).
        """
        precios = np.asarray(precios, float)
        p = self.p_min.copy()
        restante = demanda_mw - p.sum()
        for i in np.argsort(precios):
            if restante <= 0:
                break
            x = min(self.p_max[i] - p[i], restante)
            p[i] += x
            restante -= x
        return p, self.costo_despacho(p, precios)

    # ------------------------------------------------------------------
    # OPF AC exacto (referencia de evaluación)
    # ------------------------------------------------------------------
    def resolver_opf(self, p_carga, q_carga, precios, con_margen: bool = False) -> dict:
        """OPF AC exacto de pandapower (pp.runopp, interior-point PIPS).

        Minimiza sum_i precio_i * P_i sujeto a: flujo AC, límites de P y Q de
        los generadores, cargabilidad <= CARGABILIDAD_MAX_PCT y tensiones en
        [TENSION_MIN_PU, TENSION_MAX_PU]. Es el "óptimo verdadero" contra el que
        se mide al agente (brecha de costo).

        con_margen=True resuelve el OPF con los mismos límites estrictos que la
        recompensa usa en entrenamiento (98 %, [0.955, 1.045]): da el MEJOR
        despacho que el agente podría aprender y, por tanto, el techo teórico
        de la recompensa.

        Devuelve dict con: convergio, p_gen (todas), v_gen (todas), costo.
        Tarda ~0.3-0.5 s, por eso NO se usa durante el entrenamiento.
        """
        if self._net_opf is None:
            n = copy.deepcopy(self.net)
            n.line["max_loading_percent"] = C.CARGABILIDAD_MAX_PCT
            if self.n_trafos:
                n.trafo["max_loading_percent"] = C.CARGABILIDAD_MAX_PCT
            n.gen["controllable"] = True
            # Costos lineales, una fila por generador.
            n.poly_cost.drop(n.poly_cost.index, inplace=True)
            pp.create_poly_cost(n, n.ext_grid.index[0], "ext_grid", cp1_eur_per_mw=1.0)
            for gi in n.gen.index:
                pp.create_poly_cost(n, gi, "gen", cp1_eur_per_mw=1.0)
            self._net_opf = n
        n = self._net_opf
        m_carg = C.MARGEN_CARGABILIDAD_PCT if con_margen else 0.0
        m_v = C.MARGEN_TENSION_PU if con_margen else 0.0
        n.line["max_loading_percent"] = C.CARGABILIDAD_MAX_PCT - m_carg
        if self.n_trafos:
            n.trafo["max_loading_percent"] = C.CARGABILIDAD_MAX_PCT - m_carg
        n.bus["min_vm_pu"] = C.TENSION_MIN_PU + m_v
        n.bus["max_vm_pu"] = C.TENSION_MAX_PU - m_v
        # Consigna de tensión de los generadores: dentro de la banda de barras
        # y del rango que también tiene el agente (VG_MIN_PU..VG_MAX_PU).
        n.gen["min_vm_pu"] = max(C.VG_MIN_PU, C.TENSION_MIN_PU + m_v)
        n.gen["max_vm_pu"] = min(C.VG_MAX_PU, C.TENSION_MAX_PU - m_v)
        n.load["p_mw"] = np.asarray(p_carga, float)
        n.load["q_mvar"] = np.asarray(q_carga, float)
        n.poly_cost["cp1_eur_per_mw"] = np.asarray(precios, float)
        try:
            pp.runopp(n, delta=1e-10, init="flat")
            ok = bool(n.OPF_converged)
        except Exception:
            ok = False
        if not ok:
            return {"convergio": False}
        p_gen = np.r_[n.res_ext_grid.p_mw.values, n.res_gen.p_mw.values]
        v_gen = n.res_bus.vm_pu.loc[self.bus_gen].values.astype(float)
        return {
            "convergio": True,
            "p_gen": p_gen,
            "v_gen": v_gen,
            "costo": self.costo_despacho(p_gen, precios),
        }
