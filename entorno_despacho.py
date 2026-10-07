"""
entorno_despacho.py
===================
Ambiente Gymnasium para que un agente PPO aprenda DESPACHO ECONÓMICO SEGURO.

-----------------------------------------------------------------------------
FORMULACIÓN (MDP de un paso / "bandido contextual")
-----------------------------------------------------------------------------
Cada EPISODIO es UN escenario operativo y dura UN paso:

    reset()  -> se sortea un escenario nuevo (cargas + precios) y se entrega la
                observación al agente.
    step(a)  -> el agente da su despacho, se resuelve el flujo de carga AC en
                pandapower, se calcula la recompensa y el episodio TERMINA.

¿Por qué un solo paso? El despacho económico de un instante es una decisión
ESTÁTICA: para unas cargas y unos precios dados existe un despacho óptimo, y no
depende de lo que se hizo antes. Lo que el agente debe aprender es la función
    (cargas, precios)  ->  despacho
y la diversidad de escenarios entre episodios es lo que lo obliga a
generalizar. (Por lo mismo, gamma y gae_lambda de PPO no tienen efecto.)

-----------------------------------------------------------------------------
OBSERVACIÓN  (lo que ve el agente)  — dimensión 2·n_cargas + n_gen
-----------------------------------------------------------------------------
    [ P_1/P_1,base, ..., P_L/P_L,base,      <- carga activa de cada barra
      Q_1/Q_1,base, ..., Q_L/Q_L,base,      <- carga reactiva de cada barra
      precio_1_norm, ..., precio_G_norm ]   <- precio de cada generador
    Cargas: relativas a su valor base del caso (≈ FD · ruido, del orden de 0.4-1.1).
    Precios: escalados a [-1, 1] con el rango [PRECIO_MIN, PRECIO_MAX].
    Todo queda en rangos del orden de la unidad, que es lo que mejor maneja la
    red neuronal (sin necesidad de VecNormalize).

-----------------------------------------------------------------------------
ACCIÓN  (lo que decide el agente), en [-1, 1]. Dos variantes (config.CAPA_BALANCE)
-----------------------------------------------------------------------------
Mapeo común:  P = Pmin + (a+1)/2 · (Pmax − Pmin)
              V = VG_MIN + (a+1)/2 · (VG_MAX − VG_MIN)
Mapear desde [-1, 1] hace que los límites de P de cada máquina se cumplan POR
CONSTRUCCIÓN: el agente no puede pedir más de Pmax ni menos de Pmin.

(A) CAPA_BALANCE = False  — dimensión n_pv + n_gen
    [ P de los PV (n_pv) ,  V de todos (n_gen) ]
    La slack no recibe consigna de P: su potencia sale del balance del flujo
    de carga. El agente tiene que "adivinar" la suma exacta para que la slack
    quede dentro de sus límites.

(B) CAPA_BALANCE = True   — dimensión 2 · n_gen
    [ P de TODOS los generadores (n_gen) ,  V de todos (n_gen) ]
    Así despacha un operador real: da consigna a todas las unidades, y la
    slack solo cubre los desvíos. Como lo que pide el agente no tiene por qué
    sumar exactamente carga + pérdidas, una CAPA DE BALANCE física lo ajusta:
        1. objetivo = demanda + pérdidas estimadas
        2. si falta potencia, cada máquina sube en proporción a su holgura
           hacia arriba (Pmax − P); si sobra, baja en proporción a su holgura
           hacia abajo (P − Pmin). Nadie se sale de sus límites.
        3. se resuelve el flujo, se miden las pérdidas reales y se repite
           (2-3 iteraciones) hasta que la slack queda donde el agente la pidió.
    El agente sigue decidiendo el REPARTO (qué máquinas generan y cuánto, y las
    tensiones), que es donde están la economía y la congestión; la capa solo
    garantiza que el despacho sea físicamente consistente con la demanda.

-----------------------------------------------------------------------------
RECOMPENSA: ver recompensa.py (documentada término a término).
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config as C
from escenarios import Escenario, GeneradorEscenarios
from recompensa import calcular_recompensa
from red_electrica import RedElectrica


class EntornoDespacho(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, semilla: int | None = None,
                 escenarios_fijos: list[Escenario] | None = None,
                 red: RedElectrica | None = None,
                 capa_balance: bool | None = None):
        """
        semilla          : semilla del generador de escenarios (reproducible).
        escenarios_fijos : si se da, el ambiente recorre esta lista en orden
                           (validación / evaluación con escenarios idénticos
                           para todos los modelos). Si es None, cada reset
                           sortea un escenario nuevo (entrenamiento).
        red              : permite compartir una RedElectrica ya construida.
        capa_balance     : variante de acción (ver docstring del módulo); por
                           defecto config.CAPA_BALANCE.
        """
        super().__init__()
        self.red = red if red is not None else RedElectrica()
        self.generador = GeneradorEscenarios(self.red, semilla)
        self.escenarios_fijos = escenarios_fijos
        self._i_fijo = 0
        self.capa_balance = C.CAPA_BALANCE if capa_balance is None else bool(capa_balance)

        r = self.red
        self.n_obs = 2 * r.n_cargas + r.n_gen
        self.n_p = r.n_gen if self.capa_balance else r.n_pv   # cuántas P decide el agente
        self.n_acc = self.n_p + r.n_gen
        self.observation_space = spaces.Box(-5.0, 5.0, (self.n_obs,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_acc,), np.float32)

        # Bases para normalizar (se evita dividir por cero si alguna carga
        # base es 0).
        self._base_p = np.where(np.abs(r.p_carga_base) > 1e-9, r.p_carga_base, 1.0)
        self._base_q = np.where(np.abs(r.q_carga_base) > 1e-9, r.q_carga_base, 1.0)

        self.escenario: Escenario | None = None

    # ------------------------------------------------------------------
    # Codificación de observación y acción (públicas: las usan la
    # evaluación y la interfaz para no duplicar lógica)
    # ------------------------------------------------------------------
    def observacion(self, esc: Escenario) -> np.ndarray:
        p = esc.p_carga / self._base_p
        q = esc.q_carga / self._base_q
        precios = 2.0 * (esc.precios - C.PRECIO_MIN) / (C.PRECIO_MAX - C.PRECIO_MIN) - 1.0
        obs = np.concatenate([p, q, precios]).astype(np.float32)
        return np.clip(obs, -5.0, 5.0)

    def decodificar_accion(self, accion) -> tuple[np.ndarray, np.ndarray]:
        """Acción normalizada [-1, 1] -> (P pedidas en MW, V de todos en p.u.).

        Con capa de balance las P son de TODOS los generadores (slack incluida);
        sin ella, solo de los PV. Son las consignas "crudas": la función que da
        el despacho final es aplicar_accion().
        """
        r = self.red
        a = np.clip(np.asarray(accion, dtype=float), -1.0, 1.0)
        a_p, a_v = a[:self.n_p], a[self.n_p:]
        lo = r.p_min if self.capa_balance else r.p_min[1:]
        hi = r.p_max if self.capa_balance else r.p_max[1:]
        p = lo + (a_p + 1.0) / 2.0 * (hi - lo)
        v_gen = C.VG_MIN_PU + (a_v + 1.0) / 2.0 * (C.VG_MAX_PU - C.VG_MIN_PU)
        return p, v_gen

    def codificar_accion(self, p_gen_todos, v_gen) -> np.ndarray:
        """Inversa de decodificar_accion. Recibe la P de TODOS los generadores
        [slack, PV...] (como la da el OPF) y descarta la de la slack si no hay
        capa de balance. Útil para pruebas."""
        r = self.red
        p = np.asarray(p_gen_todos, float)
        if not self.capa_balance:
            p, lo, hi = p[1:], r.p_min[1:], r.p_max[1:]
        else:
            lo, hi = r.p_min, r.p_max
        a_p = 2.0 * (p - lo) / (hi - lo) - 1.0
        a_v = 2.0 * (np.asarray(v_gen) - C.VG_MIN_PU) / (C.VG_MAX_PU - C.VG_MIN_PU) - 1.0
        return np.clip(np.r_[a_p, a_v], -1, 1).astype(np.float32)

    def _proyectar_balance(self, p: np.ndarray, objetivo_mw: float) -> np.ndarray:
        """Ajusta las P pedidas para que sumen `objetivo_mw` sin salirse de
        [Pmin, Pmax]: el faltante (o sobrante) se reparte en proporción a la
        holgura de cada máquina hacia arriba (o hacia abajo)."""
        r = self.red
        delta = objetivo_mw - p.sum()
        if delta >= 0:
            holgura = r.p_max - p
        else:
            holgura = p - r.p_min
        total = holgura.sum()
        if total <= 1e-9:
            return p.copy()
        fraccion = min(1.0, abs(delta) / total)  # si no alcanza, todos al límite
        return p + np.sign(delta) * holgura * fraccion

    def aplicar_accion(self, esc: Escenario, accion):
        """Acción -> despacho final + flujo de carga. ÚNICA forma de convertir
        una acción en despacho (la usan el entrenamiento, la evaluación y la
        interfaz). Devuelve (p_pv, v_gen, resultado_flujo)."""
        p, v_gen = self.decodificar_accion(accion)
        if not self.capa_balance:
            res = self.red.resolver_flujo(esc.p_carga, esc.q_carga, p, v_gen)
            return p, v_gen, res
        demanda = esc.demanda_total_mw
        perdidas = C.PERDIDAS_ESTIMADAS_PU * demanda
        res = None
        for _ in range(C.ITERACIONES_BALANCE):
            p_bal = self._proyectar_balance(p, demanda + perdidas)
            res = self.red.resolver_flujo(esc.p_carga, esc.q_carga, p_bal[1:], v_gen)
            if not res.convergio or abs(res.perdidas_mw - perdidas) < 1e-3:
                break
            perdidas = res.perdidas_mw
        return p_bal[1:], v_gen, res

    def evaluar_despacho(self, esc: Escenario, accion):
        """Aplica una acción a un escenario: despacho + flujo de carga + recompensa.
        Devuelve (recompensa, info, resultado_flujo, p_pv, v_gen)."""
        p_pv, v_gen, res = self.aplicar_accion(esc, accion)
        r, info = calcular_recompensa(res, esc.precios, esc.demanda_total_mw, self.red)
        return r, info, res, p_pv, v_gen

    # ------------------------------------------------------------------
    # API Gymnasium
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.generador.reiniciar_semilla(seed)
        if options and "escenario" in options:
            self.escenario = options["escenario"]
        elif self.escenarios_fijos is not None:
            self.escenario = self.escenarios_fijos[self._i_fijo % len(self.escenarios_fijos)]
            self._i_fijo += 1
        else:
            self.escenario = self.generador.muestrear()
        return self.observacion(self.escenario), {"fd": self.escenario.fd}

    def step(self, accion):
        r, info, _, _, _ = self.evaluar_despacho(self.escenario, accion)
        obs = self.observacion(self.escenario)  # episodio terminado; obs final
        return obs, float(r), True, False, info
