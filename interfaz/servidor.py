"""
interfaz/servidor.py
====================
Servidor web (Flask) del panel de despacho. Uso:

    python main.py interfaz        ->  abrir http://127.0.0.1:5050

El panel (panel.html) permite:
    * mover el factor de demanda y el precio de cada generador; en cada cambio
      el agente PPO despacha (en milisegundos) y se muestra el flujo de carga
      resultante: despacho, cargabilidades, tensiones, reactiva y costo;
    * comparar contra el OPF AC exacto de pandapower (brecha de costo);
    * comparar contra el despacho por orden de mérito sin red (para ver qué
      pasaría despachando solo por precio);
    * generar escenarios aleatorios como los del entrenamiento;
    * simular un día completo (24 h) con demanda y precios variables.

Arquitectura: el navegador solo envía (FD, precios, perfil de cargas) y dibuja
lo que responde el servidor. Toda la física (pandapower) y el agente viven en
este proceso. Un candado serializa las peticiones porque el modelo de flujo de
carga no es seguro para hilos concurrentes.
"""

from __future__ import annotations

import sys
import threading
import warnings
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402
from agente import cargar_modelo  # noqa: E402
from entorno_despacho import EntornoDespacho  # noqa: E402
from escenarios import Escenario  # noqa: E402
from recompensa import calcular_recompensa, calcular_violaciones  # noqa: E402

warnings.filterwarnings("ignore")

app = Flask(__name__)
_candado = threading.Lock()
_entorno = EntornoDespacho(semilla=0)
_red = _entorno.red
try:
    _modelo = cargar_modelo(entorno=_entorno)
    _error_modelo = None
except (FileNotFoundError, ValueError) as ex:
    _modelo = None
    _error_modelo = str(ex)


# -----------------------------------------------------------------------------
# Construcción del escenario a partir de lo que manda el navegador
# -----------------------------------------------------------------------------
def _escenario(datos: dict) -> Escenario:
    """FD + precios + perfil de cargas.

    perfil = "uniforme"   -> todas las cargas escaladas igual por FD.
    perfil = <entero>     -> además, cada carga con un ruido individual
                             reproducible (semilla), como en el entrenamiento.
    """
    fd = float(datos.get("fd", 0.7))
    precios = np.asarray(datos.get("precios"), float)
    perfil = datos.get("perfil", "uniforme")
    p = _red.p_carga_base * fd
    q = _red.q_carga_base * fd
    if perfil != "uniforme":
        rng = np.random.default_rng(int(perfil))
        ruido = np.clip(rng.normal(1.0, C.SIGMA_CARGA, _red.n_cargas),
                        1 - C.RECORTE_CARGA, 1 + C.RECORTE_CARGA)
        ruido_q = np.clip(rng.normal(1.0, C.SIGMA_REACTIVA, _red.n_cargas),
                          1 - C.RECORTE_CARGA, 1 + C.RECORTE_CARGA)
        p, q = p * ruido, q * ruido * ruido_q
    return Escenario(p, q, precios, fd)


def _nativo(x):
    """Convierte tipos de numpy a tipos nativos de Python (para JSON)."""
    if isinstance(x, dict):
        return {k: _nativo(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_nativo(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _evaluar(esc: Escenario, p_pv, v_gen) -> dict:
    """Flujo de carga + métricas de un despacho, listo para JSON."""
    res = _red.resolver_flujo(esc.p_carga, esc.q_carga, p_pv, v_gen)
    if not res.convergio:
        return {"convergio": False}
    viol = calcular_violaciones(res, _red, con_margen=False)
    r, info = calcular_recompensa(res, esc.precios, esc.demanda_total_mw, _red)
    return {
        "convergio": True,
        "p_gen": res.p_gen_mw.round(3).tolist(),
        "q_gen": res.q_gen_mvar.round(3).tolist(),
        "v_gen": np.asarray(v_gen).round(4).tolist(),
        "vm": res.vm_pu.round(4).tolist(),
        "cargabilidad": res.cargabilidad_ramas_pct.round(2).tolist(),
        "costo": round(_red.costo_despacho(res.p_gen_mw, esc.precios), 2),
        "perdidas": round(res.perdidas_mw, 3),
        "seguro": bool(viol["seguro"]),
        "violaciones": _nativo(viol),
        "recompensa": round(float(r), 4),
        "sobrecosto_merito_pct": round(100 * info.get("sobrecosto", float("nan")), 2),
    }


def _despacho_agente(esc: Escenario):
    """(p_pv, v_gen) finales del agente (incluye la capa de balance si aplica)."""
    obs = _entorno.observacion(esc)
    accion, _ = _modelo.predict(obs, deterministic=True)
    p_pv, v_gen, _ = _entorno.aplicar_accion(esc, accion)
    return p_pv, v_gen


# -----------------------------------------------------------------------------
# Rutas
# -----------------------------------------------------------------------------
@app.route("/")
def panel():
    return send_from_directory(Path(__file__).parent, "panel.html")


@app.route("/api/info")
def info():
    """Datos estáticos de la red para que el panel arme sus gráficas."""
    coords = _red.coordenadas_barras()
    return jsonify({
        "caso": C.CASO_RED,
        "modelo_cargado": _modelo is not None,
        "error_modelo": _error_modelo,
        "generadores": _red.nombres_gen,
        "bus_gen": _red.bus_gen.tolist(),
        "p_min": _red.p_min.tolist(), "p_max": _red.p_max.tolist(),
        "q_min": _red.q_min.tolist(), "q_max": _red.q_max.tolist(),
        "ramas": _red.nombres_ramas,
        "extremos_ramas": _red.ramas(),
        "n_barras": _red.n_barras,
        "coordenadas": coords.round(4).tolist(),
        "bus_carga": _red.bus_carga.tolist(),
        "demanda_base_mw": float(_red.p_carga_base.sum()),
        "fd_min": C.FD_MIN, "fd_max": C.FD_MAX,
        "precio_min": C.PRECIO_MIN, "precio_max": C.PRECIO_MAX,
        "v_min": C.TENSION_MIN_PU, "v_max": C.TENSION_MAX_PU,
        "carga_max": C.CARGABILIDAD_MAX_PCT,
        "margen_carga": C.MARGEN_CARGABILIDAD_PCT,
    })


@app.route("/api/despachar", methods=["POST"])
def despachar():
    """Despacho del agente (+ OPF y orden de mérito si se piden)."""
    if _modelo is None:
        return jsonify({"error": _error_modelo}), 503
    datos = request.get_json(force=True)
    with _candado:
        esc = _escenario(datos)
        _, costo_merito = _red.despacho_orden_merito(esc.demanda_total_mw, esc.precios)
        p_pv, v_gen = _despacho_agente(esc)
        salida = {
            "fd": esc.fd,
            "demanda_mw": round(esc.demanda_total_mw, 3),
            "costo_merito_placa_cobre": round(costo_merito, 2),
            "agente": _evaluar(esc, p_pv, v_gen),
        }
        if datos.get("comparar_merito", True):
            pm, _ = _red.despacho_orden_merito(esc.demanda_total_mw, esc.precios)
            salida["merito"] = _evaluar(esc, pm[1:], np.ones(_red.n_gen))
        if datos.get("comparar_opf", False):
            o = _red.resolver_opf(esc.p_carga, esc.q_carga, esc.precios)
            if o["convergio"]:
                salida["opf"] = _evaluar(esc, o["p_gen"][1:], o["v_gen"])
                ca, co = salida["agente"].get("costo"), salida["opf"]["costo"]
                if ca is not None:
                    salida["brecha_opf_pct"] = round(100 * (ca - co) / co, 3)
            else:
                salida["opf"] = {"convergio": False}
    return jsonify(_nativo(salida))


@app.route("/api/aleatorio")
def aleatorio():
    """Escenario aleatorio como los del entrenamiento (FD, precios y ruido de
    cargas). Devuelve los controles para que el panel los refleje."""
    rng = np.random.default_rng()
    return jsonify({
        "fd": float(rng.uniform(C.FD_MIN, C.FD_MAX)),
        "precios": rng.uniform(C.PRECIO_MIN, C.PRECIO_MAX, _red.n_gen).round(1).tolist(),
        "perfil": int(rng.integers(1, 1_000_000)),
    })


@app.route("/api/simular_dia", methods=["POST"])
def simular_dia():
    """24 despachos horarios con demanda y precios variables."""
    if _modelo is None:
        return jsonify({"error": _error_modelo}), 503
    datos = request.get_json(force=True)
    precios_base = np.asarray(datos.get("precios"), float)
    comparar_opf = bool(datos.get("comparar_opf", True))
    desfases = np.linspace(0, 2 * np.pi, _red.n_gen, endpoint=False)
    horas = []
    with _candado:
        for h, nivel in enumerate(C.CURVA_DEMANDA_DIARIA):
            fd = C.FD_MIN + nivel * (C.FD_MAX - C.FD_MIN)
            precios = np.clip(
                precios_base * (1 + C.VARIACION_PRECIO_DIA * np.sin(2 * np.pi * h / 24 + desfases)),
                C.PRECIO_MIN, C.PRECIO_MAX)
            esc = _escenario({"fd": fd, "precios": precios, "perfil": datos.get("perfil", "uniforme")})
            p_pv, v_gen = _despacho_agente(esc)
            ag = _evaluar(esc, p_pv, v_gen)
            fila = {"hora": h, "fd": round(fd, 3), "demanda_mw": round(esc.demanda_total_mw, 2),
                    "precios": precios.round(1).tolist(), "agente": ag}
            pm, _ = _red.despacho_orden_merito(esc.demanda_total_mw, esc.precios)
            me = _evaluar(esc, pm[1:], np.ones(_red.n_gen))
            fila["merito"] = {k: me.get(k) for k in ("convergio", "costo", "seguro")}
            fila["merito"]["carga_max"] = max(me["cargabilidad"]) if me.get("convergio") else None
            if comparar_opf:
                o = _red.resolver_opf(esc.p_carga, esc.q_carga, esc.precios)
                fila["opf_costo"] = round(o["costo"], 2) if o["convergio"] else None
            horas.append(fila)
    return jsonify(_nativo({"horas": horas}))


def ejecutar(host: str = C.INTERFAZ_HOST, puerto: int = C.INTERFAZ_PUERTO):
    if _modelo is None:
        print(f"AVISO: {_error_modelo}\nEl panel abrirá pero no podrá despachar.")
    print(f"\nPanel de despacho en  http://{host}:{puerto}   (Ctrl+C para detener)\n")
    app.run(host=host, port=puerto, debug=False, threaded=True)


if __name__ == "__main__":
    ejecutar()
