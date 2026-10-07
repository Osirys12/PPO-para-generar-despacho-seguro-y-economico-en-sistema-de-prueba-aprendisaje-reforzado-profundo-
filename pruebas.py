"""
pruebas.py
==========
Verificaciones automáticas del proyecto. Ejecutar con:

    python main.py probar               (rápidas, ~20 s)
    python main.py probar --completo    (incluye estudio de factibilidad con OPF, ~2 min)

Cada prueba imprime OK o FALLA con el detalle. Si alguna falla, el programa
termina con código de error 1.

Pruebas:
  1. El motor rápido (lightsim2grid) da EXACTAMENTE lo mismo que pp.runpp.
  2. El ambiente cumple la API de Gymnasium y de Stable-Baselines3.
  3. Codificar/decodificar acciones es consistente y respeta Pmin/Pmax.
     Con capa de balance, la slack queda dentro de sus límites.
  4. La recompensa ordena correctamente despachos de calidad conocida:
       OPF con márgenes > orden de mérito sin red > despacho aleatorio.
  5. Los pesos respetan la jerarquía: BONO_SEGURO > W_COSTO·máx(sobrecosto
     del OPF seguro), es decir, el máximo de la recompensa es el OPF seguro.
  6. Velocidad del ambiente (pasos por segundo).
  7. Si existe el modelo entrenado: carga y despacha.
  8. Si existe el modelo entrenado: la interfaz web responde.
  9. (--completo) Estudio de factibilidad del rango de demanda con el OPF.
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np

import config as C
from entorno_despacho import EntornoDespacho
from escenarios import GeneradorEscenarios
from red_electrica import RedElectrica

warnings.filterwarnings("ignore")
_fallas: list[str] = []


def _reporte(nombre: str, ok: bool, detalle: str = ""):
    print(f"  [{'OK   ' if ok else 'FALLA'}] {nombre}" + (f"  — {detalle}" if detalle else ""))
    if not ok:
        _fallas.append(nombre)


def prueba_motor_vs_pandapower(n: int = 40):
    """Mismos escenarios y despachos aleatorios en ambos motores."""
    rapida = RedElectrica(motor="lightsim")
    clasica = RedElectrica(motor="pandapower")
    gen = GeneradorEscenarios(rapida, 11)
    rng = np.random.default_rng(11)
    dv = dl = dp = dq = 0.0
    n_ok = 0
    for _ in range(n):
        e = gen.muestrear()
        p_pv = rng.uniform(rapida.p_min[1:], rapida.p_max[1:] * 0.7)
        v = rng.uniform(0.97, 1.04, rapida.n_gen)
        a = rapida.resolver_flujo(e.p_carga, e.q_carga, p_pv, v)
        b = clasica.resolver_flujo(e.p_carga, e.q_carga, p_pv, v)
        if a.convergio != b.convergio:
            _reporte("Motor lightsim == pp.runpp", False, "convergencia distinta")
            return
        if not a.convergio:
            continue
        n_ok += 1
        dv = max(dv, np.abs(a.vm_pu - b.vm_pu).max())
        dl = max(dl, np.abs(a.cargabilidad_ramas_pct - b.cargabilidad_ramas_pct).max())
        dp = max(dp, np.abs(a.p_gen_mw - b.p_gen_mw).max())
        dq = max(dq, np.abs(a.q_gen_mvar - b.q_gen_mvar).max())
    ok = n_ok > 0 and max(dv, dp / 100, dq / 100, dl / 100) < 1e-6
    _reporte("Motor lightsim == pp.runpp", ok,
             f"{n_ok} flujos; máx. dif: V={dv:.1e} pu, carga={dl:.1e} %, P={dp:.1e} MW, Q={dq:.1e} MVAr")


def prueba_api_gymnasium():
    from gymnasium.utils.env_checker import check_env as check_gym
    from stable_baselines3.common.env_checker import check_env as check_sb3
    try:
        check_gym(EntornoDespacho(semilla=0), skip_render_check=True)
        check_sb3(EntornoDespacho(semilla=0), warn=False)
        env = EntornoDespacho(semilla=0)
        obs, _ = env.reset()
        _, r, term, trunc, info = env.step(env.action_space.sample())
        ok = obs.shape == env.observation_space.shape and term and not trunc
        _reporte("API Gymnasium / SB3", ok,
                 f"obs={env.observation_space.shape}, acción={env.action_space.shape}, episodio de 1 paso")
    except Exception as ex:  # noqa: BLE001
        _reporte("API Gymnasium / SB3", False, repr(ex))


def prueba_codificacion_accion():
    """Para las dos variantes de acción: codificar/decodificar es consistente y
    los límites de P se respetan aunque la acción venga fuera de [-1, 1]."""
    ok = True
    for capa in (False, True):
        env = EntornoDespacho(semilla=0, capa_balance=capa)
        r = env.red
        rng = np.random.default_rng(0)
        p = rng.uniform(r.p_min, r.p_max)
        v = rng.uniform(C.VG_MIN_PU, C.VG_MAX_PU, r.n_gen)
        p2, v2 = env.decodificar_accion(env.codificar_accion(p, v))
        p_esp = p if capa else p[1:]
        pe, _ = env.decodificar_accion(np.full(env.n_acc, 5.0))  # fuera de rango a propósito
        hi = r.p_max if capa else r.p_max[1:]
        ok &= (np.allclose(p_esp, p2, atol=1e-4) and np.allclose(v, v2, atol=1e-6)
               and np.allclose(pe, hi))
    _reporte("Codificación de acciones y límites de P (ambas variantes)", ok)


def prueba_capa_balance(n: int = 300):
    """Con capa de balance, la slack debe quedar dentro de sus límites y donde
    la pidió el agente (salvo que la demanda no quepa), para acciones al azar."""
    env = EntornoDespacho(semilla=3, capa_balance=True)
    r = env.red
    rng = np.random.default_rng(3)
    fuera, err = 0, []
    for _ in range(n):
        env.reset()
        a = rng.uniform(-1, 1, env.n_acc)
        p_pedida, _ = env.decodificar_accion(a)
        p_pv, _, res = env.aplicar_accion(env.escenario, a)
        if not res.convergio:
            continue
        ps = res.p_gen_mw[0]
        fuera += not (r.p_min[0] - 1e-3 <= ps <= r.p_max[0] + 1e-3)
        # Consigna de la slack que resulta de balancear con las pérdidas REALES:
        # si la iteración de pérdidas convergió, la slack real coincide con ella.
        consigna = env._proyectar_balance(p_pedida, env.escenario.demanda_total_mw + res.perdidas_mw)[0]
        err.append(abs(ps - consigna))
    _reporte("Capa de balance: slack dentro de límites", fuera == 0,
             f"{n} acciones al azar, slack fuera de límites: {fuera}, "
             f"desvío máx. slack vs consigna: {max(err):.1e} MW")


def prueba_orden_recompensa(n: int = 25):
    env = EntornoDespacho(semilla=0)
    red = env.red
    gen = GeneradorEscenarios(red, 5)
    rng = np.random.default_rng(5)
    r_opf, r_mer, r_ale = [], [], []
    for _ in range(n):
        e = gen.muestrear()
        o = red.resolver_opf(e.p_carga, e.q_carga, e.precios, con_margen=True)
        if not o["convergio"]:
            continue
        r_opf.append(env.evaluar_despacho(e, env.codificar_accion(o["p_gen"], o["v_gen"]))[0])
        pm, _ = red.despacho_orden_merito(e.demanda_total_mw * 1.02, e.precios)
        r_mer.append(env.evaluar_despacho(e, env.codificar_accion(pm, np.ones(red.n_gen)))[0])
        r_ale.append(env.evaluar_despacho(e, rng.uniform(-1, 1, env.n_acc))[0])
    a, b, c = np.mean(r_opf), np.mean(r_mer), np.mean(r_ale)
    _reporte("Recompensa: OPF > mérito sin red > aleatorio", a > b > c,
             f"OPF={a:.2f}  mérito={b:.2f}  aleatorio={c:.2f}  (n={len(r_opf)})")


def prueba_jerarquia_pesos(n: int = 40):
    """BONO_SEGURO > W_COSTO · máx(sobrecosto del OPF seguro): garantiza que en
    cada escenario el máximo de la recompensa sea el despacho seguro más barato."""
    red = RedElectrica()
    gen = GeneradorEscenarios(red, 99)
    xs = []
    for _ in range(n):
        e = gen.muestrear()
        o = red.resolver_opf(e.p_carga, e.q_carga, e.precios, con_margen=True)
        if o["convergio"]:
            _, cm = red.despacho_orden_merito(e.demanda_total_mw, e.precios)
            xs.append(o["costo"] / cm - 1)
    peor = C.W_COSTO * max(xs)
    _reporte("Jerarquía: bono > costo del OPF seguro", peor < C.BONO_SEGURO,
             f"W_COSTO·máx(sobrecosto OPF) = {C.W_COSTO}·{max(xs):.3f} = {peor:.2f} < bono {C.BONO_SEGURO}")


def prueba_velocidad(n: int = 3000):
    env = EntornoDespacho(semilla=0)
    t = time.perf_counter()
    for _ in range(n):
        env.reset()
        env.step(env.action_space.sample())
    pps = n / (time.perf_counter() - t)
    _reporte("Velocidad del ambiente", pps > 200, f"{pps:,.0f} pasos/s ({C.MOTOR_FLUJO})")


def prueba_modelo_entrenado():
    if not C.RUTA_MODELO_FINAL.exists():
        print("  [--   ] Modelo entrenado: aún no existe (se omite)")
        return
    from agente import cargar_modelo, despachar
    env = EntornoDespacho(semilla=0)
    try:
        modelo = cargar_modelo(entorno=env)
    except ValueError as ex:
        _reporte("Modelo entrenado compatible con config", False, str(ex))
        return
    e = GeneradorEscenarios(env.red, 1).muestrear()
    d = despachar(modelo, env, e)
    ok = d["resultado"].convergio and d["p_pv"].shape == (env.red.n_pv,)
    _reporte("Modelo entrenado carga y despacha", ok,
             f"seguro={d['violaciones_reales']['seguro']}, costo={d['costo']:.0f} USD/h")


def prueba_interfaz():
    """Levanta el servidor Flask en modo prueba (sin abrir puerto) y llama a
    sus rutas principales."""
    if not C.RUTA_MODELO_FINAL.exists():
        print("  [--   ] Interfaz web: requiere el modelo entrenado (se omite)")
        return
    try:
        from interfaz.servidor import app
        cli = app.test_client()
        info = cli.get("/api/info").get_json()
        precios = [60.0] * len(info["generadores"])
        d = cli.post("/api/despachar", json={"fd": 0.7, "precios": precios, "perfil": 123,
                                              "comparar_opf": True, "comparar_merito": True}).get_json()
        html = cli.get("/").status_code
        ok = (html == 200 and d["agente"]["convergio"] and "opf" in d and "merito" in d
              and len(d["agente"]["cargabilidad"]) == len(info["ramas"]))
        _reporte("Interfaz web (rutas /, /api/info, /api/despachar)", ok,
                 f"agente seguro={d['agente']['seguro']}, brecha vs OPF={d.get('brecha_opf_pct', float('nan')):.2f} %")
    except Exception as ex:  # noqa: BLE001
        _reporte("Interfaz web", False, repr(ex))


def estudio_factibilidad(n: int = 240, fd_min: float = 0.45, fd_max: float = 1.05):
    """Repite el estudio con el que se eligió [FD_MIN, FD_MAX]: % de escenarios
    para los que el OPF exacto encuentra un despacho seguro, por franja de FD."""
    print(f"\n  Estudio de factibilidad (OPF exacto, {n} escenarios, FD {fd_min}-{fd_max}):")
    red = RedElectrica()
    gen = GeneradorEscenarios(red, 0)
    fd_orig = (C.FD_MIN, C.FD_MAX)
    C.FD_MIN, C.FD_MAX = fd_min, fd_max
    filas = []
    for _ in range(n):
        e = gen.muestrear()
        filas.append((e.fd, red.resolver_opf(e.p_carga, e.q_carga, e.precios)["convergio"]))
    C.FD_MIN, C.FD_MAX = fd_orig
    f = np.array(filas, float)
    bordes = np.arange(fd_min, fd_max + 1e-9, 0.1)
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        m = (f[:, 0] >= lo) & (f[:, 0] < hi)
        if m.any():
            print(f"     FD [{lo:.2f}, {hi:.2f}):  {100 * f[m, 1].mean():5.1f} % factible  (n={m.sum()})")
    print(f"     Rango configurado: FD_MIN={C.FD_MIN}, FD_MAX={C.FD_MAX}")


def ejecutar(completo: bool = False) -> int:
    print("\nPRUEBAS DEL PROYECTO DE DESPACHO PPO")
    print("=" * 60)
    prueba_motor_vs_pandapower()
    prueba_api_gymnasium()
    prueba_codificacion_accion()
    prueba_capa_balance()
    prueba_orden_recompensa()
    prueba_jerarquia_pesos()
    prueba_velocidad()
    prueba_modelo_entrenado()
    prueba_interfaz()
    if completo:
        estudio_factibilidad()
    print("=" * 60)
    if _fallas:
        print(f"FALLARON {len(_fallas)} prueba(s): {', '.join(_fallas)}")
        return 1
    print("Todas las pruebas pasaron.")
    return 0


if __name__ == "__main__":
    sys.exit(ejecutar("--completo" in sys.argv))
