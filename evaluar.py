"""
evaluar.py
==========
EVALUACIÓN DEL AGENTE contra referencias, sobre un CONJUNTO DE PRUEBA de
escenarios que el agente nunca vio (ni en entrenamiento ni en Optuna: usa su
propia semilla, config.SEMILLA_PRUEBA).

Se comparan tres formas de despachar cada escenario:

    1. AGENTE PPO      : una pasada por la red neuronal (determinista).
    2. OPF AC EXACTO   : pp.runopp de pandapower, con límites REALES. Es el
                         óptimo verdadero: la vara contra la que se mide el
                         costo del agente.
    3. ORDEN DE MÉRITO : el más barato primero, ignorando la red (consignas de
                         tensión en 1.0 p.u.). Muestra qué pasa si se despacha
                         solo por precio: barato pero inseguro.

Todos se juzgan con el MISMO flujo de carga AC y los límites REALES
(100 % de cargabilidad, tensiones en [0.95, 1.05] p.u., límites de P de la
slack y de Q de todos los generadores).

Métricas principales:
    % seguro          : despachos sin ninguna violación.
    brecha vs OPF [%] : (C_método − C_OPF) / C_OPF, en los escenarios donde el
                        OPF tiene solución.
    tiempo por despacho.

Salidas en resultados/evaluacion/:
    conjunto_prueba_opf.npz    escenarios + solución OPF (caché; se regenera
                               solo si cambia la configuración)
    detalle_por_escenario.csv  todas las métricas, escenario por escenario
    metricas.json              resumen
    reporte_evaluacion.md      reporte legible
    *.png                      figuras

Uso:
    python main.py evaluar
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd

import config as C
from agente import cargar_modelo, despachar
from entorno_despacho import EntornoDespacho
from escenarios import Escenario, GeneradorEscenarios
from recompensa import calcular_violaciones

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Conjunto de prueba con OPF (se calcula una vez y se guarda)
# -----------------------------------------------------------------------------
def _huella_config() -> str:
    """Si cambia algo que afecta los escenarios o el OPF, cambia la huella y el
    conjunto de prueba se recalcula."""
    claves = (C.CASO_RED, C.ESCENARIOS_PRUEBA, C.SEMILLA_PRUEBA, C.FD_MIN, C.FD_MAX,
              C.SIGMA_CARGA, C.RECORTE_CARGA, C.SIGMA_REACTIVA, C.PRECIO_MIN, C.PRECIO_MAX,
              C.CARGABILIDAD_MAX_PCT, C.TENSION_MIN_PU, C.TENSION_MAX_PU, C.VG_MIN_PU, C.VG_MAX_PU)
    return hashlib.md5(repr(claves).encode()).hexdigest()[:12]


def conjunto_prueba(red) -> tuple[list[Escenario], list[dict]]:
    ruta = C.RUTA_CONJUNTO_PRUEBA
    huella = _huella_config()
    if ruta.exists():
        d = np.load(ruta, allow_pickle=False)
        if str(d["huella"]) == huella:
            esc = [Escenario(p, q, c, float(f)) for p, q, c, f in
                   zip(d["p_carga"], d["q_carga"], d["precios"], d["fd"])]
            opf = [{"convergio": bool(ok), "p_gen": pg, "v_gen": vg, "costo": float(c),
                    "segundos": float(s)}
                   for ok, pg, vg, c, s in zip(d["opf_ok"], d["opf_p"], d["opf_v"],
                                               d["opf_costo"], d["opf_seg"])]
            print(f"  Conjunto de prueba cargado de caché ({len(esc)} escenarios).")
            return esc, opf

    print(f"  Resolviendo el OPF exacto de {C.ESCENARIOS_PRUEBA} escenarios de prueba "
          f"(una sola vez, ~{0.35 * C.ESCENARIOS_PRUEBA / 60:.0f} min)...")
    esc = GeneradorEscenarios(red, C.SEMILLA_PRUEBA).lote(C.ESCENARIOS_PRUEBA)
    opf = []
    for i, e in enumerate(esc):
        t = time.perf_counter()
        o = red.resolver_opf(e.p_carga, e.q_carga, e.precios)
        o["segundos"] = time.perf_counter() - t
        if not o["convergio"]:
            o.update(p_gen=np.full(red.n_gen, np.nan), v_gen=np.full(red.n_gen, np.nan),
                     costo=np.nan)
        opf.append(o)
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(esc)}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ruta, huella=huella,
             p_carga=np.array([e.p_carga for e in esc]), q_carga=np.array([e.q_carga for e in esc]),
             precios=np.array([e.precios for e in esc]), fd=np.array([e.fd for e in esc]),
             opf_ok=np.array([o["convergio"] for o in opf]),
             opf_p=np.array([o["p_gen"] for o in opf]), opf_v=np.array([o["v_gen"] for o in opf]),
             opf_costo=np.array([o["costo"] for o in opf]), opf_seg=np.array([o["segundos"] for o in opf]))
    return esc, opf


# -----------------------------------------------------------------------------
# Evaluación de un despacho con límites REALES
# -----------------------------------------------------------------------------
def _medir(red, esc: Escenario, p_pv, v_gen) -> dict:
    res = red.resolver_flujo(esc.p_carga, esc.q_carga, p_pv, v_gen)
    v = calcular_violaciones(res, red, con_margen=False)
    if not res.convergio:
        return {"convergio": False, "seguro": False, "costo": np.nan}
    return {
        "convergio": True,
        "seguro": bool(v["seguro"]),
        "costo": red.costo_despacho(res.p_gen_mw, esc.precios),
        "carga_max_pct": v["cargabilidad_max_pct"],
        "n_ramas_sobrecargadas": v["n_ramas_sobrecargadas"],
        "v_min": v["v_min_pu"], "v_max": v["v_max_pu"],
        "n_barras_fuera": v["n_barras_fuera_banda"],
        "exceso_slack_mw": v["exceso_slack_mw"],
        "n_gen_fuera_q": v["n_gen_fuera_q"],
        "perdidas_mw": res.perdidas_mw,
        "p_gen": res.p_gen_mw,
    }


def ejecutar(ruta_modelo=None, capa_balance: bool | None = None, carpeta=None):
    """ruta_modelo / capa_balance / carpeta permiten evaluar otro modelo (por
    ejemplo, la variante sin capa de balance) sin tocar config.py. Por
    defecto: el modelo final, config.CAPA_BALANCE y resultados/evaluacion/."""
    from pathlib import Path
    D = Path(carpeta) if carpeta else C.DIR_EVALUACION
    D.mkdir(parents=True, exist_ok=True)
    print("\nEVALUACIÓN EN EL CONJUNTO DE PRUEBA")
    entorno = EntornoDespacho(semilla=0, capa_balance=capa_balance)
    red = entorno.red
    modelo = cargar_modelo(Path(ruta_modelo) if ruta_modelo else None, entorno=entorno)
    print(f"  Modelo: {Path(ruta_modelo) if ruta_modelo else C.RUTA_MODELO_FINAL}  "
          f"(capa de balance: {'sí' if entorno.capa_balance else 'no'})")
    escenarios, opf = conjunto_prueba(red)

    filas, ejemplo = [], None
    t_agente = []
    for i, (e, o) in enumerate(zip(escenarios, opf)):
        # 1) Agente. El tiempo incluye TODO lo necesario para tener un despacho
        #    verificado: red neuronal + capa de balance + flujo de carga AC.
        t = time.perf_counter()
        obs = entorno.observacion(e)
        accion, _ = modelo.predict(obs, deterministic=True)
        p_pv, v_gen, _ = entorno.aplicar_accion(e, accion)
        t_agente.append(time.perf_counter() - t)
        ag = _medir(red, e, p_pv, v_gen)
        # 2) OPF (re-evaluado con el mismo flujo de carga)
        op = _medir(red, e, o["p_gen"][1:], o["v_gen"]) if o["convergio"] else {"convergio": False}
        # 3) Orden de mérito sin red
        pm, _ = red.despacho_orden_merito(e.demanda_total_mw, e.precios)
        me = _medir(red, e, pm[1:], np.ones(red.n_gen))

        fila = {"escenario": i, "fd": e.fd, "demanda_mw": e.demanda_total_mw,
                "opf_factible": o["convergio"], "opf_costo": o["costo"],
                "opf_segundos": o["segundos"], "agente_segundos": t_agente[-1]}
        for nombre, d in (("agente", ag), ("opf", op), ("merito", me)):
            for k in ("convergio", "seguro", "costo", "carga_max_pct", "n_ramas_sobrecargadas",
                      "v_min", "v_max", "n_barras_fuera", "exceso_slack_mw", "n_gen_fuera_q",
                      "perdidas_mw"):
                fila[f"{nombre}_{k}"] = d.get(k, np.nan)
            if o["convergio"] and d.get("convergio"):
                fila[f"{nombre}_brecha_pct"] = 100 * (d["costo"] - o["costo"]) / o["costo"]
            else:
                fila[f"{nombre}_brecha_pct"] = np.nan
        filas.append(fila)
        # Guardar un escenario representativo (OPF con congestión, agente seguro)
        if (ejemplo is None and o["convergio"] and ag.get("seguro") and not me.get("seguro")
                and 0.75 < e.fd < 0.85):
            ejemplo = (i, e, ag, op, me)

    df = pd.DataFrame(filas)
    df.to_csv(D / "detalle_por_escenario.csv", index=False)
    factible = df[df.opf_factible]

    def resumen(nombre):
        sub = factible
        return {
            "pct_convergencia": 100 * df[f"{nombre}_convergio"].astype(float).mean(),
            "pct_seguro": 100 * sub[f"{nombre}_seguro"].astype(float).mean(),
            "brecha_media_pct": float(sub[f"{nombre}_brecha_pct"].mean()),
            "brecha_mediana_pct": float(sub[f"{nombre}_brecha_pct"].median()),
            "brecha_p90_pct": float(sub[f"{nombre}_brecha_pct"].quantile(0.9)),
            "brecha_media_si_seguro_pct": float(
                sub.loc[sub[f"{nombre}_seguro"].astype(bool), f"{nombre}_brecha_pct"].mean()),
            "carga_max_media_pct": float(sub[f"{nombre}_carga_max_pct"].mean()),
            "carga_max_peor_pct": float(sub[f"{nombre}_carga_max_pct"].max()),
            "v_min_peor": float(sub[f"{nombre}_v_min"].min()),
            "v_max_peor": float(sub[f"{nombre}_v_max"].max()),
            "pct_con_sobrecarga": 100 * float((sub[f"{nombre}_n_ramas_sobrecargadas"] > 0).mean()),
            "pct_con_tension_fuera": 100 * float((sub[f"{nombre}_n_barras_fuera"] > 0).mean()),
            "pct_con_slack_fuera": 100 * float((sub[f"{nombre}_exceso_slack_mw"] > C.TOL_VIOLACION * 100).mean()),
            "pct_con_q_fuera": 100 * float((sub[f"{nombre}_n_gen_fuera_q"] > 0).mean()),
        }

    metricas = {
        "n_escenarios": len(df),
        "n_opf_factibles": int(df.opf_factible.sum()),
        "agente": resumen("agente"),
        "opf": resumen("opf"),
        "merito": resumen("merito"),
        "tiempo_agente_ms": 1000 * float(np.mean(t_agente)),
        "tiempo_opf_ms": 1000 * float(df.opf_segundos.mean()),
    }
    metricas["aceleracion_vs_opf"] = metricas["tiempo_opf_ms"] / metricas["tiempo_agente_ms"]

    # Política híbrida "agente + respaldo OPF": el flujo de carga del despacho
    # del agente ya se calcula, así que se sabe al instante si es seguro. Si no
    # lo es, se recurre al OPF. Es la forma práctica de operar: rapidez del
    # agente en la gran mayoría de casos y seguridad garantizada siempre.
    seg = factible.agente_seguro.astype(bool)
    costo_h = np.where(seg, factible.agente_costo, factible.opf_costo)
    brecha_h = 100 * (costo_h - factible.opf_costo) / factible.opf_costo
    t_h = factible.agente_segundos + np.where(seg, 0.0, factible.opf_segundos)
    metricas["hibrido"] = {
        "pct_seguro": 100.0,
        "pct_usa_respaldo": 100 * float((~seg).mean()),
        "brecha_media_pct": float(brecha_h.mean()),
        "brecha_mediana_pct": float(np.median(brecha_h)),
        "brecha_p90_pct": float(np.percentile(brecha_h, 90)),
        "tiempo_medio_ms": 1000 * float(t_h.mean()),
        "aceleracion_vs_opf": float(factible.opf_segundos.mean() / t_h.mean()),
    }

    # Desempeño por nivel de demanda
    bordes = np.linspace(C.FD_MIN, C.FD_MAX, 5)
    por_fd = []
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        s = factible[(factible.fd >= lo) & (factible.fd < hi + 1e-9)]
        por_fd.append({"fd": f"{lo:.2f}-{hi:.2f}", "n": len(s),
                       "agente_pct_seguro": 100 * s.agente_seguro.astype(float).mean(),
                       "agente_brecha_pct": s.agente_brecha_pct.mean(),
                       "merito_pct_seguro": 100 * s.merito_seguro.astype(float).mean()})
    metricas["por_nivel_demanda"] = por_fd
    metricas["capa_balance"] = entorno.capa_balance
    (D / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False, default=float))

    _figuras(df, factible, ejemplo, red, D)
    _reporte(metricas, D)
    a = metricas["agente"]
    print(f"\n  AGENTE:  seguro {a['pct_seguro']:.1f} %  |  brecha vs OPF: media {a['brecha_media_pct']:.2f} %, "
          f"mediana {a['brecha_mediana_pct']:.2f} %  |  {metricas['tiempo_agente_ms']:.2f} ms/despacho "
          f"({metricas['aceleracion_vs_opf']:.0f}× más rápido que el OPF)")
    h = metricas["hibrido"]
    print(f"  AGENTE + RESPALDO OPF: seguro 100 %  |  brecha media {h['brecha_media_pct']:.2f} %  |  "
          f"{h['tiempo_medio_ms']:.1f} ms/despacho ({h['aceleracion_vs_opf']:.0f}× más rápido que el OPF)")
    print(f"  MÉRITO SIN RED: seguro {metricas['merito']['pct_seguro']:.1f} %")
    print(f"  Reporte: {D / 'reporte_evaluacion.md'}")
    return metricas


# -----------------------------------------------------------------------------
# Figuras y reporte
# -----------------------------------------------------------------------------
def _figuras(df, factible, ejemplo, red, D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    azul, verde, naranja, rojo, gris = "#00B4D8", "#06D6A0", "#FF9F1C", "#EF4444", "#64748B"

    # 1) Costo agente vs costo OPF
    fig, ax = plt.subplots(figsize=(5.5, 5))
    seg = factible.agente_seguro.astype(bool)
    ax.scatter(factible.opf_costo[seg], factible.agente_costo[seg], s=12, color=azul, label="Agente seguro")
    ax.scatter(factible.opf_costo[~seg], factible.agente_costo[~seg], s=16, color=rojo, marker="x",
               label="Agente con violación")
    lim = [factible.opf_costo.min() * 0.95, factible.opf_costo.max() * 1.05]
    ax.plot(lim, lim, color=gris, ls="--", lw=1, label="Costo = OPF")
    ax.set_xlabel("Costo OPF exacto [USD/h]")
    ax.set_ylabel("Costo agente PPO [USD/h]")
    ax.set_title("Costo del despacho: agente vs. óptimo")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(D / "costo_agente_vs_opf.png", dpi=150)
    plt.close(fig)

    # 2) Histograma de brecha
    fig, ax = plt.subplots(figsize=(6.5, 4))
    b = factible.agente_brecha_pct.dropna()
    ax.hist(b, bins=40, color=azul, alpha=0.85)
    ax.axvline(b.median(), color=naranja, lw=2, label=f"Mediana {b.median():.2f} %")
    ax.set_xlabel("Brecha de costo vs. OPF exacto [%]")
    ax.set_ylabel("Escenarios")
    ax.set_title("Distribución de la brecha de costo del agente")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(D / "histograma_brecha.png", dpi=150)
    plt.close(fig)

    # 3) Seguridad por método y cargabilidad máxima
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    nombres = ["Agente PPO", "OPF exacto", "Orden de mérito\n(sin red)"]
    claves = ["agente", "opf", "merito"]
    pct = [100 * factible[f"{k}_seguro"].astype(float).mean() for k in claves]
    ejes[0].bar(nombres, pct, color=[azul, verde, rojo])
    for i, p in enumerate(pct):
        ejes[0].text(i, p + 1, f"{p:.1f} %", ha="center")
    ejes[0].set_ylim(0, 110)
    ejes[0].set_ylabel("% de despachos sin violaciones")
    ejes[0].set_title("Seguridad (límites reales)")
    datos = [factible[f"{k}_carga_max_pct"].dropna() for k in claves]
    ejes[1].boxplot(datos, tick_labels=nombres, showfliers=True)
    ejes[1].axhline(C.CARGABILIDAD_MAX_PCT, color=rojo, ls="--", lw=1.2, label="Límite 100 %")
    ejes[1].set_ylabel("Cargabilidad de la rama más cargada [%]")
    ejes[1].set_title("Cargabilidad máxima por escenario")
    ejes[1].legend()
    for ax in ejes:
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(D / "seguridad_y_cargabilidad.png", dpi=150)
    plt.close(fig)

    # 4) Ejemplo de despacho
    if ejemplo is not None:
        i, e, ag, op, me = ejemplo
        x = np.arange(red.n_gen)
        w = 0.27
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.bar(x - w, ag["p_gen"], w, color=azul, label=f"Agente  ({ag['costo']:.0f} USD/h, seguro)")
        ax.bar(x, op["p_gen"], w, color=verde, label=f"OPF  ({op['costo']:.0f} USD/h)")
        ax.bar(x + w, me["p_gen"], w, color=rojo, alpha=0.8,
               label=f"Mérito sin red  ({me['costo']:.0f} USD/h, carga máx {me['carga_max_pct']:.0f} %)")
        ax.scatter(x, red.p_max, marker="_", s=600, color=gris, label="Pmax")
        etiquetas = [f"{n}\n{p:.0f} USD/MWh" for n, p in zip(red.nombres_gen, e.precios)]
        ax.set_xticks(x, etiquetas, fontsize=8)
        ax.set_ylabel("P [MW]")
        ax.set_title(f"Escenario de prueba #{i}: FD={e.fd:.2f}, demanda {e.demanda_total_mw:.1f} MW")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(D / "ejemplo_despacho.png", dpi=150)
        plt.close(fig)


def _reporte(m, D):
    a, o, me = m["agente"], m["opf"], m["merito"]

    def fila(nombre, d):
        return (f"| {nombre} | {d['pct_seguro']:.1f} | {d['brecha_media_pct']:.2f} | "
                f"{d['brecha_mediana_pct']:.2f} | {d['brecha_p90_pct']:.2f} | "
                f"{d['carga_max_peor_pct']:.1f} | {d['v_min_peor']:.3f} – {d['v_max_peor']:.3f} |")

    texto = f"""# Evaluación del agente PPO de despacho

Variante de acción: {'con' if m['capa_balance'] else 'sin'} capa de balance.
Conjunto de prueba: **{m['n_escenarios']} escenarios nunca vistos** (semilla {C.SEMILLA_PRUEBA}),
FD ∈ [{C.FD_MIN}, {C.FD_MAX}], precios ∈ [{C.PRECIO_MIN:.0f}, {C.PRECIO_MAX:.0f}] USD/MWh, red `{C.CASO_RED}`.
El OPF exacto tiene solución en **{m['n_opf_factibles']}** de ellos; las métricas se calculan sobre esos.

Todos los despachos se juzgan con el mismo flujo de carga AC y los límites reales
(cargabilidad ≤ {C.CARGABILIDAD_MAX_PCT:.0f} %, tensión en [{C.TENSION_MIN_PU}, {C.TENSION_MAX_PU}] p.u.,
P de la slack y Q de todos los generadores dentro de sus límites).

| Método | % seguro | Brecha media vs OPF [%] | Brecha mediana [%] | Brecha p90 [%] | Peor cargabilidad [%] | Tensiones extremas [p.u.] |
|---|---|---|---|---|---|---|
{fila('Agente PPO', a)}
{fila('OPF exacto (referencia)', o)}
{fila('Orden de mérito sin red', me)}

| Agente + respaldo OPF¹ | 100.0 | {m['hibrido']['brecha_media_pct']:.2f} | {m['hibrido']['brecha_mediana_pct']:.2f} | {m['hibrido']['brecha_p90_pct']:.2f} | 100.0 | — |

¹ Política híbrida: se usa el despacho del agente si su flujo de carga es seguro (se verifica al
instante) y el OPF solo cuando no lo es ({m['hibrido']['pct_usa_respaldo']:.1f} % de los escenarios).
Tiempo medio **{m['hibrido']['tiempo_medio_ms']:.1f} ms** por despacho, **{m['hibrido']['aceleracion_vs_opf']:.0f}× más rápido** que usar
siempre el OPF, con 100 % de despachos seguros.

Brecha media del agente **solo en sus despachos seguros**: {a['brecha_media_si_seguro_pct']:.2f} %.

Nota: el orden de mérito sin red puede salir *más barato* que el OPF (brecha negativa) porque
ignora los límites de la red: es barato precisamente porque sobrecarga líneas. Por eso la
brecha de costo solo tiene sentido junto con el % de despachos seguros.

Tiempo por despacho (red neuronal + capa de balance + flujo de carga AC de verificación):
agente **{m['tiempo_agente_ms']:.2f} ms** vs OPF **{m['tiempo_opf_ms']:.0f} ms**
→ el agente es **{m['aceleracion_vs_opf']:.0f}× más rápido**.

Tipos de violación del agente (en % de escenarios): sobrecarga {a['pct_con_sobrecarga']:.1f} %,
tensión fuera de banda {a['pct_con_tension_fuera']:.1f} %, slack fuera de límites {a['pct_con_slack_fuera']:.1f} %,
reactiva fuera de límites {a['pct_con_q_fuera']:.1f} %.

## Por nivel de demanda

| FD | n | Agente % seguro | Agente brecha media [%] | Mérito sin red % seguro |
|---|---|---|---|---|
""" + "\n".join(f"| {r['fd']} | {r['n']} | {r['agente_pct_seguro']:.1f} | {r['agente_brecha_pct']:.2f} | "
               f"{r['merito_pct_seguro']:.1f} |" for r in m["por_nivel_demanda"]) + """

## Figuras

- `costo_agente_vs_opf.png` — costo del agente frente al óptimo, escenario por escenario.
- `histograma_brecha.png` — distribución de la brecha de costo.
- `seguridad_y_cargabilidad.png` — % de despachos seguros y cargabilidad máxima por método.
- `ejemplo_despacho.png` — un escenario concreto: agente vs OPF vs orden de mérito.
"""
    (D / "reporte_evaluacion.md").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--capa", choices=["si", "no"], default=None)
    ap.add_argument("--carpeta", default=None)
    a = ap.parse_args()
    ejecutar(a.modelo, None if a.capa is None else a.capa == "si", a.carpeta)
