"""
entrenar.py
===========
ENTRENAMIENTO FINAL del agente PPO de despacho.

Qué hace:
    1. Toma los hiperparámetros ganadores de Optuna
       (resultados/optuna/mejores_hiperparametros.json). Si no existen, usa
       config.HIPERPARAMETROS_PPO_DEFECTO.
    2. Entrena un agente por cada semilla de config.SEMILLAS_ENTRENAMIENTO, en
       procesos paralelos (uno por núcleo). Cada uno crea N_ENVS ambientes que
       sortean un escenario nuevo (cargas y precios) en cada episodio, así que
       el agente ve millones de escenarios distintos -> generaliza.
    3. Cada agente entrena PASOS_ENTRENAMIENTO_FINAL pasos (= episodios, porque
       cada episodio es de un paso) con UN SOLO model.learn().
    4. Cada EVAL_CADA_PASOS se evalúa la política determinista sobre los
       escenarios fijos de validación y se guarda el MEJOR checkpoint de esa
       semilla (no el último: si el entrenamiento oscila al final, no se pierde
       lo mejor que se alcanzó).
    5. Entre semillas se elige la de mejor recompensa de validación; ese es el
       modelo final. Entrenar varias semillas además muestra cuánto depende el
       resultado del azar de la inicialización (robustez).

Salidas (resultados/modelos/):
    despacho_ppo_final.zip           el modelo que usan la evaluación y la interfaz
    entrenamiento.json               hiperparámetros, tiempos, métricas por semilla
    curva_validacion.png             evolución de recompensa, seguridad y costo
    semilla_<s>/mejor.zip            mejor checkpoint de cada semilla
    semilla_<s>/ultimo.zip           estado al terminar
    semilla_<s>/historia_validacion.csv

Uso:
    python main.py entrenar
    python main.py entrenar --pasos 1000000 --semillas 1 2 3
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import time
import warnings

import pandas as pd

import config as C

warnings.filterwarnings("ignore")


def _entrenar_semilla(semilla: int, pasos: int, hp: dict) -> dict:
    """Entrena un agente con una semilla (se ejecuta en su propio proceso)."""
    import warnings as _w
    _w.filterwarnings("ignore")
    from agente import (CallbackEvaluacion, construir_ppo, crear_entornos,
                        escenarios_validacion)
    from entorno_despacho import EntornoDespacho

    carpeta = C.DIR_MODELOS / f"semilla_{semilla}"
    carpeta.mkdir(parents=True, exist_ok=True)
    modelo = construir_ppo(crear_entornos(semilla=semilla), hp, semilla=semilla)
    entorno_eval = EntornoDespacho(semilla=0)
    escenarios = escenarios_validacion(C.ESCENARIOS_VALIDACION, C.SEMILLA_VALIDACION, entorno_eval)
    cb = CallbackEvaluacion(entorno_eval, escenarios, C.EVAL_CADA_PASOS,
                            ruta_mejor=carpeta / "mejor.zip", verbose=0)

    t0 = time.time()

    def al_evaluar(paso, m):  # progreso en consola
        print(f"  [semilla {semilla}] paso {paso:>9,}  r={m['recompensa_media']:7.3f}  "
              f"seguro={m['pct_seguro_real']:5.1f}%  sobrecosto={100 * m['sobrecosto_medio']:5.2f}%  "
              f"({(time.time() - t0) / 60:.0f} min)", flush=True)
        return True

    cb.al_evaluar = al_evaluar
    modelo.learn(pasos, callback=cb)
    modelo.save(str(carpeta / "ultimo.zip"))
    pd.DataFrame(cb.historia).to_csv(carpeta / "historia_validacion.csv", index=False)
    return {"semilla": semilla, "minutos": round((time.time() - t0) / 60, 1),
            "mejor_validacion": cb.mejores_metricas}


def _proceso(semilla, pasos, hp, cola):
    cola.put(_entrenar_semilla(semilla, pasos, hp))


def graficar_historia(historias: dict, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colores = ["#00B4D8", "#FF9F1C", "#06D6A0", "#A78BFA"]
    fig, ejes = plt.subplots(1, 3, figsize=(14, 3.8))
    for k, (semilla, df) in enumerate(historias.items()):
        x = df["paso"] / 1e6
        c = colores[k % len(colores)]
        ejes[0].plot(x, df["recompensa_media"], color=c, lw=2, label=f"semilla {semilla}")
        ejes[1].plot(x, df["pct_seguro_real"], color=c, lw=2)
        ejes[2].plot(x, 100 * df["sobrecosto_medio"], color=c, lw=2)
    ejes[0].set_title("Recompensa media (validación)")
    ejes[0].legend()
    ejes[1].set_title("Despachos seguros (límites reales) [%]")
    ejes[2].set_title("Sobrecosto vs. orden de mérito [%]")
    for ax in ejes:
        ax.set_xlabel("Millones de pasos (= episodios)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def ejecutar(pasos: int = C.PASOS_ENTRENAMIENTO_FINAL, semillas=None):
    from agente import hiperparametros_actuales

    semillas = list(semillas or C.SEMILLAS_ENTRENAMIENTO)
    C.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    hp = hiperparametros_actuales()
    origen = "Optuna" if C.RUTA_MEJORES_HIPERPARAMETROS.exists() else "valores por defecto"
    print(f"\nENTRENAMIENTO FINAL — {pasos:,} pasos × semillas {semillas}, hiperparámetros de {origen}:")
    for k, v in hp.items():
        print(f"    {k:15s} = {v}")
    print(f"\n  Evaluando cada {C.EVAL_CADA_PASOS:,} pasos sobre {C.ESCENARIOS_VALIDACION} escenarios fijos:")

    t0 = time.time()
    if len(semillas) == 1:
        resultados = [_entrenar_semilla(semillas[0], pasos, hp)]
    else:
        ctx = mp.get_context("spawn")  # igual en Windows y Linux
        cola = ctx.Queue()
        procesos = [ctx.Process(target=_proceso, args=(s, pasos, hp, cola)) for s in semillas]
        for p in procesos:
            p.start()
        resultados = [cola.get() for _ in procesos]
        for p in procesos:
            p.join()
    minutos = (time.time() - t0) / 60

    resultados.sort(key=lambda r: r["semilla"])
    mejor = max(resultados, key=lambda r: r["mejor_validacion"]["recompensa_media"])
    shutil.copyfile(C.DIR_MODELOS / f"semilla_{mejor['semilla']}" / "mejor.zip", C.RUTA_MODELO_FINAL)

    historias = {r["semilla"]: pd.read_csv(C.DIR_MODELOS / f"semilla_{r['semilla']}" / "historia_validacion.csv")
                 for r in resultados}
    graficar_historia(historias, C.DIR_MODELOS / "curva_validacion.png")
    resumen = {
        "pasos_por_semilla": pasos,
        "minutos_totales": round(minutos, 1),
        "origen_hiperparametros": origen,
        "hiperparametros": hp,
        "semilla_elegida": mejor["semilla"],
        "por_semilla": resultados,
    }
    (C.DIR_MODELOS / "entrenamiento.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))

    print(f"\n  Terminado en {minutos:.1f} min.")
    for r in resultados:
        m = r["mejor_validacion"]
        print(f"    semilla {r['semilla']}: mejor r={m['recompensa_media']:.3f} (paso {m['paso']:,}), "
              f"seguro={m['pct_seguro_real']:.1f} %, sobrecosto={100 * m['sobrecosto_medio']:.2f} %"
              f"{'   <- ELEGIDA' if r is mejor else ''}")
    print(f"  Modelo final: {C.RUTA_MODELO_FINAL}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasos", type=int, default=C.PASOS_ENTRENAMIENTO_FINAL)
    ap.add_argument("--semillas", type=int, nargs="+", default=None)
    a = ap.parse_args()
    ejecutar(a.pasos, a.semillas)
