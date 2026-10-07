"""
optimizar_hiperparametros.py
============================
BÚSQUEDA BAYESIANA DE HIPERPARÁMETROS DE PPO CON OPTUNA.

Por qué importa: con la misma recompensa y el mismo ambiente, PPO puede
estancarse en un despacho caro o llegar cerca del OPF dependiendo de la tasa de
aprendizaje, el tamaño del lote, la arquitectura, etc. En el piloto con valores
por defecto el agente se estancó en ~12 % de sobrecosto (el OPF está en ~5 %).

Cómo funciona:
    1. Optuna PROPONE un juego de hiperparámetros (muestreador TPE: aprende de
       los trials anteriores qué regiones del espacio son prometedoras).
    2. Se entrena un PPO con ellos durante OPTUNA_PASOS_POR_TRIAL pasos.
    3. Cada PASOS/OPTUNA_N_EVALUACIONES pasos se evalúa la política determinista
       sobre ESCENARIOS_VALIDACION escenarios FIJOS (los mismos para
       todos los trials -> comparación justa) y se reporta a Optuna.
    4. PODA (MedianPruner): si en una evaluación intermedia el trial va peor
       que la mediana de los trials anteriores en el mismo punto, se corta.
       Así no se gasta tiempo completo en configuraciones malas.
    5. El valor del trial es la MEJOR recompensa media de validación alcanzada.

El espacio de búsqueda está en config.ESPACIO_BUSQUEDA.

Salidas (en resultados/optuna/):
    estudio.db                     base SQLite del estudio (se puede reanudar)
    trials.csv                     todos los trials con parámetros y métricas
    mejores_hiperparametros.json   el ganador (lo lee entrenar.py)
    historia_optimizacion.png      valor de cada trial y mejor acumulado
    importancia_hiperparametros.png qué hiperparámetros pesan más

Uso:
    python main.py optimizar                       (1 proceso)
    python main.py optimizar --trabajadores 2      (2 procesos en paralelo)
    python main.py optimizar --trials 40 --pasos 300000
Si se interrumpe, volver a ejecutar REANUDA el mismo estudio.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import warnings

import numpy as np
import optuna

import config as C
from agente import (CallbackEvaluacion, construir_ppo, crear_entornos,
                    escenarios_validacion)
from entorno_despacho import EntornoDespacho

warnings.filterwarnings("ignore")


def _almacenamiento() -> str:
    C.DIR_OPTUNA.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(C.DIR_OPTUNA / 'estudio.db').as_posix()}"


def proponer_hiperparametros(trial: optuna.Trial) -> dict:
    """Muestrea un juego de hiperparámetros del espacio de config.ESPACIO_BUSQUEDA.

    Qué controla cada uno:
      learning_rate : tamaño del paso de gradiente. Muy alto -> inestable;
                      muy bajo -> aprende lento.
      esquema_lr    : constante o decaimiento lineal a 0 (ajuste fino al final).
      n_steps       : pasos por entorno en cada rollout (datos por actualización
                      = n_steps · N_ENVS). Más datos -> gradiente menos ruidoso.
      batch_size    : tamaño del minilote en cada época de optimización.
      n_epochs      : cuántas veces se reutiliza cada rollout. Más -> más
                      eficiente en datos, pero riesgo de sobreajustar el lote.
      clip_range    : cuánto puede cambiar la política por actualización (el
                      "proximal" de PPO). Pequeño -> cambios conservadores.
      ent_coef      : bonificación por entropía (exploración).
      arquitectura  : tamaño de las redes del actor y del crítico.
      activacion    : tanh o ReLU.
      log_std_init  : exploración inicial (desv. estándar de la gaussiana).
    """
    E = C.ESPACIO_BUSQUEDA
    hp = dict(C.HIPERPARAMETROS_PPO_DEFECTO)  # gamma, gae_lambda, vf_coef, ... fijos
    hp.update({
        "learning_rate": trial.suggest_float("learning_rate", *E["learning_rate"], log=True),
        "esquema_lr": trial.suggest_categorical("esquema_lr", E["esquema_lr"]),
        "n_steps": trial.suggest_categorical("n_steps", E["n_steps"]),
        "batch_size": trial.suggest_categorical("batch_size", E["batch_size"]),
        "n_epochs": trial.suggest_categorical("n_epochs", E["n_epochs"]),
        "clip_range": trial.suggest_categorical("clip_range", E["clip_range"]),
        "ent_coef": trial.suggest_float("ent_coef", *E["ent_coef"], log=True),
        "arquitectura": trial.suggest_categorical("arquitectura", E["arquitectura"]),
        "activacion": trial.suggest_categorical("activacion", E["activacion"]),
        "log_std_init": trial.suggest_float("log_std_init", *E["log_std_init"]),
    })
    return hp


class Objetivo:
    """Función objetivo de Optuna (clase para fijar pasos y escenarios)."""

    def __init__(self, pasos: int):
        self.pasos = pasos
        self.entorno_eval = EntornoDespacho(semilla=0)
        self.escenarios = escenarios_validacion(
            C.ESCENARIOS_VALIDACION, C.SEMILLA_VALIDACION, self.entorno_eval)

    def __call__(self, trial: optuna.Trial) -> float:
        hp = proponer_hiperparametros(trial)
        modelo = construir_ppo(crear_entornos(), hp, semilla=C.SEMILLA + trial.number)
        estado = {"podado": False}
        cada = self.pasos // C.OPTUNA_N_EVALUACIONES

        def al_evaluar(paso, m):
            # Se reporta con el ÍNDICE de la evaluación (1, 2, 3...) para que
            # n_warmup_steps del pruner se cuente en evaluaciones.
            trial.report(m["recompensa_media"], step=int(round(paso / cada)))
            if trial.should_prune():
                estado["podado"] = True
                return False  # detiene model.learn()
            return True

        cb = CallbackEvaluacion(self.entorno_eval, self.escenarios,
                                cada_pasos=self.pasos // C.OPTUNA_N_EVALUACIONES,
                                al_evaluar=al_evaluar, verbose=0,
                                evaluar_al_final=False)  # el pruner ya tiene sus reportes
        t0 = time.time()
        try:
            modelo.learn(self.pasos, callback=cb)
        except (ValueError, RuntimeError) as ex:  # p. ej. NaN por lr extremo
            print(f"  trial {trial.number}: falló ({ex}); se descarta")
            raise optuna.TrialPruned() from ex
        mejor = cb.mejores_metricas or {}
        trial.set_user_attr("pct_seguro_real", mejor.get("pct_seguro_real"))
        trial.set_user_attr("sobrecosto_medio", mejor.get("sobrecosto_medio"))
        trial.set_user_attr("paso_mejor", mejor.get("paso"))
        trial.set_user_attr("minutos", (time.time() - t0) / 60)
        print(f"  trial {trial.number:3d}: r={cb.mejor_recompensa:7.3f}  "
              f"seguro={mejor.get('pct_seguro_real', float('nan')):5.1f}%  "
              f"sobrecosto={100 * mejor.get('sobrecosto_medio', float('nan')):5.2f}%  "
              f"{'PODADO ' if estado['podado'] else ''}"
              f"({(time.time() - t0) / 60:.1f} min)  {trial.params}", flush=True)
        if estado["podado"]:
            raise optuna.TrialPruned()
        return float(cb.mejor_recompensa)


def _crear_estudio(semilla_muestreador: int = C.SEMILLA) -> optuna.Study:
    """Crea o abre el estudio. Cada proceso trabajador usa una semilla distinta
    para el muestreador; si compartieran semilla, propondrían los MISMOS
    hiperparámetros en paralelo (trabajo duplicado)."""
    return optuna.create_study(
        study_name=C.OPTUNA_NOMBRE_ESTUDIO,
        storage=_almacenamiento(),
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=semilla_muestreador, n_startup_trials=8),
        # No se poda nada en los primeros 5 trials ni antes de la 2.ª evaluación.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1,
                                           interval_steps=1),
    )


def _trabajador(n_trials_total: int, pasos: int, semilla_trabajador: int):
    """Proceso que corre trials hasta completar n_trials_total ENTRE TODOS los
    trabajadores (comparten el estudio en SQLite)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    np.random.seed(semilla_trabajador)
    estudio = _crear_estudio(C.SEMILLA + 1000 * semilla_trabajador)
    estudio.optimize(
        Objetivo(pasos),
        callbacks=[optuna.study.MaxTrialsCallback(
            n_trials_total, states=(optuna.trial.TrialState.COMPLETE,
                                    optuna.trial.TrialState.PRUNED))],
        gc_after_trial=True,
    )


def guardar_resultados(estudio: optuna.Study, pasos: int = C.OPTUNA_PASOS_POR_TRIAL):
    """Exporta el ganador, la tabla de trials y las figuras."""
    completos = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completos:
        print("No hay trials completos todavía.")
        return
    mejor = estudio.best_trial
    hp = dict(C.HIPERPARAMETROS_PPO_DEFECTO)
    hp.update(mejor.params)
    salida = {
        "trial": mejor.number,
        "recompensa_validacion": mejor.value,
        "pct_seguro_real": mejor.user_attrs.get("pct_seguro_real"),
        "sobrecosto_medio": mejor.user_attrs.get("sobrecosto_medio"),
        "pasos_por_trial": pasos,
        "trials_completos": len(completos),
        "trials_podados": sum(t.state == optuna.trial.TrialState.PRUNED for t in estudio.trials),
        "hiperparametros": hp,
    }
    C.RUTA_MEJORES_HIPERPARAMETROS.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    estudio.trials_dataframe().to_csv(C.DIR_OPTUNA / "trials.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Historia: valor de cada trial completo y mejor acumulado.
    nums = [t.number for t in completos]
    vals = [t.value for t in completos]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(nums, vals, s=22, color="#00B4D8", label="Trial completo")
    ax.plot(nums, np.maximum.accumulate(vals), color="#06D6A0", lw=2, label="Mejor acumulado")
    podados = [t.number for t in estudio.trials if t.state == optuna.trial.TrialState.PRUNED]
    for p in podados:
        ax.axvline(p, color="#EF4444", alpha=0.15, lw=3)
    ax.set_xlabel("Número de trial")
    ax.set_ylabel("Recompensa media de validación")
    ax.set_title(f"Optuna: {len(completos)} completos, {len(podados)} podados (franjas rojas)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.DIR_OPTUNA / "historia_optimizacion.png", dpi=150)
    plt.close(fig)

    # Importancia de hiperparámetros (fANOVA de Optuna).
    try:
        imp = optuna.importance.get_param_importances(estudio)
        fig, ax = plt.subplots(figsize=(7, 4))
        nombres = list(imp.keys())[::-1]
        ax.barh(nombres, [imp[n] for n in nombres], color="#FF9F1C")
        ax.set_xlabel("Importancia relativa (fANOVA)")
        ax.set_title("¿Qué hiperparámetros pesan más en la recompensa?")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(C.DIR_OPTUNA / "importancia_hiperparametros.png", dpi=150)
        plt.close(fig)
        salida["importancias"] = imp
        C.RUTA_MEJORES_HIPERPARAMETROS.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    except Exception as ex:  # noqa: BLE001  (requiere >= 2 trials completos distintos)
        print(f"  (no se pudo calcular importancias: {ex})")

    print("\nMEJOR TRIAL")
    print(json.dumps(salida, indent=2, ensure_ascii=False))


def ejecutar(n_trials: int = C.OPTUNA_N_TRIALS, pasos: int = C.OPTUNA_PASOS_POR_TRIAL,
             trabajadores: int = 1):
    print(f"\nOPTUNA: {n_trials} trials × {pasos:,} pasos, {trabajadores} proceso(s)")
    print(f"Estudio: {_almacenamiento()}\n")
    _crear_estudio()  # crea la base antes de lanzar procesos
    if trabajadores <= 1:
        _trabajador(n_trials, pasos, 0)
    else:
        ctx = mp.get_context("spawn")  # igual en Windows y Linux
        procesos = [ctx.Process(target=_trabajador, args=(n_trials, pasos, i))
                    for i in range(trabajadores)]
        for p in procesos:
            p.start()
            time.sleep(2)  # escalona el arranque (evita choques en SQLite)
        for p in procesos:
            p.join()
    # Con varios procesos puede terminar 1 trial más de lo pedido por trabajador
    # (cada uno revisa el total al acabar su trial en curso).
    guardar_resultados(_crear_estudio(), pasos)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=C.OPTUNA_N_TRIALS)
    ap.add_argument("--pasos", type=int, default=C.OPTUNA_PASOS_POR_TRIAL)
    ap.add_argument("--trabajadores", type=int, default=1)
    a = ap.parse_args()
    ejecutar(a.trials, a.pasos, a.trabajadores)
