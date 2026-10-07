"""
agente.py
=========
Utilidades del agente PPO compartidas por TODO el proyecto (Optuna,
entrenamiento, evaluación e interfaz), para que exista una sola forma de:

    * crear los entornos vectorizados            -> crear_entornos()
    * construir un PPO a partir de hiperparámetros -> construir_ppo()
    * cargar el modelo entrenado                 -> cargar_modelo()
    * pedirle un despacho al agente              -> despachar()
    * medir el desempeño de una política         -> evaluar_politica()
    * evaluar periódicamente durante el entrenamiento -> CallbackEvaluacion
"""

from __future__ import annotations

import json

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import config as C
from entorno_despacho import EntornoDespacho
from escenarios import Escenario, GeneradorEscenarios
from recompensa import calcular_violaciones

# El ambiente es muy liviano: con un solo hilo de PyTorch PPO va más rápido
# que dejando que PyTorch reparta operaciones pequeñas entre núcleos.
torch.set_num_threads(1)


def crear_entornos(n_envs: int = C.N_ENVS, semilla: int = C.SEMILLA) -> DummyVecEnv:
    """n_envs copias del ambiente, cada una con su propia semilla de escenarios
    (si todas usaran la misma, verían exactamente los mismos escenarios)."""
    def _fabrica(i):
        def _crear():
            return Monitor(EntornoDespacho(semilla=semilla + 1000 * i))
        return _crear
    return DummyVecEnv([_fabrica(i) for i in range(n_envs)])


def politica_kwargs(hp: dict) -> dict:
    """Traduce los hiperparámetros 'legibles' a los argumentos de SB3."""
    capas = C.ARQUITECTURAS[hp["arquitectura"]]
    activacion = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[hp["activacion"]]
    return {
        # Actor (pi) y crítico (vf) con redes separadas del mismo tamaño.
        "net_arch": {"pi": list(capas), "vf": list(capas)},
        "activation_fn": activacion,
        # Desviación estándar inicial de la política gaussiana (log): controla
        # cuánto explora el agente al principio.
        "log_std_init": hp["log_std_init"],
        # Ortogonal por defecto en SB3; se deja explícito.
        "ortho_init": True,
    }


def construir_ppo(entornos, hp: dict, semilla: int = C.SEMILLA, verbose: int = 0,
                  tensorboard_log: str | None = None) -> PPO:
    """Crea un PPO de Stable-Baselines3 con los hiperparámetros dados.

    batch_size debe dividir a n_steps * n_envs (tamaño del rollout); si no,
    SB3 avisa y descarta el sobrante. Aquí se ajusta automáticamente.
    """
    n_envs = entornos.num_envs
    rollout = hp["n_steps"] * n_envs
    batch = min(hp["batch_size"], rollout)
    while rollout % batch:
        batch //= 2
    lr0 = hp["learning_rate"]
    if hp.get("esquema_lr", "constante") == "lineal":
        # SB3 llama a esta función con progress_remaining: 1 al inicio, 0 al final.
        tasa = lambda progreso_restante: lr0 * progreso_restante  # noqa: E731
    else:
        tasa = lr0
    return PPO(
        "MlpPolicy",
        entornos,
        learning_rate=tasa,
        n_steps=hp["n_steps"],
        batch_size=batch,
        n_epochs=hp["n_epochs"],
        gamma=hp["gamma"],
        gae_lambda=hp["gae_lambda"],
        clip_range=hp["clip_range"],
        ent_coef=hp["ent_coef"],
        vf_coef=hp["vf_coef"],
        max_grad_norm=hp["max_grad_norm"],
        policy_kwargs=politica_kwargs(hp),
        seed=semilla,
        device="cpu",
        verbose=verbose,
        tensorboard_log=tensorboard_log,
    )


def hiperparametros_actuales() -> dict:
    """Los de Optuna si ya se corrió la búsqueda; si no, los por defecto."""
    hp = dict(C.HIPERPARAMETROS_PPO_DEFECTO)
    if C.RUTA_MEJORES_HIPERPARAMETROS.exists():
        hp.update(json.loads(C.RUTA_MEJORES_HIPERPARAMETROS.read_text())["hiperparametros"])
    return hp


def cargar_modelo(ruta=None, entorno: EntornoDespacho | None = None) -> PPO:
    """Carga el modelo entrenado. Si se da `entorno`, verifica que las
    dimensiones de observación y acción coincidan (p. ej. un modelo entrenado
    con otra red o con otro valor de CAPA_BALANCE no sirve)."""
    ruta = ruta or C.RUTA_MODELO_FINAL
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el modelo {ruta}. Entrénelo primero con: python main.py entrenar")
    modelo = PPO.load(str(ruta), device="cpu")
    if entorno is not None and (modelo.observation_space.shape != entorno.observation_space.shape
                                or modelo.action_space.shape != entorno.action_space.shape):
        raise ValueError(
            f"El modelo {ruta.name} espera obs={modelo.observation_space.shape}, "
            f"acción={modelo.action_space.shape}, pero la configuración actual da "
            f"obs={entorno.observation_space.shape}, acción={entorno.action_space.shape}. "
            f"Revise CASO_RED / CAPA_BALANCE en config.py o vuelva a entrenar.")
    return modelo


def despachar(modelo: PPO, entorno: EntornoDespacho, esc: Escenario) -> dict:
    """Pide al agente el despacho de un escenario y lo evalúa en pandapower.

    Usa deterministic=True: en operación se toma la MEDIA de la política
    gaussiana (sin ruido de exploración), así la misma entrada da siempre la
    misma salida.
    """
    obs = entorno.observacion(esc)
    accion, _ = modelo.predict(obs, deterministic=True)
    r, info, res, p_pv, v_gen = entorno.evaluar_despacho(esc, accion)
    real = calcular_violaciones(res, entorno.red, con_margen=False)
    costo = float(entorno.red.costo_despacho(res.p_gen_mw, esc.precios)) if res.convergio else float("nan")
    return {"recompensa": r, "info": info, "resultado": res, "violaciones_reales": real,
            "p_pv": p_pv, "v_gen": v_gen, "costo": costo, "accion": accion}


def escenarios_validacion(n: int, semilla: int, entorno: EntornoDespacho) -> list[Escenario]:
    """Lista FIJA de escenarios (misma semilla -> mismos escenarios)."""
    return GeneradorEscenarios(entorno.red, semilla).lote(n)


def evaluar_politica(modelo: PPO, entorno: EntornoDespacho, escenarios: list[Escenario]) -> dict:
    """Desempeño determinista de una política sobre escenarios fijos.

    Métricas:
        recompensa_media  : lo que se optimiza (Optuna usa esta).
        pct_seguro_real   : % de despachos sin NINGUNA violación con los
                            límites reales (100 %, [0.95, 1.05] p.u., P/Q).
        pct_seguro_margen : % que cumplen incluso los límites de entrenamiento.
        sobrecosto_medio  : sobrecosto relativo vs. orden de mérito (despachos
                            seguros reales), en fracción.
        pct_convergencia  : % de flujos de carga que convergieron.
    """
    rs, seguro_real, seguro_margen, sobrecostos, conv = [], [], [], [], []
    for esc in escenarios:
        d = despachar(modelo, entorno, esc)
        rs.append(d["recompensa"])
        conv.append(d["resultado"].convergio)
        seguro_real.append(bool(d["violaciones_reales"].get("seguro", False)))
        seguro_margen.append(bool(d["info"].get("seguro", False)))
        if seguro_real[-1]:
            sobrecostos.append(d["info"]["sobrecosto"])
    return {
        "recompensa_media": float(np.mean(rs)),
        "pct_seguro_real": 100.0 * float(np.mean(seguro_real)),
        "pct_seguro_margen": 100.0 * float(np.mean(seguro_margen)),
        "sobrecosto_medio": float(np.mean(sobrecostos)) if sobrecostos else float("nan"),
        "pct_convergencia": 100.0 * float(np.mean(conv)),
    }


class CallbackEvaluacion(BaseCallback):
    """Cada `cada_pasos` pasos de entrenamiento evalúa la política (determinista)
    sobre escenarios FIJOS de validación.

    - Guarda el mejor modelo (por recompensa media) si se da `ruta_mejor`.
    - Registra la historia de métricas en `self.historia`.
    - Llama a `al_evaluar(paso, metricas)`; si devuelve False, detiene el
      entrenamiento (así Optuna "poda" los trials malos).

    Se usa UN SOLO model.learn(total) con este callback en lugar de entrenar
    por tramos, porque al entrenar por tramos SB3 reinicia el progreso y un
    esquema de tasa de aprendizaje lineal quedaría en "dientes de sierra".
    """

    def __init__(self, entorno_eval: EntornoDespacho, escenarios: list[Escenario],
                 cada_pasos: int, ruta_mejor=None, al_evaluar=None, verbose: int = 1,
                 evaluar_al_final: bool = True):
        super().__init__(verbose)
        self.entorno_eval = entorno_eval
        self.escenarios = escenarios
        self.cada_pasos = cada_pasos
        self.ruta_mejor = ruta_mejor
        self.al_evaluar = al_evaluar
        # Si el entrenamiento no termina justo en un múltiplo de cada_pasos (o
        # es más corto), se hace una última evaluación para no perder el final.
        self.evaluar_al_final = evaluar_al_final
        self.mejor_recompensa = -np.inf
        self.mejores_metricas: dict | None = None
        self.historia: list[dict] = []
        self._proxima = cada_pasos

    def _evaluar(self) -> dict:
        m = evaluar_politica(self.model, self.entorno_eval, self.escenarios)
        m["paso"] = int(self.num_timesteps)
        self.historia.append(m)
        mejoro = m["recompensa_media"] > self.mejor_recompensa
        if mejoro:
            self.mejor_recompensa = m["recompensa_media"]
            self.mejores_metricas = m
            if self.ruta_mejor is not None:
                self.model.save(str(self.ruta_mejor))
        if self.verbose:
            print(f"    paso {m['paso']:>9,}  r={m['recompensa_media']:7.3f}  "
                  f"seguro={m['pct_seguro_real']:5.1f}%  "
                  f"sobrecosto={100 * m['sobrecosto_medio']:5.2f}%"
                  f"{'  <- mejor' if mejoro else ''}", flush=True)
        return m

    def _on_step(self) -> bool:
        if self.num_timesteps < self._proxima:
            return True
        self._proxima += self.cada_pasos
        m = self._evaluar()
        if self.al_evaluar is not None:
            return bool(self.al_evaluar(m["paso"], m))
        return True

    def _on_training_end(self) -> None:
        if self.evaluar_al_final and (not self.historia
                                      or self.historia[-1]["paso"] != self.num_timesteps):
            m = self._evaluar()
            if self.al_evaluar is not None:
                self.al_evaluar(m["paso"], m)
