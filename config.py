"""
config.py
=========
ÚNICA FUENTE DE VERDAD DEL PROYECTO.

Todo parámetro que afecte el comportamiento del ambiente, la recompensa, el
entrenamiento, la búsqueda de hiperparámetros o la interfaz vive AQUÍ. Ningún
otro archivo define números "mágicos": todos importan de este módulo.

Si quiere cambiar la red, los rangos de carga, los precios, los pesos de la
recompensa o los hiperparámetros, este es el único archivo que debe tocar.

Organización:
    1. Rutas
    2. Red eléctrica y motor de flujo de carga
    3. Límites operativos (seguridad)
    4. Generación de escenarios (lo que cambia entre episodios)
    5. Función de recompensa (pesos)
    6. Hiperparámetros PPO por defecto
    7. Búsqueda de hiperparámetros (Optuna)
    8. Entrenamiento final y evaluación
    9. Interfaz web
"""

from pathlib import Path

# =============================================================================
# 1. RUTAS
# =============================================================================
RAIZ = Path(__file__).resolve().parent
DIR_RESULTADOS = RAIZ / "resultados"
DIR_MODELOS = DIR_RESULTADOS / "modelos"
DIR_OPTUNA = DIR_RESULTADOS / "optuna"
DIR_EVALUACION = DIR_RESULTADOS / "evaluacion"
DIR_LOGS = DIR_RESULTADOS / "logs"

# Modelo final que usan la evaluación y la interfaz.
RUTA_MODELO_FINAL = DIR_MODELOS / "despacho_ppo_final.zip"
# Hiperparámetros ganadores de Optuna (los lee entrenar.py si existen).
RUTA_MEJORES_HIPERPARAMETROS = DIR_OPTUNA / "mejores_hiperparametros.json"
# Conjunto de prueba con el OPF exacto ya resuelto (se genera una sola vez).
RUTA_CONJUNTO_PRUEBA = DIR_EVALUACION / "conjunto_prueba_opf.npz"


# =============================================================================
# 2. RED ELÉCTRICA Y MOTOR DE FLUJO DE CARGA
# =============================================================================
# Caso de pandapower.networks a usar. "case30" es la red IEEE de 30 barras en
# su versión MATPOWER: 6 generadores síncronos (1 slack + 5 PV), 20 cargas,
# 41 líneas a 135 kV con límites térmicos reales (varias de solo 16 MVA), lo que
# la hace una red CONGESTIONADA: el despacho por orden de mérito puro suele
# sobrecargar líneas y el agente está obligado a aprender a redespachar.
#
# El código es genérico: cualquier caso de pandapower.networks con generadores
# (net.gen) y una red externa (net.ext_grid) funciona cambiando esta línea
# (p. ej. "case39", la red IEEE de 39 barras). Para otra red habría que
# re-verificar el rango factible de carga (ver pruebas.py -> rango_factible).
CASO_RED = "case30"

# Motor del flujo de carga AC (Newton-Raphson):
#   "lightsim" -> lightsim2grid resolviendo el MISMO modelo pandapower. Es el
#                 motor acelerado que el propio pandapower usa con
#                 runpp(lightsim2grid=True), pero llamado directamente para
#                 evitar la sobrecarga de conversión en cada paso. ~0.02 ms por
#                 flujo vs ~8-30 ms de pp.runpp. Resultados idénticos (la prueba
#                 automática verifica diferencias < 1e-6).
#   "pandapower" -> pp.runpp clásico. Más lento; útil para auditar resultados.
MOTOR_FLUJO = "lightsim"
MAX_ITER_FLUJO = 30
TOLERANCIA_FLUJO = 1e-8


# =============================================================================
# 3. LÍMITES OPERATIVOS (SEGURIDAD)
# =============================================================================
# Límites REALES con los que se juzga si un despacho es seguro (evaluación,
# interfaz y reporte).
CARGABILIDAD_MAX_PCT = 100.0      # % de la corriente nominal de cada línea
TENSION_MIN_PU = 0.95             # banda de tensión admisible en barras
TENSION_MAX_PU = 1.05

# Márgenes de seguridad que se usan SOLO durante el entrenamiento: la
# recompensa exige estar un poco más adentro de los límites reales. Así el
# agente aprende a operar con colchón y, en evaluación, los pequeños errores
# de la política no se traducen en violaciones reales.
MARGEN_CARGABILIDAD_PCT = 2.0     # entrena contra 98 %
MARGEN_TENSION_PU = 0.005         # entrena contra [0.955, 1.045]

# Variante de la acción (ver entorno_despacho.py):
#   False -> el agente fija la P de los generadores PV; la slack cierra el
#            balance sola (y puede salirse de sus límites si la suma no cuadra).
#   True  -> el agente fija la P de TODOS los generadores y una capa de balance
#            física reparte el desbalance (carga + pérdidas) según la holgura de
#            cada máquina. Así despacha un operador real.
CAPA_BALANCE = True
PERDIDAS_ESTIMADAS_PU = 0.02   # estimación inicial de pérdidas (fracción de la demanda)
ITERACIONES_BALANCE = 4        # corrección de pérdidas con el flujo de carga

# Rango de las consignas de tensión que el agente puede dar a cada generador.
VG_MIN_PU = 0.95
VG_MAX_PU = 1.05

# Tolerancia numérica para declarar una violación en evaluación (evita contar
# como violación un 100.0000001 %).
TOL_VIOLACION = 1e-4


# =============================================================================
# 4. GENERACIÓN DE ESCENARIOS (LO QUE CAMBIA ENTRE EPISODIOS)
# =============================================================================
# Cada episodio es un escenario operativo nuevo: cargas y precios distintos.
# Esta diversidad es la que obliga al agente a GENERALIZAR en lugar de memorizar
# un despacho.
#
# Carga de cada barra:  P_l = P_l,base * FD * ruido_l
#   FD     : factor de demanda global, uniforme en [FD_MIN, FD_MAX].
#   ruido_l: variación individual de cada carga, normal(1, SIGMA_CARGA)
#            recortada a [1 - RECORTE_CARGA, 1 + RECORTE_CARGA].
# La reactiva sigue a la activa con su propio ruido pequeño (factor de potencia
# variable).
#
# El rango de FD se eligió con el OPF exacto de pandapower (240 escenarios):
#     FD 0.45-0.85 -> 100 % de escenarios con despacho seguro posible
#     FD 0.85-0.95 ->  90 %
#     FD 0.95-1.05 ->  53 %  (las líneas de 16 MVA de case30 no dan abasto)
# Por encima de ~0.9 se le estaría pidiendo al agente un imposible con mucha
# frecuencia. Ver pruebas.py -> estudio_factibilidad para repetir el estudio.
FD_MIN = 0.50
FD_MAX = 0.90
SIGMA_CARGA = 0.08
RECORTE_CARGA = 0.20
SIGMA_REACTIVA = 0.05

# Precio de oferta de cada generador (USD/MWh), uniforme e independiente en
# [PRECIO_MIN, PRECIO_MAX] en cada episodio. El costo de un despacho es
# sum_i precio_i * P_i, incluyendo la slack (que paga las pérdidas).
PRECIO_MIN = 20.0
PRECIO_MAX = 120.0


# =============================================================================
# 5. FUNCIÓN DE RECOMPENSA (ver recompensa.py para la explicación completa)
# =============================================================================
#   r = BONO_SEGURO * 1[despacho seguro]
#       - W_COSTO   * min(sobrecosto, SOBRECOSTO_MAX)
#       - W_LINEAS  * sum(sobrecarga de líneas en p.u. de su capacidad)
#       - W_TENSION * sum(desvío de tensión fuera de banda, en p.u.)
#       - W_SLACK   * exceso de la slack fuera de [Pmin, Pmax] / rango
#       - W_REACTIVA* sum(exceso de Q de generadores fuera de límites / rango)
#   Si el flujo de carga no converge: r = R_DIVERGENCIA.
BONO_SEGURO = 3.0
W_COSTO = 10.0
W_LINEAS = 20.0
W_TENSION = 200.0
W_SLACK = 20.0
W_REACTIVA = 20.0
R_DIVERGENCIA = -20.0
# Tope del sobrecosto relativo: solo acota valores absurdos (despachos que
# cuestan más del doble que la referencia) para que la recompensa sea acotada.
# En el rango útil (0-100 %) el término de costo es LINEAL: pendiente constante,
# el agente siempre "siente" que abaratar mejora.
#
# CONDICIÓN DE DISEÑO (la verifica pruebas.py):
#     BONO_SEGURO > W_COSTO * (máximo sobrecosto del OPF seguro)
# El OPF seguro en case30 queda a lo sumo ~15 % sobre el orden de mérito, así
# que su recompensa es >= 3 - 10·0.15 = 1.5 > 0, mientras que cualquier
# despacho inseguro tiene recompensa <= 0. Por tanto, en CADA escenario el
# despacho que maximiza la recompensa es el despacho seguro más barato, es
# decir, el OPF (con márgenes). Eso es exactamente lo que queremos que aprenda.
SOBRECOSTO_MAX = 1.0


# =============================================================================
# 6. HIPERPARÁMETROS PPO POR DEFECTO
# =============================================================================
# Se usan si todavía no se ha corrido Optuna. Si existe
# RUTA_MEJORES_HIPERPARAMETROS, entrenar.py usa esos en su lugar.
#
# Nota sobre gamma y gae_lambda: cada episodio es de UN paso (el despacho de un
# escenario es una decisión estática), así que el retorno es la recompensa
# inmediata y gamma / gae_lambda no tienen efecto práctico. Se dejan en valores
# estándar y no se buscan con Optuna.
HIPERPARAMETROS_PPO_DEFECTO = {
    "learning_rate": 3e-4,
    # "constante" o "lineal" (decae linealmente hasta 0 al final del
    # entrenamiento: pasos grandes al inicio, ajuste fino al final).
    "esquema_lr": "lineal",
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "arquitectura": "media",      # ver ARQUITECTURAS
    "activacion": "tanh",
    "log_std_init": -0.5,
}

# Arquitecturas candidatas de la red neuronal (actor y crítico separados).
ARQUITECTURAS = {
    "pequena": [64, 64],
    "media": [256, 256],
    "grande": [256, 256, 256],
}

# Semilla de los escenarios de VALIDACIÓN (selección de trials en Optuna y del
# mejor checkpoint en el entrenamiento). Distinta a la de entrenamiento y a la
# del conjunto de PRUEBA, que nunca se usa para elegir nada.
SEMILLA_VALIDACION = 2024
ESCENARIOS_VALIDACION = 300

# Entornos en paralelo. El ambiente es tan rápido que el cuello de botella es la
# red neuronal; con DummyVecEnv (mismo proceso) basta y funciona igual en
# Windows, Linux y macOS.
N_ENVS = 8
SEMILLA = 42


# =============================================================================
# 7. BÚSQUEDA DE HIPERPARÁMETROS (OPTUNA)
# =============================================================================
OPTUNA_N_TRIALS = 30
OPTUNA_PASOS_POR_TRIAL = 300_000
OPTUNA_N_EVALUACIONES = 6          # evaluaciones intermedias (para el pruning)
OPTUNA_NOMBRE_ESTUDIO = "despacho_ppo"

# Espacio de búsqueda (lo usa optimizar_hiperparametros.py).
ESPACIO_BUSQUEDA = {
    "learning_rate": (1e-5, 1e-3),            # log-uniforme
    "esquema_lr": ["constante", "lineal"],
    "n_steps": [64, 128, 256, 512, 1024],     # por entorno (x N_ENVS = rollout)
    "batch_size": [64, 128, 256, 512],
    "n_epochs": [5, 10, 20],
    "clip_range": [0.1, 0.2, 0.3],
    "ent_coef": (1e-8, 1e-2),                 # log-uniforme
    "arquitectura": list(ARQUITECTURAS.keys()),
    "activacion": ["tanh", "relu"],
    "log_std_init": (-3.0, 0.0),
}


# =============================================================================
# 8. ENTRENAMIENTO FINAL Y EVALUACIÓN
# =============================================================================
PASOS_ENTRENAMIENTO_FINAL = 3_000_000
# Se entrena un agente por semilla, en paralelo (un proceso por semilla), y se
# queda el de mejor validación. Con 2 núcleos, 2 semillas cuestan lo mismo en
# tiempo que 1.
SEMILLAS_ENTRENAMIENTO = [42, 7]
EVAL_CADA_PASOS = 100_000

# Conjunto de prueba (nunca visto en entrenamiento ni en Optuna) que se compara
# contra el OPF exacto de pandapower.
ESCENARIOS_PRUEBA = 500
SEMILLA_PRUEBA = 7777


# =============================================================================
# 9. INTERFAZ WEB
# =============================================================================
INTERFAZ_HOST = "127.0.0.1"
INTERFAZ_PUERTO = 5050

# Modo "Simular 24 h" de la interfaz: perfil horario de demanda (0 = valle,
# 1 = pico) que se lleva al rango [FD_MIN, FD_MAX], con picos de mañana y
# noche típicos de un día hábil.
CURVA_DEMANDA_DIARIA = [0.18, 0.10, 0.05, 0.03, 0.06, 0.18, 0.42, 0.66, 0.78, 0.80,
                        0.78, 0.76, 0.72, 0.70, 0.70, 0.72, 0.78, 0.88, 1.00, 0.98,
                        0.90, 0.74, 0.52, 0.32]
# Los precios de cada generador oscilan durante el día alrededor del valor que
# fije el usuario: precio_h = precio * (1 + VARIACION_PRECIO_DIA * sen(...)),
# con un desfase distinto por máquina (así el orden de mérito cambia en el día).
VARIACION_PRECIO_DIA = 0.35
