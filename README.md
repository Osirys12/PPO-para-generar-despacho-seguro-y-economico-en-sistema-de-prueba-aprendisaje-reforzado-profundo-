# Despacho económico seguro con PPO sobre pandapower

Agente de aprendizaje por refuerzo profundo (**PPO**, Stable-Baselines3) que aprende a
**despachar los generadores síncronos** de una red de **pandapower** de forma:

- **segura** — ninguna línea por encima de su capacidad, tensiones en banda, la slack dentro
  de sus límites de P y todos los generadores dentro de sus límites de Q;
- **económica** — entre los despachos seguros, el más barato según el precio de cada máquina.

**Entradas del agente:** la carga (P y Q) de cada barra y el precio de oferta de cada generador.
**Salidas:** la potencia activa de cada generador PV y la consigna de tensión de todos.
Cuando cambian las cargas o los precios, el agente da un despacho nuevo en milisegundos.

## Resultados (500 escenarios de prueba nunca vistos, `case30`)

| Método | % despachos seguros | Sobrecosto medio vs OPF exacto | Tiempo por despacho |
|---|---|---|---|
| **Agente PPO** | **95.8 %** | **3.72 %** (mediana 3.29 %) | **1.1 ms** |
| **Agente PPO + respaldo OPF** | **100 %** | **3.85 %** | **14.3 ms** |
| OPF AC exacto (`pp.runopp`) | 100 % | 0 % (es la referencia) | 313 ms |
| Orden de mérito sin red | 26.6 % | -1.10 % (barato porque sobrecarga líneas, hasta 161 %) | — |

- El agente despacha en 1.1 ms incluyendo el flujo de carga AC que verifica su despacho:
  **287 veces más rápido** que el OPF, a 3.7 % de su costo.
- Nunca viola tensiones ni la slack; sus violaciones (4.2 % de los escenarios) son
  sobrecargas, casi todas con demanda alta (FD 0.8–0.9: 84.1 % seguro; con FD ≤ 0.8 es
  seguro en el 98–100 % de los casos).
- **Modo de operación recomendado — agente + respaldo OPF:** como el flujo de carga del despacho del
  agente ya está calculado, se sabe al instante si es seguro; solo cuando no lo es
  (4.2 % de los casos) se corre el OPF. Resultado: 100 % seguro,
  3.85 % de sobrecosto y 22 veces más rápido que usar siempre el OPF.
- Despachar solo por precio (orden de mérito) es seguro apenas en el 26.6 % de los escenarios:
  la red sí importa y el agente la aprendió.

Detalle completo: `resultados/evaluacion/reporte_evaluacion.md`. Comparación de las dos variantes de
acción (con y sin capa de balance): `resultados/comparativa_capa/README.md`. Hiperparámetros
ganadores de Optuna (14 trials completos, 17 podados):
`resultados/optuna/mejores_hiperparametros.json`.

Las curvas de validación (`resultados/modelos/curva_validacion.png`) siguen mejorando al final de
los 3 millones de episodios: entrenar más (`python main.py entrenar --pasos 6000000`) debería
reducir todavía el sobrecosto y las sobrecargas en demanda alta.


---

## 1. Inicio rápido

Python 3.10–3.12. En Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py probar        # 9 verificaciones automáticas, deben salir todas OK
python main.py interfaz      # abrir http://127.0.0.1:5050
```

En Linux/macOS es igual cambiando la activación por `source .venv/bin/activate`.

El paquete **ya trae el modelo entrenado y los resultados de Optuna**, así que la interfaz y
la evaluación funcionan sin entrenar. Para reproducir todo desde cero:

```bash
python main.py optimizar --trabajadores 2   # 1. Optuna (≈1.5 h con 2 núcleos; se puede reanudar)
python main.py entrenar                     # 2. entrenamiento final, 2 semillas en paralelo (≈80 min)
python main.py evaluar                      # 3. comparación con el OPF exacto (≈4 min la 1.ª vez)
python main.py interfaz                     # 4. panel web
```

## 2. Estructura

```
despacho_ppo/
├── config.py                    ÚNICA fuente de verdad: red, límites, escenarios, recompensa,
│                                hiperparámetros, Optuna, entrenamiento, interfaz
├── main.py                      punto de entrada: python main.py <probar|optimizar|entrenar|evaluar|interfaz>
├── red_electrica.py             todo lo de pandapower: flujo de carga AC, costo, orden de mérito, OPF exacto
├── escenarios.py                lo que cambia entre episodios: cargas y precios
├── recompensa.py                función de recompensa, documentada término a término
├── entorno_despacho.py          ambiente Gymnasium (observación, acción, paso)
├── agente.py                    construir/cargar PPO, despachar, evaluar política, callback de evaluación
├── optimizar_hiperparametros.py búsqueda bayesiana con Optuna (TPE + poda por mediana)
├── entrenar.py                  entrenamiento final y guardado del mejor checkpoint
├── evaluar.py                   agente vs OPF exacto vs orden de mérito, figuras y reporte
├── pruebas.py                   verificaciones automáticas
├── interfaz/
│   ├── servidor.py              servidor Flask
│   ├── panel.html               panel web (estilo FLEXIBILITY)
│   └── static/chart.umd.min.js  Chart.js local: el panel funciona sin internet
├── requirements.txt
└── resultados/
    ├── optuna/                  estudio, trials, mejores hiperparámetros, figuras
    ├── modelos/                 modelo final, historia de validación por semilla
    ├── evaluacion/              reporte, métricas, detalle por escenario, figuras
    ├── comparativa_capa/        variante sin capa de balance: modelo, evaluación y comparación
    └── capturas_interfaz/       capturas del panel con el modelo final
```

Cada archivo empieza con un bloque que explica qué hace y por qué. Ningún archivo tiene números
"mágicos": todo lo que gobierna el comportamiento está en `config.py`, comentado.

## 3. La red

`case30` de pandapower (IEEE 30 barras, versión MATPOWER): 6 generadores síncronos
(G1 en la barra 1 es la slack, G2–G6 son PV), 20 cargas y 41 líneas a 135 kV con **límites
térmicos reales** (varias de solo 16 MVA). Es una red congestionada a propósito: despachar solo
por precio sobrecarga líneas en la gran mayoría de los casos, así que el agente está obligado a
aprender la red, no solo el orden de mérito.

Para usar otra red basta cambiar `CASO_RED` en `config.py` (por ejemplo `"case39"`, la IEEE de
39 barras). Hay que revisar el rango de demanda factible con `python main.py probar --completo`
y volver a correr Optuna y el entrenamiento.

### Motor del flujo de carga

El flujo de carga AC (Newton-Raphson) se resuelve sobre el modelo pandapower con
**lightsim2grid**, el mismo motor acelerado que pandapower usa con `pp.runpp(lightsim2grid=True)`,
pero llamado directamente para evitar la conversión de datos en cada paso: ~0.2 ms por flujo en
lugar de 8–30 ms. Eso es lo que permite entrenar con millones de escenarios y correr Optuna en
serio. `pruebas.py` verifica en cada ejecución que los resultados son idénticos a los de
`pp.runpp` (diferencias < 10⁻⁶). Si se quiere usar `pp.runpp` directamente: `MOTOR_FLUJO =
"pandapower"` en `config.py` (todo funciona igual, solo que mucho más lento).

## 4. Formulación

### Un episodio = un escenario = un paso

El despacho económico de un instante es una decisión **estática**: para unas cargas y unos
precios dados existe un despacho óptimo que no depende de lo que se hizo antes. Por eso cada
episodio es de **un paso**: se sortea un escenario, el agente despacha, se resuelve el flujo de
carga, se calcula la recompensa y termina. Lo que el agente aprende es la función
`(cargas, precios) → despacho`. (En consecuencia `gamma` y `gae_lambda` de PPO no influyen.)

### Diversidad entre episodios (generalización)

En cada episodio cambian (`escenarios.py`, rangos en `config.py`):

| Qué | Cómo | Para qué |
|---|---|---|
| Nivel de demanda | factor de demanda FD ∈ [0.50, 0.90] | demanda baja y alta |
| Reparto de la demanda | ruido individual por carga, σ = 8 % | cambian los flujos por las líneas |
| Factor de potencia | ruido propio de Q, σ = 5 % | cambian las necesidades de reactiva |
| Precios | cada generador U(20, 120) USD/MWh | cambia el orden de mérito |

El rango de FD se eligió con el OPF exacto: hasta 0.85 siempre existe un despacho seguro,
entre 0.85 y 0.95 en el 90 % de los casos y por encima de 0.95 apenas en la mitad (las líneas
de 16 MVA no dan abasto). No tiene sentido pedirle al agente un imposible.

### Observación (46 valores)

`[P_carga / P_base (20), Q_carga / Q_base (20), precio normalizado a [-1, 1] (6)]`

### Acción (12 valores en [-1, 1])

`[P de G1…G6 (6), V de G1…G6 (6)]`, mapeados a `[Pmin, Pmax]` y `[0.95, 1.05]` p.u. Los límites
de P de cada máquina se cumplen por construcción.

El agente da consigna de P a **todas** las máquinas, slack incluida, como lo haría un operador.
Como lo que pide no tiene por qué sumar exactamente carga + pérdidas, una **capa de balance**
física lo ajusta: si falta potencia, cada máquina sube en proporción a su holgura hacia arriba;
si sobra, baja en proporción a su holgura hacia abajo; luego se resuelve el flujo, se miden las
pérdidas reales y se repite hasta que la slack queda a menos de 0.001 MW de su consigna. El
agente sigue decidiendo el reparto (qué máquinas generan, cuánto, y las tensiones), que es donde
están la economía y la congestión. Se puede desactivar con `CAPA_BALANCE = False`: entonces el
agente solo fija las P de los PV (11 acciones) y la slack cierra el balance por su cuenta. La
comparación entre ambas variantes está en la sección de resultados.

### Recompensa (`recompensa.py`)

```
r =  3 · 1[despacho seguro]                         bono de seguridad
   − 10 · sobrecosto relativo                       economía
   − 20 · Σ sobrecarga de ramas (p.u.)              cargabilidad
   − 200 · Σ desvío de tensión fuera de banda (p.u.) tensiones
   − 20 · exceso de la slack / rango                balance de potencia
   − 20 · Σ exceso de Q / rango                     capacidad de reactiva
si el flujo no converge: r = −20
```

- **Sobrecosto relativo** = (C_agente − C_mérito) / C_mérito, donde C_mérito es el costo del
  orden de mérito sin red (el más barato imaginable). En relativo, "5 % por encima" significa lo
  mismo en un escenario de 90 MW que en uno de 170 MW.
- **Jerarquía seguridad → economía:** un despacho inseguro tiene recompensa ≤ 0, mientras que el
  despacho seguro más barato (el OPF) tiene `3 − 10·sobrecosto_OPF > 0`, porque el sobrecosto del
  OPF nunca supera ~15 % (verificado por `pruebas.py`). Por tanto **el máximo de la recompensa en
  cada escenario es el OPF seguro**: ninguna violación se puede "pagar" con ahorro.
- **Márgenes de entrenamiento:** la recompensa exige 98 % de cargabilidad y [0.955, 1.045] p.u.
  El agente aprende a dejar colchón y, evaluado contra los límites reales (100 %, [0.95, 1.05]),
  sus pequeños errores no se vuelven violaciones.
- Se probó también una variante que garantiza "cualquier despacho seguro > cualquier inseguro"
  saturando el costo; aplanaba la pendiente del costo y el aprendizaje económico casi se detenía.
  La justificación completa está en el encabezado de `recompensa.py`.

## 5. Hiperparámetros (Optuna)

`optimizar_hiperparametros.py`: búsqueda bayesiana (TPE) sobre tasa de aprendizaje y su esquema
(constante o lineal), tamaño del rollout, minilote, épocas, `clip_range`, coeficiente de
entropía, arquitectura (64×64, 256×256, 256×256×256), activación y exploración inicial
(`log_std_init`). Cada trial entrena 300 mil pasos y se evalúa 6 veces sobre 300 escenarios de
validación **fijos** (iguales para todos los trials → comparación justa). Los trials que van
peor que la mediana se podan. El estudio se guarda en SQLite: si se interrumpe, se reanuda. El
paquete trae el estudio ya hecho, así que `optimizar` lo **continúa**; para empezar una búsqueda
desde cero, borrar `resultados/optuna/estudio.db`.

## 6. Entrenamiento y evaluación

`entrenar.py` usa los hiperparámetros ganadores y entrena dos agentes (semillas 42 y 7) en
paralelo, 3 millones de episodios cada uno (escenarios distintos). Cada 100 mil evalúa sobre los
escenarios de validación y guarda el **mejor** checkpoint; el modelo final es el de la semilla con
mejor validación. (El paquete entregado conserva solo el modelo final y la historia de cada
semilla, para no duplicar archivos.)

`evaluar.py` mide al agente en **500 escenarios de prueba nunca vistos** (semilla propia, nunca
usada para elegir nada) y lo compara con:

- **OPF AC exacto** de pandapower (`pp.runopp`): el óptimo verdadero, para medir la brecha de costo;
- **orden de mérito sin red**: lo que pasaría despachando solo por precio.

Todos se juzgan con el mismo flujo de carga AC y los límites reales. Resultados en
`resultados/evaluacion/reporte_evaluacion.md`.

## 7. Interfaz

`python main.py interfaz` → http://127.0.0.1:5050

- Controles de **factor de demanda**, **perfil de cargas** (uniforme o aleatorio) y **precio de
  cada generador**. Cada cambio es un escenario nuevo: el agente despacha y pandapower resuelve el
  flujo de carga.
- KPIs: costo, **brecha vs OPF**, demanda, cargabilidad máxima, tensiones y estado de seguridad.
- Despacho por generador (agente vs OPF vs orden de mérito), diagrama de la red coloreado por
  cargabilidad y tensión, cargabilidad de las 41 ramas, tensiones de las 30 barras y reactiva de
  cada máquina contra sus límites.
- **Simular 24 h**: demanda con picos de mañana y noche y precios que cambian en el día; muestra
  costo horario (agente vs OPF vs mérito), cargabilidad máxima y despacho por hora.

Capturas con el modelo final en `resultados/capturas_interfaz/` (escenario FD = 0.82: agente a
+1.7 % del OPF y seguro; día completo: agente seguro 24/24 h, orden de mérito sin red 2/24 h).

## 8. Cómo modificar

| Quiero… | Cambiar en `config.py` |
|---|---|
| otra red | `CASO_RED` (y revisar `FD_MIN`, `FD_MAX`) |
| otro rango de demanda o precios | `FD_MIN`, `FD_MAX`, `PRECIO_MIN`, `PRECIO_MAX` |
| límites de seguridad | `CARGABILIDAD_MAX_PCT`, `TENSION_MIN_PU`, `TENSION_MAX_PU`, `MARGEN_*` |
| pesos de la recompensa | `BONO_SEGURO`, `W_*` (mantener `BONO_SEGURO > W_COSTO · 0.15`) |
| espacio de búsqueda de Optuna | `ESPACIO_BUSQUEDA`, `OPTUNA_*` |
| duración del entrenamiento | `PASOS_ENTRENAMIENTO_FINAL` |

Después de cambiar la red, los rangos o la recompensa hay que volver a correr `optimizar` y
`entrenar` (el modelo incluido está entrenado para la configuración actual).

## 9. Problemas comunes

- **`pip` no encuentra `lightsim2grid`**: actualizar pip (`python -m pip install -U pip`). Si aun
  así falla, poner `MOTOR_FLUJO = "pandapower"` en `config.py` (funciona, pero entrenar es mucho
  más lento).
- **PyTorch pesado en Windows sin GPU**: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- **El panel no carga gráficas**: Chart.js va incluido en `interfaz/static`; revisar que la
  carpeta exista.
- **Optuna con varios procesos en Windows**: siempre lanzar con `python main.py optimizar ...`
  (el bloque `if __name__ == "__main__"` es obligatorio para multiprocessing en Windows).
