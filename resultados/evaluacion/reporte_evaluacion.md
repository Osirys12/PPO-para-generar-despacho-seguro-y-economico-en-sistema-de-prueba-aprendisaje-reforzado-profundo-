# Evaluación del agente PPO de despacho

Variante de acción: con capa de balance.
Conjunto de prueba: **500 escenarios nunca vistos** (semilla 7777),
FD ∈ [0.5, 0.9], precios ∈ [20, 120] USD/MWh, red `case30`.
El OPF exacto tiene solución en **500** de ellos; las métricas se calculan sobre esos.

Todos los despachos se juzgan con el mismo flujo de carga AC y los límites reales
(cargabilidad ≤ 100 %, tensión en [0.95, 1.05] p.u.,
P de la slack y Q de todos los generadores dentro de sus límites).

| Método | % seguro | Brecha media vs OPF [%] | Brecha mediana [%] | Brecha p90 [%] | Peor cargabilidad [%] | Tensiones extremas [p.u.] |
|---|---|---|---|---|---|---|
| Agente PPO | 95.2 | 3.75 | 3.34 | 7.54 | 111.0 | 0.970 – 1.046 |
| OPF exacto (referencia) | 100.0 | 0.00 | 0.00 | 0.00 | 100.0 | 0.950 – 1.050 |
| Orden de mérito sin red | 26.6 | -1.10 | -0.13 | 1.39 | 160.8 | 0.950 – 1.000 |

| Agente + respaldo OPF¹ | 100.0 | 3.85 | 3.30 | 7.39 | 100.0 | — |

¹ Política híbrida: se usa el despacho del agente si su flujo de carga es seguro (se verifica al
instante) y el OPF solo cuando no lo es (4.8 % de los escenarios).
Tiempo medio **15.5 ms** por despacho, **20× más rápido** que usar
siempre el OPF, con 100 % de despachos seguros.

Brecha media del agente **solo en sus despachos seguros**: 4.05 %.

Nota: el orden de mérito sin red puede salir *más barato* que el OPF (brecha negativa) porque
ignora los límites de la red: es barato precisamente porque sobrecarga líneas. Por eso la
brecha de costo solo tiene sentido junto con el % de despachos seguros.

Tiempo por despacho (red neuronal + capa de balance + flujo de carga AC de verificación):
agente **0.43 ms** vs OPF **313 ms**
→ el agente es **730× más rápido**.

Tipos de violación del agente (en % de escenarios): sobrecarga 4.6 %,
tensión fuera de banda 0.0 %, slack fuera de límites 0.0 %,
reactiva fuera de límites 0.2 %.

## Por nivel de demanda

| FD | n | Agente % seguro | Agente brecha media [%] | Mérito sin red % seguro |
|---|---|---|---|---|
| 0.50-0.60 | 134 | 99.3 | 3.44 | 49.3 |
| 0.60-0.70 | 133 | 99.2 | 4.09 | 33.1 |
| 0.70-0.80 | 120 | 98.3 | 4.02 | 14.2 |
| 0.80-0.90 | 113 | 82.3 | 3.41 | 5.3 |

## Figuras

- `costo_agente_vs_opf.png` — costo del agente frente al óptimo, escenario por escenario.
- `histograma_brecha.png` — distribución de la brecha de costo.
- `seguridad_y_cargabilidad.png` — % de despachos seguros y cargabilidad máxima por método.
- `ejemplo_despacho.png` — un escenario concreto: agente vs OPF vs orden de mérito.
