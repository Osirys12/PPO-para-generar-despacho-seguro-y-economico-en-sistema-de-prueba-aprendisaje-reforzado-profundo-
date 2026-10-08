# Comparación: con y sin capa de balance

Mismos hiperparámetros (los ganadores de Optuna), mismo presupuesto (3 millones de episodios × 2
semillas), mismo conjunto de prueba (500 escenarios nunca vistos) y mismo juez (flujo de carga AC
con límites reales).

- **A — sin capa de balance** (`CAPA_BALANCE = False`): el agente fija la P de los 5 generadores PV;
  la slack cierra el balance por su cuenta.
- **B — con capa de balance** (`CAPA_BALANCE = True`, la configuración final): el agente fija la P de
  los 6 generadores y la capa reparte el desbalance según la holgura de cada máquina.

| Métrica | A: sin capa de balance (11 acciones) | B: con capa de balance (12 acciones) |
|---|---|---|
| Recompensa de validación (semillas 7 / 42) | 1.631, 1.771 | 1.842, 1.807 |
| % despachos seguros (prueba) | 93.0 | 95.8 |
| Brecha media vs OPF [%] | 4.62 | 3.72 |
| Brecha mediana vs OPF [%] | 3.87 | 3.29 |
| Brecha p90 vs OPF [%] | 10.03 | 7.84 |
| % escenarios con sobrecarga | 4.0 | 4.0 |
| % escenarios con slack fuera de límites | 3.0 | 0.0 |
| % escenarios con Q fuera de límites | 0.2 | 0.2 |
| Agente + respaldo OPF: brecha media [%] | 4.69 | 3.85 |
| Agente + respaldo OPF: % usa respaldo | 7.0 | 4.2 |
| Agente + respaldo OPF: ms por despacho | 22.8 | 14.3 |

## Por nivel de demanda

| FD | A % seguro | B % seguro | A brecha [%] | B brecha [%] |
|---|---|---|---|---|
| 0.50-0.60 | 96.3 | 99.3 | 5.10 | 3.89 |
| 0.60-0.70 | 96.2 | 100.0 | 4.64 | 3.66 |
| 0.70-0.80 | 96.7 | 98.3 | 4.43 | 4.05 |
| 0.80-0.90 | 81.4 | 84.1 | 4.22 | 3.24 |

## Conclusión

B es igual o mejor en todas las métricas: ambas semillas de B superan a ambas de A en validación, la
seguridad sube de 93.0 % a 95.8 %, la brecha frente al OPF baja de
4.62 % a 3.72 % y las violaciones de la slack desaparecen
(3.0 % → 0.0 %). Sin la capa, el agente tenía que acertar la suma exacta
de cinco potencias para que la slack quedara dentro de sus límites; con la capa solo decide el
reparto, que es donde están la economía y la congestión. Por eso B es la configuración por
defecto. Las sobrecargas que quedan se concentran en demanda alta (FD 0.8–0.9).

Reproducir:
```
python main.py evaluar                                       # B (modelo final)
python main.py evaluar --modelo resultados/comparativa_capa/modelos_sin_capa/despacho_ppo_final.zip --capa no --carpeta resultados/comparativa_capa/evaluacion_sin_capa
```
