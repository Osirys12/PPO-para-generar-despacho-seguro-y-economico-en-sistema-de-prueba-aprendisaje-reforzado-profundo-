"""
main.py
=======
Punto de entrada único del proyecto. Flujo completo recomendado:

    python main.py probar              # 1. verifica instalación, ambiente y recompensa (~30 s)
    python main.py optimizar           # 2. búsqueda de hiperparámetros con Optuna
    python main.py entrenar            # 3. entrenamiento final con los mejores hiperparámetros
    python main.py evaluar             # 4. comparación contra el OPF exacto y el orden de mérito
    python main.py interfaz            # 5. panel web en http://127.0.0.1:5050

Opciones útiles:
    python main.py probar --completo                  # + estudio de factibilidad con OPF
    python main.py optimizar --trabajadores 2         # Optuna en 2 procesos
    python main.py optimizar --trials 40 --pasos 300000
    python main.py entrenar --pasos 1000000 --semillas 1 2

El repositorio ya incluye un modelo entrenado y los resultados de Optuna, así
que los pasos 2-4 son opcionales: `python main.py interfaz` funciona directo.
"""

from __future__ import annotations

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description="Despacho económico seguro con PPO (pandapower)")
    sub = ap.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("probar", help="Pruebas automáticas")
    p.add_argument("--completo", action="store_true", help="incluye estudio de factibilidad con OPF")

    o = sub.add_parser("optimizar", help="Búsqueda de hiperparámetros con Optuna")
    o.add_argument("--trials", type=int, default=None)
    o.add_argument("--pasos", type=int, default=None)
    o.add_argument("--trabajadores", type=int, default=1)

    e = sub.add_parser("entrenar", help="Entrenamiento final del agente")
    e.add_argument("--pasos", type=int, default=None)
    e.add_argument("--semillas", type=int, nargs="+", default=None)

    ev = sub.add_parser("evaluar", help="Evaluación contra OPF exacto y orden de mérito")
    ev.add_argument("--modelo", default=None, help="otro .zip (por defecto el modelo final)")
    ev.add_argument("--capa", choices=["si", "no"], default=None,
                    help="variante de acción del modelo (por defecto config.CAPA_BALANCE)")
    ev.add_argument("--carpeta", default=None, help="carpeta de salida")
    sub.add_parser("interfaz", help="Panel web")

    a = ap.parse_args()
    import config as C

    if a.comando == "probar":
        import pruebas
        sys.exit(pruebas.ejecutar(a.completo))
    elif a.comando == "optimizar":
        import optimizar_hiperparametros as opt
        opt.ejecutar(a.trials or C.OPTUNA_N_TRIALS, a.pasos or C.OPTUNA_PASOS_POR_TRIAL, a.trabajadores)
    elif a.comando == "entrenar":
        import entrenar
        entrenar.ejecutar(a.pasos or C.PASOS_ENTRENAMIENTO_FINAL, a.semillas)
    elif a.comando == "evaluar":
        import evaluar
        evaluar.ejecutar(a.modelo, None if a.capa is None else a.capa == "si", a.carpeta)
    elif a.comando == "interfaz":
        from interfaz import servidor
        servidor.ejecutar()


if __name__ == "__main__":
    # Necesario en Windows para Optuna con varios procesos (multiprocessing "spawn").
    main()
