"""
escenarios.py
=============
Generador de ESCENARIOS OPERATIVOS: lo que cambia entre episodios.

Un escenario es la "pregunta" que se le hace al agente:
    * cuánto consume cada carga (P y Q), y
    * a qué precio oferta cada generador síncrono.

El agente responde con un despacho. Como en cada episodio el escenario es
distinto (demanda total, reparto geográfico de la demanda y orden de mérito de
los precios cambian), el agente no puede memorizar una respuesta: tiene que
aprender la RELACIÓN entre cargas/precios y despacho económico-seguro, es
decir, generalizar.

Todos los rangos vienen de config.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config as C


@dataclass
class Escenario:
    p_carga: np.ndarray   # MW por carga
    q_carga: np.ndarray   # MVAr por carga
    precios: np.ndarray   # USD/MWh por generador [slack, PV...]
    fd: float             # factor de demanda global con que se generó

    @property
    def demanda_total_mw(self) -> float:
        return float(self.p_carga.sum())


class GeneradorEscenarios:
    """Muestrea escenarios aleatorios reproducibles (con semilla)."""

    def __init__(self, red, semilla: int | None = None):
        self.red = red
        self.rng = np.random.default_rng(semilla)

    def reiniciar_semilla(self, semilla: int | None):
        self.rng = np.random.default_rng(semilla)

    def muestrear(self) -> Escenario:
        rng, red = self.rng, self.red
        # 1) Nivel general de demanda del sistema.
        fd = rng.uniform(C.FD_MIN, C.FD_MAX)
        # 2) Variación individual de cada carga (no todas las barras suben o
        #    bajan igual: cambia el reparto geográfico y por ende los flujos).
        ruido = np.clip(rng.normal(1.0, C.SIGMA_CARGA, red.n_cargas),
                        1.0 - C.RECORTE_CARGA, 1.0 + C.RECORTE_CARGA)
        p = red.p_carga_base * fd * ruido
        # 3) Reactiva: sigue a la activa con un ruido propio pequeño
        #    (variación del factor de potencia).
        ruido_q = np.clip(rng.normal(1.0, C.SIGMA_REACTIVA, red.n_cargas),
                          1.0 - C.RECORTE_CARGA, 1.0 + C.RECORTE_CARGA)
        q = red.q_carga_base * fd * ruido * ruido_q
        # 4) Precios de oferta independientes: cambian el orden de mérito.
        precios = rng.uniform(C.PRECIO_MIN, C.PRECIO_MAX, red.n_gen)
        return Escenario(p, q, precios, float(fd))

    def desde_valores(self, fd: float, precios) -> Escenario:
        """Escenario determinista: todas las cargas escaladas por fd (sin ruido).
        Lo usa la interfaz cuando el usuario mueve los controles."""
        red = self.red
        return Escenario(red.p_carga_base * fd, red.q_carga_base * fd,
                         np.asarray(precios, float), float(fd))

    def lote(self, n: int) -> list[Escenario]:
        return [self.muestrear() for _ in range(n)]
