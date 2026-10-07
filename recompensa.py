"""
recompensa.py
=============
FUNCIÓN DE RECOMPENSA del agente de despacho y cálculo de VIOLACIONES.

-----------------------------------------------------------------------------
QUÉ QUEREMOS QUE APRENDA EL AGENTE
-----------------------------------------------------------------------------
Dado un escenario (cargas + precios), producir un despacho que sea, EN ESTE
ORDEN DE PRIORIDAD:
    1. SEGURO:     ninguna línea/trafo por encima de su capacidad, todas las
                   tensiones dentro de banda, la slack dentro de sus límites de
                   P y todos los generadores dentro de sus límites de Q.
    2. ECONÓMICO:  entre los despachos seguros, el más barato posible.

Esa jerarquía es exactamente la de un OPF: las restricciones de seguridad son
duras y el costo es lo que se minimiza dentro de ellas.

-----------------------------------------------------------------------------
LA RECOMPENSA (un solo paso por episodio)
-----------------------------------------------------------------------------

    r =  BONO_SEGURO · 1[despacho seguro]                      (a)
       − W_COSTO     · min(sobrecosto, SOBRECOSTO_MAX)         (b)
       − W_LINEAS    · Σ_ramas max(0, carga_k − Lmax) / 100    (c)
       − W_TENSION   · Σ_barras dist(V_b, [Vmin, Vmax])        (d)
       − W_SLACK     · exceso_slack / (Pmax_s − Pmin_s)        (e)
       − W_REACTIVA  · Σ_gen exceso_Q_i / (Qmax_i − Qmin_i)    (f)

    si el flujo de carga NO converge:  r = R_DIVERGENCIA        (g)

(a) BONO DE SEGURIDAD. Un premio fijo que solo se gana si TODAS las
    restricciones se cumplen (con margen). Crea un "escalón" entre despachos
    seguros e inseguros que impone la jerarquía "seguridad primero":
    un despacho inseguro tiene recompensa <= 0 (solo le quedan términos
    negativos), mientras que el despacho seguro más barato (el OPF) tiene
    recompensa  BONO − W_COSTO·sobrecosto_OPF  > 0  siempre que
        BONO_SEGURO > W_COSTO · (máximo sobrecosto del OPF seguro).
    En case30 ese máximo es ~15 %, así que 3 > 10·0.15 = 1.5 con holgura
    (pruebas.py lo verifica). CONSECUENCIA: en cada escenario, el despacho que
    MAXIMIZA la recompensa es el OPF seguro. Ninguna violación "se paga" con
    ahorro, porque el ahorro máximo posible respecto al OPF es menor que el bono.

    Nota de diseño: se probó también exigir que CUALQUIER despacho seguro (por
    caro que sea) supere a CUALQUIER inseguro, saturando el costo con tanh.
    Funciona en teoría, pero aplana la pendiente del costo justo donde está el
    agente al inicio (40-60 % de sobrecosto) y el aprendizaje económico casi se
    detiene (piloto: 62 % -> 57 % de sobrecosto en 200 mil pasos, contra
    52 % -> 27 % con costo lineal). La condición de arriba es la que realmente
    importa: que el óptimo de la recompensa sea el óptimo del problema.

(b) SOBRECOSTO RELATIVO.
        sobrecosto = (C_agente − C_mérito) / C_mérito      (≥ 0 en la práctica)
    C_agente = Σ precio_i · P_i  (incluye la slack, que paga las pérdidas).
    C_mérito = costo del despacho de orden de mérito "placa de cobre" (sin red):
               el despacho más barato imaginable si no hubiera líneas.
    Se usa el costo RELATIVO, no el absoluto, porque un escenario con 170 MW de
    demanda y precios altos cuesta 10 veces más que uno con 90 MW y precios
    bajos; en términos absolutos la recompensa variaría con el escenario y no
    con la calidad del despacho. En relativo, "5 % por encima de la referencia"
    significa lo mismo en cualquier escenario. Como el OPF real ya queda en
    promedio ~5 % por encima del orden de mérito (por la congestión y las
    pérdidas), el agente nunca puede llegar a 0: lo que se busca es acercarse
    al OPF. El término es LINEAL (pendiente constante: abaratar siempre mejora
    la recompensa en la misma medida) y solo se topa en SOBRECOSTO_MAX = 100 %
    para que la recompensa quede acotada.

(c) SOBRECARGA DE RAMAS. Suma de los excesos de cargabilidad en por unidad de
    la capacidad (una línea al 110 % aporta 0.10). Es continua y proporcional:
    al agente le duele más sobrecargar mucho que poco, y le duele más
    sobrecargar varias ramas que una. Eso le da una dirección clara de mejora
    incluso cuando todavía no logra el bono.

(d) TENSIÓN FUERA DE BANDA. Suma de las distancias (p.u.) de cada barra a la
    banda admisible. El peso es grande porque las desviaciones son numéricamente
    pequeñas (0.01 p.u. ya es una violación importante): 0.01 p.u. × 200 = 2.

(e) LÍMITES DE LA SLACK. El agente NO decide la potencia de la slack: sale del
    balance (carga + pérdidas − lo que despachó el agente). Si el agente
    despacha de menos, la slack se va por encima de su Pmax; si despacha de
    más, por debajo de su Pmin. Este término le enseña a CERRAR EL BALANCE de
    potencia con los generadores PV. Normalizado por el rango de la slack.

(f) LÍMITES DE REACTIVA. Las consignas de tensión que da el agente determinan
    cuánta reactiva debe producir cada máquina. Si pide una tensión que obliga a
    una máquina a salirse de su curva de capacidad, se penaliza. Normalizado por
    el rango de Q de cada máquina.

(g) DIVERGENCIA. Un despacho tan malo que el Newton-Raphson no converge es la
    peor situación posible (el punto de operación no existe): recompensa fija
    muy negativa.

MÁRGENES DE SEGURIDAD: durante el entrenamiento, (a), (c) y (d) se evalúan con
límites un poco más estrictos que los reales (98 % y [0.955, 1.045] p.u. por
defecto, ver config.MARGEN_*). El agente aprende a dejar colchón y, al
evaluarlo contra los límites reales (100 %, [0.95, 1.05]), los pequeños
errores de la política no se convierten en violaciones.

ESCALAS TÍPICAS con los pesos por defecto (config.py):
    seguro y cercano al OPF (5 % sobre mérito)  ->  r ≈ 3 − 0.5        ≈ +2.5
    seguro pero caro (20 % sobre mérito)        ->  r ≈ 3 − 2.0        ≈ +1.0
    barato pero una línea al 110 %              ->  r ≈ −0.3 − 20·0.12 ≈ −2.7
    no converge                                 ->  r = −20
"""

from __future__ import annotations

import numpy as np

import config as C
from red_electrica import RedElectrica, ResultadoFlujo


def calcular_violaciones(res: ResultadoFlujo, red: RedElectrica, con_margen: bool) -> dict:
    """Mide cuánto viola un resultado de flujo cada restricción de seguridad.

    con_margen=True  -> límites de ENTRENAMIENTO (más estrictos, ver config).
    con_margen=False -> límites REALES (evaluación, interfaz, reporte).

    Devuelve un dict con magnitudes (para la recompensa) y conteos (para
    reportar), y la bandera 'seguro'.
    """
    if not res.convergio:
        return {"seguro": False, "convergio": False}

    m_carg = C.MARGEN_CARGABILIDAD_PCT if con_margen else 0.0
    m_v = C.MARGEN_TENSION_PU if con_margen else 0.0
    lim_carg = C.CARGABILIDAD_MAX_PCT - m_carg
    v_min, v_max = C.TENSION_MIN_PU + m_v, C.TENSION_MAX_PU - m_v
    tol = C.TOL_VIOLACION

    # (c) Ramas
    carg = res.cargabilidad_ramas_pct
    exceso_carg = np.maximum(0.0, carg - lim_carg)
    sobrecarga_pu = float(exceso_carg.sum() / 100.0)

    # (d) Tensiones
    desvio_v = np.maximum(0.0, v_min - res.vm_pu) + np.maximum(0.0, res.vm_pu - v_max)
    desvio_tension_pu = float(desvio_v.sum())

    # (e) Slack (índice 0)
    ps = res.p_gen_mw[0]
    exceso_slack_mw = max(0.0, ps - red.p_max[0]) + max(0.0, red.p_min[0] - ps)
    exceso_slack_pu = exceso_slack_mw / (red.p_max[0] - red.p_min[0])

    # (f) Reactiva de todos los generadores
    q = res.q_gen_mvar
    exceso_q = np.maximum(0.0, q - red.q_max) + np.maximum(0.0, red.q_min - q)
    exceso_reactiva_pu = float((exceso_q / (red.q_max - red.q_min)).sum())

    n_ramas = int((exceso_carg > tol * 100).sum())
    n_barras = int((desvio_v > tol).sum())
    n_q = int((exceso_q > tol * 100).sum())
    slack_ok = bool(exceso_slack_mw <= tol * 100)

    return {
        "convergio": True,
        "seguro": bool(n_ramas == 0 and n_barras == 0 and n_q == 0 and slack_ok),
        "sobrecarga_pu": sobrecarga_pu,
        "desvio_tension_pu": desvio_tension_pu,
        "exceso_slack_pu": float(exceso_slack_pu),
        "exceso_slack_mw": float(exceso_slack_mw),
        "exceso_reactiva_pu": exceso_reactiva_pu,
        "n_ramas_sobrecargadas": n_ramas,
        "n_barras_fuera_banda": n_barras,
        "n_gen_fuera_q": n_q,
        "cargabilidad_max_pct": float(carg.max()) if len(carg) else 0.0,
        "v_min_pu": float(res.vm_pu.min()),
        "v_max_pu": float(res.vm_pu.max()),
    }


def calcular_recompensa(res: ResultadoFlujo, precios, demanda_mw: float,
                        red: RedElectrica) -> tuple[float, dict]:
    """Recompensa del despacho (ver docstring del módulo). Devuelve (r, info)."""
    if not res.convergio:
        return C.R_DIVERGENCIA, {"convergio": False, "seguro": False,
                                 "recompensa": C.R_DIVERGENCIA}

    v = calcular_violaciones(res, red, con_margen=True)

    costo = red.costo_despacho(res.p_gen_mw, precios)
    _, costo_merito = red.despacho_orden_merito(demanda_mw, precios)
    sobrecosto = float(max(0.0, (costo - costo_merito) / costo_merito))

    termino_bono = C.BONO_SEGURO * float(v["seguro"])
    termino_costo = C.W_COSTO * min(sobrecosto, C.SOBRECOSTO_MAX)
    termino_lineas = C.W_LINEAS * v["sobrecarga_pu"]
    termino_tension = C.W_TENSION * v["desvio_tension_pu"]
    termino_slack = C.W_SLACK * v["exceso_slack_pu"]
    termino_reactiva = C.W_REACTIVA * v["exceso_reactiva_pu"]

    r = (termino_bono - termino_costo - termino_lineas - termino_tension
         - termino_slack - termino_reactiva)

    info = dict(v)
    info.update({
        "recompensa": float(r),
        "costo": costo,
        "costo_merito": costo_merito,
        "sobrecosto": sobrecosto,
        "r_bono": termino_bono,
        "r_costo": -termino_costo,
        "r_lineas": -termino_lineas,
        "r_tension": -termino_tension,
        "r_slack": -termino_slack,
        "r_reactiva": -termino_reactiva,
    })
    return float(r), info
