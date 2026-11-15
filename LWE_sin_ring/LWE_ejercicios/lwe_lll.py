#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lwe_lll.py — Reducción de retículas: algoritmo LLL implementado desde cero.

  * lll            versión exacta (aritmética Fraction, alg. incremental de
                   Cohen 2.6.3): referencia de corrección y respaldo
  * lll_flotante   Gram-Schmidt en float64 vectorizado con numpy (filas
                   enteras exactas en int64: solo μ y b* son aproximados);
                   requiere numpy
  * reducir        elige el reductor (flotante si numpy, exacto si no)
  * reductores     orden de reductores que usan los ataques: flotante
                   (rápido) primero y exacto de respaldo; los candidatos se
                   verifican SIEMPRE de forma exacta módulo q en el atacante,
                   de modo que la corrección no depende de la aritmética
                   flotante

La versión exacta está escrita con bucles explícitos, paso a paso. La
version numpy conserva sus operaciones vectorizadas porque son la clave
del rendimiento en N=64 (no son comprehensions: son productos matriz).
"""
import math
from fractions import Fraction

try:
    import numpy as _np
except ImportError:      # numpy es opcional: sin él se usa el LLL exacto
    _np = None


# ======================================================================
# LLL exacto (Fraction)
# ======================================================================
def _redondear_fr(fr):
    """Entero más próximo a una fracción (empate hacia arriba)."""
    return math.floor(fr + Fraction(1, 2))    # floor(x + 1/2): redondeo al entero más próximo


def lll(bases, delta=Fraction(99, 100), candidato=None, cota=None, max_iter=None):
    """Reducción LLL exacta. `bases` es una lista de filas enteras;
    devuelve la base reducida (listas de enteros).

    `candidato(fila) -> bool` (con `cota` como prefiltrado por máxima
    coordenada, igual que en la versión flotante) permite SALIDA
    ANTICIPADA: cada 200 iteraciones se comprueba si alguna fila de la
    base ya satisface el predicado y se devuelve en ese momento."""
    n = len(bases)                            # nº de filas = dimensión de la retícula
    m = len(bases[0])                         # coordenadas de cada vector

    # --- copia de la base en aritmética EXACTA (fracciones) ---
    B = []
    for fila in bases:
        fila_fr = []
        for x in fila:
            fila_fr.append(Fraction(x))
        B.append(fila_fr)

    # --- estado de Gram-Schmidt: b*, μ y normas cuadradas ---
    Bs = []                                   # Bs[i] = b*_i (vectores ortogonales)
    for _ in range(n):
        Bs.append([Fraction(0)] * m)
    mu = []                                   # mu[i][j] = <b_i, b*_j> / ||b*_j||²
    for _ in range(n):
        mu.append([Fraction(0)] * n)
    nrm = []                                  # nrm[i] = ||b*_i||²
    for _ in range(n):
        nrm.append(Fraction(0))

    def gso(i):
        """Recalcula b*_i, su fila de μ y su norma cuadrada (Gram-Schmidt)."""
        Bs[i] = B[i][:]                       # b*_i empieza siendo b_i, sin proyectar
        for j in range(i):                    # se resta su proyección sobre cada b*_j previo
            if nrm[j] != 0:
                producto = Fraction(0)
                for t in range(m):
                    producto += B[i][t] * Bs[j][t]     # <b_i, b*_j>
                mu[i][j] = producto / nrm[j]           # μ_ij = <b_i, b*_j> / ||b*_j||²
                if mu[i][j] != 0:
                    nueva = []
                    for t in range(m):
                        nueva.append(Bs[i][t] - mu[i][j] * Bs[j][t])   # b*_i −= μ_ij·b*_j
                    Bs[i] = nueva
            else:
                mu[i][j] = Fraction(0)         # b*_j nulo ⟹ proyección nula
        norma = Fraction(0)
        for t in range(m):
            norma += Bs[i][t] * Bs[i][t]
        nrm[i] = norma                         # ||b*_i||²

    for i in range(n):                         # Gram-Schmidt inicial de toda la base
        gso(i)

    k = 1                                      # fila en revisión (LLL avanza y retrocede)
    it = 0                                     # contador de iteraciones
    while k < n:                               # hasta que TODA la base esté reducida
        it += 1
        if max_iter is not None and it > max_iter:
            return None                        # aborto barato: nada ha aflorado a tiempo

        # --- SALIDA ANTICIPADA: cada 200 iteraciones, ¿ya hay alguna fila
        # --- que sea el vector buscado?
        if candidato is not None and it % 200 == 0:
            for i in range(n):
                if abs(int(B[i][m - 1])) != 1:
                    continue                   # prefiltro: última coordenada ±1 (la del embedding)
                fila = []
                for x in B[i]:
                    fila.append(int(x))
                cumple_cota = True
                if cota is not None:
                    for x in fila[:-1]:
                        if abs(x) > cota:      # el resto de coordenadas debe ser ≤ cota
                            cumple_cota = False
                            break
                if cumple_cota and candidato(fila):
                    salida = []
                    for f in B:                # encontrada: se devuelve la base tal cual
                        fila_int = []
                        for x in f:
                            fila_int.append(int(x))
                        salida.append(fila_int)
                    return salida

        # --- reducción de tamaño de b_k ---
        for j in range(k - 1, -1, -1):         # de atrás hacia delante (orden del criterio LLL)
            if abs(mu[k][j]) > Fraction(1, 2):
                c = _redondear_fr(mu[k][j])    # entero más próximo a μ_kj
                if c != 0:
                    for t in range(m):
                        B[k][t] = B[k][t] - c * B[j][t]     # b_k ← b_k − c·b_j (misma retícula)
                    for l in range(j):
                        mu[k][l] = mu[k][l] - c * mu[j][l]  # los μ menores también cambian
                    mu[k][j] = mu[k][j] - c    # μ_kj queda en (−1/2, 1/2]

        # --- condición de Lovász ---
        if nrm[k] >= (delta - mu[k][k - 1] * mu[k][k - 1]) * nrm[k - 1]:
            k += 1                             # b*_k suficientemente corta: avanzar
            continue

        # --- intercambio b_k <-> b_{k-1} con actualización incremental ---
        muk = mu[k][k - 1]                     # μ_k,k−1 antes del intercambio
        Bk1 = nrm[k - 1]                       # ||b*_{k−1}||²
        Bk = nrm[k]                            # ||b*_k||²
        Bnueva = Bk + muk * muk * Bk1          # identidad LLL: nueva ||b*_{k−1}||² tras permutar
        B[k], B[k - 1] = B[k - 1], B[k]        # el intercambio físico de las dos filas
        for j in range(k - 1):                 # sus μ hacia índices < k−1 se permutan
            mu[k - 1][j], mu[k][j] = mu[k][j], mu[k - 1][j]
        bsk1_nueva = []
        for t in range(m):
            bsk1_nueva.append(Bs[k][t] + muk * Bs[k - 1][t])   # nuevo b*_{k−1} = b*_k + μ·b*_{k−1}
        mu[k][k - 1] = muk * Bk1 / Bnueva      # nuevo μ_k,k−1
        for t in range(m):
            Bs[k][t] = Bs[k - 1][t] - mu[k][k - 1] * bsk1_nueva[t]   # nuevo b*_k
        Bs[k - 1] = bsk1_nueva
        nrm[k] = Bk1 * Bk / Bnueva             # nueva ||b*_k||²
        nrm[k - 1] = Bnueva                    # nueva ||b*_{k−1}||²
        for i in range(k + 1, n):              # filas posteriores: sus μ hacia k−1 y k cambian
            nu = mu[i][k - 1]
            w = mu[i][k]
            mu[i][k - 1] = (w * Bk + muk * nu * Bk1) / Bnueva
            mu[i][k] = nu - muk * w
        k = max(k - 1, 1)                      # el intercambio pudo desordenar atrás: retrocede uno

    # --- base reducida, convertida a enteros ordinarios ---
    salida = []
    for fila in B:
        fila_int = []
        for x in fila:
            fila_int.append(int(x))
        salida.append(fila_int)
    return salida


# ======================================================================
# LLL flotante vectorizado (numpy) — el rápido
# ======================================================================
def lll_flotante(bases, delta=0.99, refresco=30, candidato=None, cota=None,
                 max_iter=None):
    """LLL con filas enteras EXACTAS (array int64 de numpy: las combinaciones
    usan coeficientes enteros, sin riesgo de desborde en estas tallas) y solo
    μ y b* en float64. El GSO completo se recalcula cada `refresco`
    iteraciones para frenar la deriva numérica. Si se pasa `candidato(fila)`
    (y opcionalmente `cota`, un prefiltrado por máxima coordenada), en cada
    iteración se comprueba con numpy si la base ya contiene el vector
    buscado: SALIDA ANTICIPADA sin esperar a la reducción completa.
    Requiere numpy: sin él, usar `lll` (los selectores `reducir` y
    `reductores` ya hacen ese reparto)."""
    n, m = len(bases), len(bases[0])
    B = _np.array(bases, dtype=_np.int64)      # filas ENTERAS exactas (int64)
    Bs = _np.zeros((n, m), dtype=_np.float64)  # b*_i (aproximados en float64)
    mu = _np.zeros((n, n), dtype=_np.float64)  # coeficientes μ_ij
    nrm = _np.zeros(n, dtype=_np.float64)      # ||b*_i||²

    def gso(i):
        bs = B[i].astype(_np.float64)          # b_i en flotante para proyectar
        if i:
            # Gram-Schmidt modificado en dos operaciones matriciales, con
            # operandos contiguos y SIN transpuestas (dot es simétrico):
            # (i,m)@(m,) para los productos y (i,)@(i,m) para la resta
            mu[i, :i] = (Bs[:i] @ bs) / nrm[:i]    # μ_ij para todo j<i, de golpe
            bs -= mu[i, :i] @ Bs[:i]               # resta todas las proyecciones de golpe
        Bs[i] = bs
        nrm[i] = bs @ bs                           # ||b*_i||²

    def gso_completo():
        for i in range(n):
            gso(i)

    def hay_candidato():
        # prefiltro vectorizado: filas con última coordenada ±1 y, si hay
        # cota, con todas las coordenadas pequeñas; el predicado Python
        # solo se ejecuta para las (pocas) supervivientes
        ult = B[:, -1]                             # última coordenada de todas las filas
        idx = _np.nonzero(ult * ult == 1)[0]       # filas con ±1 ahí (el vector del embedding)
        if len(idx) and cota is not None:
            idx = idx[_np.abs(B[idx, :-1]).max(axis=1) <= cota]   # y coordenadas ≤ cota
        for i in idx:
            if candidato(B[i].tolist()):           # predicado exacto solo en supervivientes
                return True
        return False

    gso_completo()
    k = 1
    it = 0
    while k < n:
        it += 1
        if it % refresco == 0:                     # GSO periódico: frena la deriva del float64
            gso_completo()
        if max_iter is not None and it > max_iter:
            return None                            # aborto barato: nada relevante ha aflorado
        if candidato is not None and hay_candidato():
            return B.tolist()                      # salida anticipada: el corto ya está en la base
        # --- reducción de tamaño de b_k (escaneo escalar descendente: se
        # midió contra un escaneo vectorizado de violaciones con re-rondas
        # y este es 2x más rápido, porque las correcciones solo alteran los
        # μ de índices menores y el escaneo temprano casi no corrige nada) ---
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:                # |μ_kj| > 1/2 ⟹ reducir tamaño
                c = int(math.floor(mu[k, j] + 0.5))    # entero más próximo a μ_kj
                if c:
                    B[k] -= c * B[j]               # b_k ← b_k − c·b_j (exacta: filas int64)
                    mu[k, :j] -= c * mu[j, :j]     # μ menores, actualizados de golpe
                    mu[k, j] -= c                  # μ_kj queda en (−1/2, 1/2]
        # condición de Lovász
        if nrm[k] >= (delta - mu[k, k - 1] * mu[k, k - 1]) * nrm[k - 1]:
            k += 1                                 # avanzar
            continue
        # intercambio b_k <-> b_{k-1} (actualización incremental vectorizada)
        muk = mu[k, k - 1]                         # μ_k,k−1 antes del intercambio
        Bk1, Bk = nrm[k - 1], nrm[k]               # normas cuadradas de b*_{k−1} y b*_k
        Bnueva = Bk + muk * muk * Bk1              # identidad LLL: nueva ||b*_{k−1}||²
        B[[k, k - 1]] = B[[k - 1, k]]              # permuta las dos filas
        mu[k - 1, :k - 1], mu[k, :k - 1] = (mu[k, :k - 1].copy(),
                                            mu[k - 1, :k - 1].copy())   # permuta sus μ (.copy(): vistas)
        bsk1_nueva = Bs[k] + muk * Bs[k - 1]       # nuevo b*_{k−1}
        mu[k, k - 1] = muk * Bk1 / Bnueva          # nuevo μ_k,k−1
        Bs[k] = Bs[k - 1] - mu[k, k - 1] * bsk1_nueva    # nuevo b*_k
        Bs[k - 1] = bsk1_nueva
        nrm[k] = Bk1 * Bk / Bnueva                 # nueva ||b*_k||²
        nrm[k - 1] = Bnueva                        # nueva ||b*_{k−1}||²
        if k + 1 < n:
            nu = mu[k + 1:, k - 1].copy()          # μ de TODAS las filas posteriores
            w = mu[k + 1:, k].copy()
            mu[k + 1:, k - 1] = (w * Bk + muk * nu * Bk1) / Bnueva
            mu[k + 1:, k] = nu - muk * w
        k = max(k - 1, 1)                          # retrocede uno: atrás pudo quedar sin reducir
    return B.tolist()


def reducir(bases):
    """Elige el reductor LLL: flotante (numpy) si está disponible y exacto
    (Fraction) si no."""
    if _np is not None:
        return lll_flotante(bases)
    return lll(bases)


def reductores():
    """Orden de reductores que usan los ataques, de más rápido a más
    exhaustivo (medido sobre el Ring N=64):

        δ=0,60 + refresco 4000 ->  ~0,8 s  (reducción 'relajada': al LLL le
                                            basta para que aflore el vector
                                            corto; refresco muy espaciado,
                                            meseta medida en 0,79 s, porque
                                            el drift lo detecta la
                                            verificación exacta y lo corrigen
                                            los escalones siguientes)
        δ=0,75 + refresco 150  ->  ~2,7 s
        δ=0,99 + refresco  60  ->  ~15 s
        exacto (Fraction)      ->  garantía

    El candidato elegido se verifica SIEMPRE de forma exacta módulo q en el
    código del ataque, así que un reductor rápido que falle solo cuesta
    reintentar con el siguiente."""
    if _np is not None:
        from functools import partial      # partial pre-fija argumentos de una función
        return (partial(lll_flotante, delta=0.60, refresco=4000),   # escalón 1: rápido, relajado
                partial(lll_flotante, delta=0.75, refresco=150),    # escalón 2: intermedio
                partial(lll_flotante, delta=0.99, refresco=60),     # escalón 3: casi exacto
                lll)                                                # respaldo final: exacto
    return (lll,)
