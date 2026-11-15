#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lwe_reticula.py — Construcción de la base de la retícula q-aria de un
sistema LWE.

Dado A (m x k) y el primo q, la retícula

    Λ = { y ∈ Z^m : y ≡ A·x (mod q) para algún x }

tiene dimensión m y determinante q^(m-k). Como b = A·S + e, el punto b está
a distancia exactamente ||e|| de Λ: recuperar el error es un CVP.

`base_reticula_qaria` produce la base en forma "sistemática" (todas las
entradas < q): reordenadas las filas para que el bloque k x k superior sea
invertible,

    B = [  I_k      M^T  ]   (k filas),   M = A_bajo·A_alto^{-1} mod q
        [  0    q·I_{m-k} ]   (m-k filas)

(una eliminación entera ingenua de los generadores haría crecer las
entradas exponencialmente; esta construcción por bloques las mantiene
acotadas). El ataque completo (embedding + LLL + lectura del vector corto)
está escrito paso a paso en los archivos de los ejercicios que lo usan.
"""
from lwe_algebra import inversa_mod


def base_reticula_qaria(A, q):
    """Base m x m de Λ en forma sistemática. Requiere que el bloque k x k
    superior de A sea invertible módulo q (ver bloque_invertible)."""
    m, k = len(A), len(A[0])
    Ainv = inversa_mod(A[:k], q)         # inversa del bloque k x k superior
    if Ainv is None:
        raise ValueError("Bloque k x k singular; reordena las filas antes")

    base = []

    # --- k primeras filas: [ I_k | M^T ], con M = A_bajo · A_alto⁻¹ (mod q) ---
    for i in range(k):
        fila = []
        for j in range(k):               # bloque identidad I_k
            if i == j:
                fila.append(1)
            else:
                fila.append(0)
        for t in range(m - k):           # entrada (t, i) de M: fila t de A_bajo · A⁻¹
            producto = 0
            for j in range(k):
                producto += A[k + t][j] * Ainv[j][i]
            fila.append(producto % q)    # columna i de Mᵀ, reducida mod q
        base.append(fila)

    # --- m-k últimas filas: [ 0 | q·I ] (los múltiplos de q están en Λ) ---
    for t in range(m - k):
        fila = []
        for j in range(k):
            fila.append(0)               # ceros a la izquierda
        for j in range(m - k):
            if j == t:
                fila.append(q)           # q en la diagonal
            else:
                fila.append(0)
        base.append(fila)

    return base
