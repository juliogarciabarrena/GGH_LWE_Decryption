#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lwe_algebra.py — Aritmética y álgebra lineal modular (módulo q primo).

Primitivas usadas por los ejercicios:
  * centrar               representante centrado en (-q/2, q/2]
  * inverso_mod           inverso de un escalar módulo q (q primo)
  * resolver_sistema_mod  eliminación gaussiana A·x ≡ b (mod q)
  * inversa_mod           Gauss-Jordan, inversa de matriz k×k módulo q
  * verificar             residuos A·s - b centrados (los errores de LWE)
  * bloque_invertible     reordenación de filas con bloque k×k invertible

Estilo: bucles explícitos y variables con nombre, sin comprehensions
anidadas, para que cada paso se pueda explicar por separado.
"""
import random


def centrar(x, q):
    """Reduce x al representante centrado en (-q/2, q/2]."""
    desplazado = (x + q // 2) % q    # lleva x al rango [0, q)
    return desplazado - q // 2       # baja medio módulo: queda en (-q/2, q/2]


def inverso_mod(x, q):
    """Inverso multiplicativo módulo q (q primo: pequeño teorema de Fermat)."""
    return pow(x % q, q - 2, q)      # x^(q-1) ≡ 1  ⟹  x^(q-2) ≡ x⁻¹ (mod q)


def resolver_sistema_mod(A, b, q):
    """Resuelve A·x ≡ b (mod q) por eliminación gaussiana (A cuadrada k x k).
    Devuelve x o None si A es singular módulo q."""
    k = len(A)

    # --- matriz ampliada [A | b]: copia de A con el término independiente ---
    M = []
    for i in range(k):
        fila = A[i][:]               # copia de la fila i (A no se toca)
        fila.append(b[i])            # última columna: término independiente
        M.append(fila)

    fila_pivote = 0                  # primera fila que aún no tiene pivote
    for col in range(k):             # un paso por columna (una incógnita)
        # --- buscar pivote: fila con valor no nulo en esta columna ---
        piv = None
        for i in range(fila_pivote, k):
            if M[i][col] % q != 0:
                piv = i
                break
        if piv is None:
            return None              # columna toda nula ⟹ A singular mod q

        # --- normalizar la fila del pivote: que el pivote valga 1 ---
        M[fila_pivote], M[piv] = M[piv], M[fila_pivote]   # pivote, arriba
        inv = inverso_mod(M[fila_pivote][col], q)         # inverso del pivote
        for t in range(k + 1):                            # fila ÷ pivote
            M[fila_pivote][t] = (M[fila_pivote][t] * inv) % q

        # --- hacer 0 esta columna en TODAS las demás filas ---
        for i in range(k):
            if i != fila_pivote and M[i][col] != 0:
                factor = M[i][col]        # cuánto vale la columna en la fila i
                for t in range(k + 1):    # fila_i −= factor · fila_pivote
                    M[i][t] = (M[i][t] - factor * M[fila_pivote][t]) % q

        fila_pivote += 1

    # --- la izquierda quedó identidad: la solución está en la última col. ---
    x = []
    for i in range(k):
        x.append(M[i][k])
    return x


def inversa_mod(A, q):
    """Inversa de una matriz k x k módulo q (Gauss-Jordan, o None si singular)."""
    k = len(A)  # Sacamos las columnas

    # --- matriz ampliada [A | I]: al reducir la izquierda a identidad, la
    # --- derecha se convierte, fila a fila, en A⁻¹
    M = []  
    for i in range(k):
        fila = A[i][:]               # copia de la fila i de A
        for j in range(k):           # a su derecha, la identidad
            if i == j:
                fila.append(1)       # diagonal de la identidad
            else:
                fila.append(0)
        M.append(fila)
    


    fila_pivote = 0
    for col in range(k):
        # --- pivote no nulo en esta columna ---
        piv = None
        for i in range(fila_pivote, k):
            if M[i][col] % q != 0:
                piv = i
                break
        if piv is None:
            return None              # sin pivote ⟹ singular

        # --- normalizar: pivote = 1 ---
        M[fila_pivote], M[piv] = M[piv], M[fila_pivote]
        inv = inverso_mod(M[fila_pivote][col], q)
        for t in range(2 * k):       # las 2k columnas: [izquierda | derecha]
            M[fila_pivote][t] = (M[fila_pivote][t] * inv) % q

        # --- hacer 0 esta columna en las demás filas ---
        for i in range(k):
            if i != fila_pivote and M[i][col] != 0:
                factor = M[i][col]
                for t in range(2 * k):
                    M[i][t] = (M[i][t] - factor * M[fila_pivote][t]) % q

        fila_pivote += 1

    # --- izquierda = identidad ⟹ derecha = A⁻¹ ---
    Ainv = []
    for i in range(k):
        Ainv.append(M[i][k:])        # mitad derecha de la fila i
    return Ainv


def verificar(A, b, q, s, cota):
    """Comprueba si s explica el sistema con errores |e_i| <= cota.
    Devuelve (ok, errores_derivados) con errores = centrar(A·s - b)."""
    errores = []
    for i in range(len(A)):
        producto = 0
        for j in range(len(s)):
            producto += A[i][j] * s[j]      # fila i de A por el vector s
        residuo = (producto - b[i]) % q     # A·s − b (mod q)
        errores.append(centrar(residuo, q)) # centrado: ese residuo ES el error e_i

    maximo = 0
    for e in errores:
        if abs(e) > maximo:
            maximo = abs(e)
    return (maximo <= cota), errores        # ok si TODOS los errores cumplen la cota


def bloque_invertible(A, q, intentos=8):
    """Devuelve una permutación de las filas cuyo bloque k x k superior es
    invertible módulo q (las instancias son aleatorias: casi siempre sale a
    la primera, sin reordenar nada)."""
    m, k = len(A), len(A[0])
    orden = list(range(m))                  # de momento, el orden natural 0,1,...
    rng = random.Random(12345)              # semilla fija: igual en cada ejecución
    for _ in range(intentos):
        bloque = []
        for i in orden[:k]:                 # coge las k primeras filas de A del orden actual para crear la primera matriz cuadrada
            bloque.append(A[i])
        if inversa_mod(bloque, q) is not None:  # Si tiene inversso la devuelve sino re-ordena la lista 
            return orden                    # este orden ya tiene bloque invertible
        rng.shuffle(orden)                  # no salía: barajar y reintentar
    raise ValueError("No se encuentra bloque k x k invertible módulo %d" % q)
