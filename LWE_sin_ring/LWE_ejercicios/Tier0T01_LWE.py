#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejercicio Tier0T01_LWE — ROTURA por enumeración de errores (k=3, q=229, m=10)
Fichero de datos: Material_asociado/Criptoan_lisis_LWE/Tier0T01_LWE.txt

Idea: el sistema es b = A·S - e (mod q) con errores |e_i| ≤ η pequeños
[A·x ≈ b + e en la notación del enunciado]. Con k=3 incógnitas basta
probar todos los errores del bloque 3x3: para cada candidato e,
S = A_k⁻¹·(b_k + e) y las 7 ecuaciones sobrantes lo validan. Este reto
publica la solución, así que sirve además para calibrar el ruido real
(aquí resulta ser e ≡ 0: sistema sin ruido).

  ------------------------------------------------------------------
  LEYENDA — qué es cada letra:
    A      matriz pública de coeficientes: m filas (ecuaciones) x k
           columnas (incógnitas); su fila i es A_i
    b      vector público de términos independientes, uno por ecuación
    S      el secreto: las k incógnitas que hay que recuperar
    e      el error (ruido): e_i es el error de la ecuación i; es
           pequeño, |e_i| <= η (el enunciado lo escribe A·S ≈ b + e)
    q      el módulo (la n del enunciado): todo se reduce mod q
    k, m   nº de incógnitas y nº de ecuaciones
    η      cota del ruido: el reto NO la publica; el ataque la
           descubre probando 0, 1, 2, ... (bucle sobre eta)
    A_k    bloque k x k superior de A tras reordenar: cuadrado e
           invertible módulo q
    A_k⁻¹  su inversa módulo q (Gauss-Jordan); da nombre al ataque:
           S = A_k⁻¹·(b_k + e) = S0 + A_k⁻¹·e
    S0     = A_k⁻¹·b_k: la solución que saldría con error cero
  ------------------------------------------------------------------
Ejecución:  python Tier0T01_LWE.py [ruta-al-reto]
"""
import sys
import time
from itertools import product
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # consola legacy (cp1252): simbolos sin morir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lwe_parser import parse_lwe
from lwe_algebra import centrar, inversa_mod, bloque_invertible

FICHERO = (Path(__file__).resolve().parent.parent
           / "Material_asociado" / "Criptoan_lisis_LWE" / "Tier0T01_LWE.txt")
ETA_MAX = 5          # cota de error máxima que se va probando


def fila_por_vector(A, i, v):
    """Producto de la fila i de A por el vector v (enteros, sin módulo)."""
    total = 0
    for j in range(len(v)):
        total += A[i][j] * v[j]
    return total


def main(path=None):
    t0 = time.perf_counter()
    print("=== Tier0T01 · rotura LWE por enumeración de errores ===")

    # ---------- paso 1: leer el reto ----------
    d = parse_lwe(path or FICHERO)
    q, k, A, b = d.q, d.k, d.A, d.b
    m = len(A)
    print("sistema: m=%d ecuaciones, k=%d incógnitas, q=%d" % (m, k, q))

    # ---------- paso 2: bloque k×k invertible; A_k⁻¹ se calcula UNA vez ----------
    orden = bloque_invertible(A, q)          # reordena filas solo si hace falta, devuleve indices
    Ap = []                                  # A con las filas en ese orden
    for i in orden:
        Ap.append(A[i])
    bp = []                                  # b en el mismo orden
    for i in orden:
        bp.append(b[i])
    Ainv = inversa_mod(Ap[:k], q)            # obtener inversainversa del bloque k×k superior
    S0 = []                                  # S0 = A_k⁻¹·b_k: la solución "sin error"
    for i in range(k):
        producto = fila_por_vector(Ainv, i, bp[:k])
        S0.append(producto % q)

    # ---------- paso 3: enumerar e ∈ [-η,η]^k ----------
    # b = A·S - e  =>  A·S = b + e  =>  S = A_k⁻¹(b_k + e) = S0 + A_k⁻¹·e
    S = None
    errores = None
    eta_ok = None
    for eta in range(ETA_MAX + 1):
        print("  probando η=%d -> %d candidatos" % (eta, (2 * eta + 1) ** k))
        for e in product(range(-eta, eta + 1), repeat=k):
            # S = S0 + A_k⁻¹·e  (mod q)
            S = []
            for i in range(k):
                producto = fila_por_vector(Ainv, i, e)
                S.append((S0[i] + producto) % q)
            # validación con las m-k ecuaciones restantes: el residuo
            # A_i·S - b_i también debe ser un error pequeño (|·| ≤ η)
            todas_bien = True
            for i in range(k, m):
                residuo = centrar(fila_por_vector(Ap, i, S) - bp[i], q)
                if abs(residuo) > eta:
                    todas_bien = False
                    break
            if todas_bien:
                errores = []                 # residuos de TODAS las ecuaciones
                for i in range(m):
                    residuo = centrar(fila_por_vector(Ap, i, S) - bp[i], q)
                    errores.append(residuo)
                eta_ok = eta
                break
        if errores is not None:
            break

    if errores is None:
        print("SIN SOLUCIÓN hasta η=%d" % ETA_MAX)
        return {"ejercicio": "Tier0T01", "tipo": d.tipo, "resultado": None,
                "ok": False, "metodo": "enumeracion", "detalle": "sin solución",
                "tiempo": time.perf_counter() - t0}

    # ---------- paso 4: errores en el orden original + verificación ----------
    errores_orig = [None] * m
    for pos, i in enumerate(orden):
        errores_orig[i] = errores[pos]       # cada error a su posición original
    if d.solucion is not None:
        ok = (S == d.solucion)
    else:
        ok = True
    print("S recuperado           :", S)
    print("errores e (derivados)  :", errores_orig, "(|e| ≤ %d)" % eta_ok)
    if d.solucion is not None:
        if ok:
            veredicto = "COINCIDEN"
        else:
            veredicto = "¡DISCREPAN!"
        print("solución publicada     :", d.solucion, "->", veredicto)

    return {"ejercicio": "Tier0T01", "tipo": d.tipo, "resultado": S,
            "esperado": d.solucion, "ok": ok,
            "metodo": "enumeracion(η=%d)" % eta_ok, "errores": errores_orig,
            "detalle": "errores derivados |e| ≤ %d" % eta_ok,
            "tiempo": time.perf_counter() - t0}


if __name__ == "__main__":
    main()
