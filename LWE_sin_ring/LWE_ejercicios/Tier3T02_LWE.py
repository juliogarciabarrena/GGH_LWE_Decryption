#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejercicio Tier3T02_LWE — ROTURA por retículas (k=7, q=1223, m=16)
Fichero de datos: Material_asociado/Criptoan_lisis_LWE/Tier3T02_LWE.txt

Mismo ataque que Tier3T01 (embedding + LLL, dimensión 17). Aquí el ruido
real llega a |e| = 15: ||e|| ≈ 40 sigue muy por debajo del primer mínimo
esperado de la retícula (≈ 180), así que LLL lo encuentra igual de fácil.

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
    Λ      la retícula: todos los puntos y ≡ A·x (mod q) que A puede
           generar; b = A·S − e está a distancia ||e|| de Λ (un CVP)
    Lam    base de Λ en forma sistemática (m x m, entradas < q)
    τ      coordenada extra del embedding: la fila [b | 1] convierte
           el CVP (punto más cercano) en SVP (vector más corto)
    v      una fila de la base tras LLL; la buscada es (−e, ±1): su
           primera parte ES el error, y con él S = A⁻¹(b + e)
  ------------------------------------------------------------------
Ejecución:  python Tier3T02_LWE.py [ruta-al-reto]
"""
import sys
import time
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # consola legacy (cp1252): simbolos sin morir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lwe_parser import parse_lwe
from lwe_algebra import (centrar, resolver_sistema_mod, verificar,
                         bloque_invertible)
from lwe_reticula import base_reticula_qaria
from lwe_lll import reductores

FICHERO = (Path(__file__).resolve().parent.parent
           / "Material_asociado" / "Criptoan_lisis_LWE" / "Tier3T02_LWE.txt")


def main(path=None):
    t0 = time.perf_counter()
    print("=== Tier3T02 · rotura LWE por reducción de retícula ===")

    # ---------- paso 1: leer el reto ----------
    d = parse_lwe(path or FICHERO)
    q, k, A, b = d.q, d.k, d.A, d.b
    m = len(A)
    print("sistema: m=%d ecuaciones, k=%d incógnitas, q=%d" % (m, k, q))

    # ---------- paso 2: base sistemática de la retícula q-aria ----------
    orden = bloque_invertible(A, q)               # permutación con bloque k×k invertible
    Ap = []                                       # A con las filas en ese orden
    for i in orden:
        Ap.append(A[i])
    bp = []                                       # b en el mismo orden
    for i in orden:
        bp.append(b[i])
    Lam = base_reticula_qaria(Ap, q)             # base m x m, entradas < q
    print("base de Λ construida (dim %d, det = q^(m-k) = %d^%d)"
          % (len(Lam), q, m - k))

    # ---------- paso 3: embedding (CVP -> SVP) y LLL ----------
    # filas de Λ con una columna extra 0, y el objetivo b con τ = 1
    filas = []
    for fila in Lam:
        fila_ampliada = fila[:]                   # copia de la fila de la base
        fila_ampliada.append(0)                   # columna extra del error
        filas.append(fila_ampliada)
    objetivo = bp[:]                              # el punto b como fila extra
    objetivo.append(1)                            # coordenada del embedding τ = 1
    filas.append(objetivo)

    # ---------- paso 4: vector corto (−e, ±1) → S ----------
    # reducción en cascada (flotante δ=0,75 → 0,99 → exacta), con
    # verificación exacta del candidato en cada intento
    S = None
    errores = None
    for reductor in reductores():
        reducida = reductor(filas)
        print("  LLL aplicado (dim %d)" % (m + 1))
        for v in reducida:
            if abs(v[m]) != 1:                    # debe llevar el τ = ±1
                continue
            for sg in (1, -1):
                e = []                            # parte del vector corto: el error
                for x in v[:m]:
                    e.append(-sg * x)
                b_mas_e = []                      # segundo miembro b + e (mod q)
                for i in range(k):
                    b_mas_e.append((bp[i] + e[i]) % q)
                S = resolver_sistema_mod(Ap[:k], b_mas_e, q)   # S = A⁻¹(b+e)
                if S is None:
                    continue
                ok, resid = verificar(Ap, bp, q, S, q // 16)
                errores_esperados = []
                for x in e:
                    errores_esperados.append(centrar(x, q))
                if ok and resid == errores_esperados:
                    errores = resid
                    break
            if errores is not None:
                break
        if errores is not None:
            break

    if errores is None:
        print("el LLL no dejó visible el vector (e, ±1)")
        return {"ejercicio": "Tier3T02", "tipo": d.tipo, "resultado": None,
                "ok": False, "metodo": "reticula", "detalle": "sin solución",
                "tiempo": time.perf_counter() - t0}

    errores_orig = [None] * m
    for pos, i in enumerate(orden):
        errores_orig[i] = errores[pos]            # cada error a su posición original
    max_error = 0
    for x in errores_orig:
        if abs(x) > max_error:
            max_error = abs(x)
    print("S recuperado           :", S)
    print("errores e (derivados)  :", errores_orig,
          "(|e| ≤ %d; las %d ecuaciones verifican)" % (max_error, m))

    return {"ejercicio": "Tier3T02", "tipo": d.tipo, "resultado": S,
            "ok": True, "metodo": "reticula(LLL)", "errores": errores_orig,
            "detalle": "errores derivados |e| ≤ %d" % max_error,
            "tiempo": time.perf_counter() - t0}


if __name__ == "__main__":
    main()
