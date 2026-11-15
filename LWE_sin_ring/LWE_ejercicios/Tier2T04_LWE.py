#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejercicio Tier2T04_LWE — CRIPTOANÁLISIS del bit (k=7, q=541, m=25)
Fichero de datos: Material_asociado/Criptoan_lisis_LWE/Tier2T04_LWE.txt

Paso 0 es una DECISIÓN de método, visible en el código: con k=7 la
enumeración cuesta (2η+1)^7 candidatos; los errores reales de este nivel
llegan a η=6, es decir 13^7 ≈ 6.3·10^7 candidatos → se descarta y se ataca
con RETÍCULAS (es la frontera exacta entre Tier 2 y Tier 3):

  1) base sistemática de Λ = {y ∈ Z^m : y ≡ A·x mod q}  (dimensión m)
  2) embedding: b como fila extra con coordenada τ=1 → CVP se vuelve SVP
  3) LLL: el vector corto de la retícula ampliada es (−e, ±1)
  4) con e se recupera S = A_k⁻¹(b_k + e) y se verifica en todas las filas
  5) decisión del bit del criptograma

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
    A_c    coeficientes del criptograma (una ecuación de una sola
           fila; en el código: coef)
    B      término independiente del criptograma:
           B = A_c·S + (q/2)·bit + e (el bit va como "media vuelta")
    R      = A_c·S mod q: el valor del criptograma sin bit ni ruido
    d      distancia circular entre B y R: el camino corto módulo q,
           d = min((B−R) mod q, q − (B−R) mod q)
    bit    0 si d < q/4 (cerca del polo 0), 1 si d >= q/4 (cerca del
           polo q/2)
  ------------------------------------------------------------------
Ejecución:  python Tier2T04_LWE.py [ruta-al-reto]
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
           / "Material_asociado" / "Criptoan_lisis_LWE" / "Tier2T04_LWE.txt")


def main(path=None):
    t0 = time.perf_counter()
    print("=== Tier2T04 · criptoanálisis del bit (ataque de retícula) ===")

    # ---------- paso 1: leer el reto ----------
    d = parse_lwe(path or FICHERO)
    q, k, A, b = d.q, d.k, d.A, d.b
    m = len(A)
    coef, B = d.cripto
    print("sistema: m=%d ecuaciones, k=%d incógnitas, q=%d" % (m, k, q))

    # ---------- paso 2: decidir método por coste ----------
    LIMITE = 10_000_000
    eta_viable = 0
    while (2 * eta_viable + 1) ** k <= LIMITE:    # coste (2η+1)^k de la enumeración
        eta_viable += 1
    print("enumeración viable solo hasta η=%d (%d candidatos); este nivel "
          "necesita η=6 -> 13^7 ≈ %.2g: se descarta" %
          (eta_viable, (2 * eta_viable - 1) ** k, 13.0 ** 7))

    # ---------- paso 3: base sistemática de la retícula q-aria ----------
    # Λ = {y : y ≡ A·x mod q}; b = A·S - e está a distancia ||e|| de Λ
    orden = bloque_invertible(A, q)               # permutación con bloque k×k invertible
    Ap = []                                       # A con las filas en ese orden
    for i in orden:
        Ap.append(A[i])
    bp = []                                       # b en el mismo orden
    for i in orden:
        bp.append(b[i])
    Lam = base_reticula_qaria(Ap, q)             # base m x m, entradas < q
    print("base de Λ construida (dim %d, det q^(m-k) = %d^%d)"
          % (len(Lam), q, m - k))

    # ---------- paso 4: embedding (CVP -> SVP) y LLL ----------
    # filas de Λ con una columna extra 0, y el objetivo b con τ = 1;
    # el vector corto es ±(b - A·S - q·c, 1) = ±(-e, 1)
    filas = []
    for fila in Lam:
        fila_ampliada = fila[:]                   # copia de la fila de la base
        fila_ampliada.append(0)                   # columna extra del error
        filas.append(fila_ampliada)
    objetivo = bp[:]                              # el punto b como fila extra
    objetivo.append(1)                            # coordenada del embedding τ = 1
    filas.append(objetivo)

    # ---------- paso 5: leer el vector corto y recuperar S ----------
    # reducción en cascada (flotante δ=0,75 → 0,99 → exacta): el candidato
    # se verifica de forma exacta antes de aceptarse, así que la cascada
    # solo puede ganar tiempo, nunca corrección
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
        return {"ejercicio": "Tier2T04", "tipo": d.tipo, "resultado": None,
                "ok": False, "metodo": "reticula", "detalle": "sin solución",
                "tiempo": time.perf_counter() - t0}

    errores_orig = [None] * m
    for pos, i in enumerate(orden):
        errores_orig[i] = errores[pos]            # cada error a su posición original
    max_error = 0
    for x in errores_orig:
        if abs(x) > max_error:
            max_error = abs(x)
    print("S recuperado  :", S, " (errores |e| ≤ %d)" % max_error)

    # ---------- paso 6: decidir el bit ----------
    R = 0                                         # R = A_c·S (mod q): valor sin ruido
    for j in range(len(coef)):
        R += coef[j] * S[j]
    R = R % q
    dist = (B - R) % q
    if dist > q - dist:                           # distancia circular: el camino corto
        dist = q - dist
    if 4 * dist < q:                              # cerca del polo 0 ⟹ bit 0
        bit = 0
    else:                                         # cerca del polo q/2 ⟹ bit 1
        bit = 1
    if bit == 0:
        margen = q / 4 - dist
    else:
        margen = dist - q / 4
    print("R = A_c·S mod q = %d ; B = %d -> distancia circular d = %d" % (R, B, dist))
    if bit == 0:
        signo = "<"
    else:
        signo = "≥"
    print("d %s q/4 = %.2f  =>  BIT %d  (margen %.2f)" % (signo, q / 4, bit, margen))

    return {"ejercicio": "Tier2T04", "tipo": d.tipo, "resultado": bit,
            "ok": True, "metodo": "reticula(LLL)", "secreto": S,
            "errores": errores_orig,
            "detalle": "d = %d, umbral q/4 = %.2f, margen %.2f"
                       % (dist, q / 4, margen),
            "tiempo": time.perf_counter() - t0}


if __name__ == "__main__":
    main()
