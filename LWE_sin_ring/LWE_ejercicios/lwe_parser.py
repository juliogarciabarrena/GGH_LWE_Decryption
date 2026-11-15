#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lwe_parser.py — Lectura del formato de los ficheros de reto LWE / Ring-LWE.

El formato real del material tiene peculiaridades que este parser tolera:
términos independientes sin llaves (Tier3T01), una fila de coeficientes sin
`{` inicial (Tier3T03), el tipo `criptograma` en Tier0T04, espacios
sobrantes, encoding latin-1 y el formato distinto de los Ring-LWE.

Las líneas de comentario (que empiezan por `#`) actúan de cabeceras de
sección: se clasifica cada línea de datos según la última cabecera vista,
que es mucho más robusto que contar posiciones.
"""
import re
from pathlib import Path

_NUM = re.compile(r"-?\d+")          # patrón: enteros con signo dentro de una línea


class DesafioLWE:
    """Reto LWE 'plano': A·S ≡ b + e (mod q)."""
    def __init__(self, tipo, k, q, A, b, cripto, solucion, nombre=""):
        self.tipo = tipo              # 'rotura' | 'criptoanalisis'
        self.k = k                    # nº de incógnitas (dimensión)
        self.q = q                    # módulo
        self.A = A                    # m x k coeficientes
        self.b = b                    # m términos independientes
        self.cripto = cripto          # (coeficientes, B) o None
        self.solucion = solucion      # lista (rotura) / int (criptoanálisis) / None
        self.nombre = nombre


class DesafioRingLWE:
    """Reto Ring-LWE: b = a·s + e mod (x^N + 1, q); criptograma (c1, c2)."""
    def __init__(self, N, q, a, b, c1, c2, solucion, nombre=""):
        self.N, self.q = N, q         # grado del anillo y módulo
        self.a, self.b = a, b         # polinomios públicos a y b
        self.c1, self.c2 = c1, c2     # criptograma (dos polinomios)
        self.solucion = solucion      # palabra de bits esperada o None
        self.nombre = nombre


def _lineas_datos(path):
    """Lee el fichero y clasifica cada línea en (sección, texto).
    La sección se deduce del último comentario-cabecera visto."""
    with open(path, "r", encoding="latin-1") as f:   # el material va en latin-1
        brutas = f.read().splitlines()

    # anclas ASCII que sobreviven al encoding de los comentarios
    ANCLAS = [
        ("problema a resolver", "tipo"),
        ("criptograma", "cripto"),
        ("dimensi", "dim"),
        ("reducci", "mod"),
        ("coeficientes de las ecuaciones", "A"),
        ("polinomio a", "pa"),
        ("polinomio b", "pb"),
        ("independientes", "b"),
        ("soluci", "sol"),
    ]

    secciones = {}       # clave de sección -> líneas de datos de esa sección
    actual = None        # sección activa (la última cabecera vista)
    for ln in brutas:
        s = ln.strip()
        if not s:
            continue                     # línea en blanco
        if s.startswith("#"):
            low = s.lower()
            for ancla, clave in ANCLAS:
                if ancla in low:         # ¿cabecera conocida?
                    actual = clave       # pasa a ser la sección activa
                    if clave not in secciones:
                        secciones[clave] = []
                    break
            continue                     # un comentario nunca es dato
        if actual is None:
            continue                     # dato antes de cabecera alguna: fuera
        if actual not in secciones:
            secciones[actual] = []
        secciones[actual].append(s)      # línea de datos a su sección
    return secciones


def _numeros(linea):
    """Extrae todos los enteros (con signo) que aparecen en la línea."""
    numeros = []
    for texto in _NUM.findall(linea):
        numeros.append(int(texto))
    return numeros


def _parse_solucion(linea):
    """La sección 'solución': None si no se publica, lista si es un vector,
    un único entero si es un bit."""
    if linea is None:
        return None                      # no hay sección de solución
    s = linea.strip()
    if s.upper() in ("N/A", "N/D", "ND"):
        return None                      # el enunciado no publica solución
    nums = _numeros(s)
    if "{" in s:
        return nums                      # con llaves: vector secreto completo
    if nums:
        return nums[0]                   # sin llaves: un único bit 0/1
    return None


def parse_lwe(path):
    """Parsea un reto LWE o Ring-LWE; devuelve DesafioLWE o DesafioRingLWE."""
    sec = _lineas_datos(path)

    # --- tipo de problema (puede venir 'criptograma' == criptoanálisis) ---
    if "tipo" in sec:
        tipo = sec["tipo"][0].strip().lower()
    else:
        tipo = ""

    # --- solución publicada (solo Tier 0 suele tenerla) ---
    if "sol" in sec:
        sol = _parse_solucion(sec["sol"][0])
    else:
        sol = None
    nombre = Path(path).stem             # p. ej. 'Tier2T03_LWE'

    if "ring" in tipo:
        # ---- formato Ring-LWE ----
        N = _numeros(sec["dim"][0])[0]   # grado del anillo
        q = _numeros(sec["mod"][0])[0]   # módulo
        # el criptograma ring ocupa dos líneas (c1 y c2)
        c1 = _numeros(sec["cripto"][0])
        if len(sec["cripto"]) > 1:
            c2 = _numeros(sec["cripto"][1])
        else:
            c2 = []
        a = _numeros(sec["pa"][0])       # polinomio público a
        b = _numeros(sec["pb"][0])       # término independiente b
        return DesafioRingLWE(N, q, a, b, c1, c2, sol, nombre)

    # normalizamos el tipo: "criptograma" == "criptoanalisis" (Tier0T04)
    if "rotura" in tipo:
        tipo_n = "rotura"
    else:
        tipo_n = "criptoanalisis"

    # --- criptograma: coeficientes del vector y bit objetivo B en una línea ---
    cripto = None
    if tipo_n == "criptoanalisis" and sec.get("cripto"):
        nums = _numeros(sec["cripto"][0])
        if len(nums) >= 2:
            coeficientes = nums[:len(nums) - 1]   # todos menos el último
            bit = nums[len(nums) - 1]             # el último es B
            cripto = (coeficientes, bit)

    k = _numeros(sec["dim"][0])[0]       # nº de incógnitas
    q = _numeros(sec["mod"][0])[0]       # módulo

    A = []                               # una fila de A por línea
    for ln in sec.get("A", []):
        A.append(_numeros(ln))

    if "b" in sec:
        b = _numeros(sec["b"][0])        # términos independientes
    else:
        b = []

    return DesafioLWE(tipo_n, k, q, A, b, cripto, sol, nombre)
