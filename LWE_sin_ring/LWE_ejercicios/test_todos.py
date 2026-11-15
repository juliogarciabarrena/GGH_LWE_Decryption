#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_todos.py — Test completo: ejecuta todos los ejercicios LWE que haya en
esta carpeta, cronometra cada uno y presenta una tabla resumen con el tiempo
total.

Uso:
    python test_todos.py              # ejecuta todos los ejercicios
    python test_todos.py tier0        # solo los que contengan "tier0" (filtro)
    python test_todos.py Tier3        # filtro sensible a mayúsculas

Cada ejercicio vive en su propio archivo <NombreReto>.py (el mismo nombre que
el fichero de datos <NombreReto>.txt) y expone main() -> dict de resultados.
La verificación es exacta cuando el reto publica solución (Tier 0) y de
consistencia interna (todas las ecuaciones satisfechas con error acotado y
márgenes de decisión firmes) en el resto.
"""
import sys
import time
import importlib.util
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # consola legacy (cp1252): simbolos sin morir

AQUI = Path(__file__).resolve().parent


def cargar_modulo(ruta):
    """Carga un archivo de ejercicio como módulo y lo devuelve."""
    nombre = ruta.stem
    espec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(espec)
    espec.loader.exec_module(mod)
    return mod


def main():
    # ---------- filtro opcional por nombre (p. ej. 'tier0') ----------
    filtro = None
    if len(sys.argv) > 1:
        filtro = sys.argv[1]

    ejercicios = sorted(AQUI.glob("Tier*_LWE.py"))
    if filtro:
        seleccion = []
        for f in ejercicios:
            if filtro in f.name:
                seleccion.append(f)
        ejercicios = seleccion
    if not ejercicios:
        print("No hay ejercicios que coincidan con el filtro %r" % filtro)
        return 1

    aviso = " (filtro: %s)" % filtro
    if filtro is None:
        aviso = ""
    print("Ejecutando %d ejercicios%s\n" % (len(ejercicios), aviso))

    # ---------- ejecutar uno a uno, cronometrando ----------
    filas = []
    fallos = 0
    t_total0 = time.perf_counter()
    for ruta in ejercicios:
        mod = cargar_modulo(ruta)
        t0 = time.perf_counter()
        try:
            r = mod.main()
        except Exception as exc:  # el runner no debe morir por un ejercicio
            r = {"ejercicio": ruta.stem, "tipo": "?", "resultado": None,
                 "ok": False, "metodo": "-", "detalle": "EXCEPCIÓN: %s" % exc}
        dt = time.perf_counter() - t0
 
        resuelto = r.get("resultado") is not None
        verif = r.get("ok")
        if not resuelto or verif is False:
            fallos += 1

        if verif in (True,):
            esperado = "publicada"
        else:
            esperado = "N/D"
        if verif:
            verif_txt = "OK"                # coincide con lo publicado
        elif verif is None and resuelto:
            verif_txt = "interna"           # verificación de consistencia
        else:
            verif_txt = "FALLO"

        filas.append({
            "nombre": r.get("ejercicio", ruta.stem),
            "tipo": r.get("tipo", "?"),
            "metodo": r.get("metodo", "-"),
            "resultado": r.get("resultado"),
            "esperado": esperado,
            "verif": verif_txt,
            "tiempo": dt,
            "detalle": r.get("detalle", ""),
        })

        # progreso en vivo
        ultima = filas[-1]
        if ultima["verif"] in ("OK", "interna"):
            marca = "OK"
        else:
            marca = "FALLO"
        print("  [" + str(len(filas)) + "/" + str(len(ejercicios)) + "] "
              + ultima["nombre"].ljust(14) + " "
              + ultima["metodo"].ljust(28) + " " + marca
              + "  (%.2f s)" % dt)

    t_total = time.perf_counter() - t_total0

    # ---------------- tabla resumen ----------------
    ancho = 74
    print("\n" + "=" * ancho)
    print("Ejercicio".ljust(14) + "Tipo".ljust(13) + "Método".ljust(22)
          + "Resultado".ljust(10) + "Tiempo (s)".rjust(9))
    print("-" * ancho)
    for f in filas:
        if isinstance(f["resultado"], list):
            res = "S recuperado"
        elif isinstance(f["resultado"], str):
            res = "bits: %s..." % f["resultado"][:8]
        elif f["resultado"] is None:
            res = "—"
        else:
            res = "bit %s" % f["resultado"]
        print(f["nombre"].ljust(14) + f["tipo"][:13].ljust(13)
              + f["metodo"][:22].ljust(22) + res.ljust(10)
              + ("%.3f" % f["tiempo"]).rjust(9))
    print("-" * ancho)
    print("Tiempo total: %.3f s   |   Resueltos: %d/%d   |   Verificación "
          "fallida: %d" % (t_total, len(filas) - fallos, len(filas), fallos))
    print("  (OK = coincide con la solución publicada; interna = consistencia "
          "del sistema + márgenes)")
    print("=" * ancho + "\n")
    if fallos:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
