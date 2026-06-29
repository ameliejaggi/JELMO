#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR-Analyse handschriftlicher Tabellen via GPUstack der Universität Bern.

Dieses Skript verarbeitet einen Ordner mit JPG/JPEG-Dateien. Pro Bild wird
eine OCR-Anfrage an das Modell qwen3-vl-8b-instruct geschickt (Temperatur 0).
Die Transkriptionen aller Bilder werden in EINE gemeinsame CSV-Datei
geschrieben. Die erste Spalte enthält den Dateinamen des Quell-JPGs.

Spaltenstabilität:
    Dem Modell wird die erwartete Spaltenstruktur (Reihenfolge und Namen
    der 12 Spalten) explizit vorgegeben. Das verhindert, dass das Modell
    eine eigene Aufteilung erfindet (z. B. ein Datum in drei Spalten zu
    zerlegen). Die Antwort kommt als strukturiertes JSON zurück, mit
    festen Schlüsseln pro Spalte.

Es findet KEINE Normierung der Spaltenzahl im Skript statt. Wenn das
Modell trotz Vorgabe einen Schlüssel weglässt, wird die CSV-Zeile
entsprechend kürzer.

Voraussetzungen:
    pip install requests

Verwendung:
    export GPUSTACK_API_KEY="dein_api_key"
    python ocr_tabellen.py /pfad/zum/ordner_mit_jpgs /pfad/zur/output.csv
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# KONFIGURATION
# ---------------------------------------------------------------------------

# GPUstack-Endpoint (OpenAI-kompatibel).
API_BASE_URL = "https://gpustack.unibe.ch/v1"
API_ENDPOINT = f"{API_BASE_URL}/chat/completions"

# Modell
MODEL_NAME = "qwen3-vl-8b-instruct"

# Generierungsparameter
TEMPERATURE = 0

# CSV-Trennzeichen
CSV_DELIMITER = ","

# Wie oft die API erneut angefragt wird, wenn das JSON-Parsing fehlschlägt
MAX_RETRIES = 2

# ----- Erwartete Spaltenstruktur ------------------------------------------
# Diese Liste gibt dem Modell die verbindliche Reihenfolge UND die Namen
# der Spalten vor. Sie wird auch als JSON-Schlüsselbasis verwendet.
COLUMN_NAMES = [
    "No.",
    "Familienname",
    "Vorname",
    "Heimatsort",
    "Land",
    "Geburtsjahr",
    "Beruf",
    "Bisheriger Wohnort",
    "Eingetreten",
    "Wohnung Strasse No.",
    "Ausgetreten",
    "Bemerkungen des Kontrollbeamten (Art. 12)",
]

# JSON-Schlüssel: kurz, eindeutig, ohne Sonderzeichen.
COLUMN_KEYS = [f"c{i:02d}" for i in range(1, len(COLUMN_NAMES) + 1)]

# Kompakte Beschreibung "c01: No., c02: Familienname, ..." für den Prompt
COLUMN_MAPPING_TEXT = ", ".join(
    f"{key}: {name}"
    for key, name in zip(COLUMN_KEYS, COLUMN_NAMES)
)

# Forschungsprompt
USER_PROMPT = (
    "Führe am angehängten JPEG eine OCR-Analyse durch. Es handelt sich um "
    "eine Tabelle mit zwölf Spalten. Transkribiere die gesamte Tabelle "
    "inklusive der Header-Zeile. Behalte die Struktur dieser Tabelle bitte "
    "exakt bei. Achte insbesondere darauf, dass die Ausgaben der richtigen "
    "Zelle zugeordnet werden."
)

# Technische Formatierungsanweisung mit explizit vorgegebener Spaltenstruktur.
FORMAT_INSTRUCTION = (
    "Die Tabelle hat IMMER genau diese 12 Spalten in dieser Reihenfolge "
    f"(von links nach rechts): {COLUMN_MAPPING_TEXT}. "
    "Auch wenn ein Datum visuell wie mehrere Teile aussieht (z. B. Jahr, "
    "Monat, Tag in scheinbar getrennten Sub-Spalten), gehört es trotzdem "
    "in EINE Spalte ('Eingetreten' oder 'Ausgetreten'). Erfinde keine "
    "zusätzlichen Spalten und lasse keine Spalte weg.\n\n"
    "Gib das Ergebnis ausschliesslich als gültiges JSON zurück, ohne "
    "Markdown-Codeblöcke, ohne Erklärungen, ohne zusätzlichen Text. "
    "Das JSON-Objekt hat genau einen Schlüssel \"rows\", dessen Wert "
    "eine Liste ist. Jeder Eintrag der Liste repräsentiert eine "
    "Tabellenzeile (inkl. Header-Zeile als erstem Eintrag) und ist "
    "selbst ein Objekt mit GENAU diesen Schlüsseln: "
    f"{', '.join(COLUMN_KEYS)}. "
    "Der Wert jedes Schlüssels ist der transkribierte Text der "
    "entsprechenden Zelle als Zeichenkette. Leere Zellen werden als "
    "leere Zeichenkette \"\" ausgegeben. Lasse keinen Schlüssel weg, "
    "auch wenn die Zelle leer ist. Verwende keine zusätzlichen "
    "Schlüssel."
)


# ---------------------------------------------------------------------------
# HILFSFUNKTIONEN
# ---------------------------------------------------------------------------

def encode_image_to_base64(image_path: Path) -> str:
    """Liest eine Bilddatei und gibt sie als base64-String zurück."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_ocr_api(image_path: Path, api_key: str) -> str:
    """Schickt das Bild an die GPUstack-API und gibt den Antworttext zurück."""
    image_b64 = encode_image_to_base64(image_path)

    payload = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "text", "text": FORMAT_INSTRUCTION},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        API_ENDPOINT,
        headers=headers,
        data=json.dumps(payload),
        timeout=300,
    )

    # Falls das Backend response_format nicht akzeptiert, erneut ohne senden.
    if response.status_code == 400 and "response_format" in response.text:
        payload.pop("response_format", None)
        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            data=json.dumps(payload),
            timeout=300,
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"API-Fehler (Status {response.status_code}): {response.text}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unerwartetes API-Antwortformat: {json.dumps(data)[:500]}"
        ) from exc


def extract_json_object(text: str) -> str:
    """Extrahiert das erste JSON-Objekt aus einem (eventuell verunreinigten)
    Antworttext.
    """
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(0)

    return cleaned


def parse_model_output(raw_text: str) -> list[list[str]]:
    """Wandelt die JSON-Modellausgabe in eine Liste von Zeilen um.

    Es findet KEINE Normalisierung statt: Pro Zeile werden die Werte
    in der Reihenfolge der vorgegebenen COLUMN_KEYS ausgelesen, aber
    nur diejenigen, die das Modell tatsächlich geliefert hat.
    """
    cleaned = extract_json_object(raw_text)
    parsed = json.loads(cleaned)

    if not isinstance(parsed, dict) or "rows" not in parsed:
        raise ValueError(
            "JSON-Antwort enthält keinen 'rows'-Schlüssel auf oberster Ebene."
        )

    rows_raw = parsed["rows"]
    if not isinstance(rows_raw, list):
        raise ValueError("'rows' ist keine Liste.")

    rows: list[list[str]] = []
    for row_obj in rows_raw:
        if not isinstance(row_obj, dict):
            if isinstance(row_obj, list):
                rows.append([str(v).strip() for v in row_obj])
            continue

        cells: list[str] = []
        for key in COLUMN_KEYS:
            if key in row_obj:
                value = row_obj[key]
                cells.append("" if value is None else str(value).strip())
            else:
                # Schlüssel fehlt im JSON – Zeile wird entsprechend kürzer.
                # KEIN Auffüllen, wie vom Auftraggeber gewünscht.
                break

        # Falls das Modell andere Schlüssel verwendet hat (z. B. "c1"
        # statt "c01") oder gar keine c-Schlüssel, fallweise alle Werte
        # des Objekts in Reihenfolge übernehmen.
        if not cells and row_obj:
            cells = [
                "" if v is None else str(v).strip() for v in row_obj.values()
            ]

        rows.append(cells)

    return rows


def transcribe_image(image_path: Path, api_key: str) -> list[list[str]]:
    """OCR + JSON-Parsing für ein einzelnes Bild, mit Retry-Logik."""
    last_error: Exception | None = None

    for versuch in range(1, MAX_RETRIES + 2):
        zusatz = "" if versuch == 1 else f" (Wiederholung {versuch - 1})"
        print(
            f"  → Sende {image_path.name} an die API …{zusatz}",
            flush=True,
        )
        try:
            raw_output = call_ocr_api(image_path, api_key)
            return parse_model_output(raw_output)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            print(
                f"  ! JSON-Parsing fehlgeschlagen: {exc}",
                flush=True,
            )
            time.sleep(1)

    raise RuntimeError(
        f"Konnte JSON-Antwort nach {MAX_RETRIES + 1} Versuchen nicht "
        f"parsen. Letzter Fehler: {last_error}"
    )


# ---------------------------------------------------------------------------
# HAUPTPROGRAMM
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "OCR-Analyse handschriftlicher Tabellen via GPUstack Uni Bern. "
            "Alle JPGs des Eingabeordners werden in eine gemeinsame CSV "
            "geschrieben."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Ordner mit den JPG/JPEG-Dateien.",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Pfad zur Ausgabe-CSV (z. B. ergebnisse.csv).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GPUSTACK_API_KEY")
    if not api_key:
        print(
            "Fehler: Umgebungsvariable GPUSTACK_API_KEY ist nicht gesetzt.\n"
            "Setze sie z. B. mit:\n"
            "    export GPUSTACK_API_KEY=\"dein_api_key\"",
            file=sys.stderr,
        )
        return 1

    if not args.input_dir.is_dir():
        print(
            f"Fehler: '{args.input_dir}' ist kein gültiger Ordner.",
            file=sys.stderr,
        )
        return 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    jpg_files = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    )

    if not jpg_files:
        print(
            f"Keine JPG/JPEG-Dateien in '{args.input_dir}' gefunden.",
            file=sys.stderr,
        )
        return 1

    print(f"Gefunden:    {len(jpg_files)} Bilddatei(en).")
    print(f"Modell:      {MODEL_NAME}")
    print(f"Endpoint:    {API_ENDPOINT}")
    print(f"Ausgabe-CSV: {args.output_csv}\n")

    erfolge = 0
    fehler: list[tuple[Path, str]] = []
    gesamt_zeilen = 0

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(
            f,
            delimiter=CSV_DELIMITER,
            quoting=csv.QUOTE_MINIMAL,
        )

        for idx, jpg in enumerate(jpg_files, start=1):
            print(f"[{idx}/{len(jpg_files)}] {jpg.name}")
            try:
                rows = transcribe_image(jpg, api_key)
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ Fehler: {exc}", flush=True)
                fehler.append((jpg, str(exc)))
                continue

            for row in rows:
                writer.writerow([jpg.name, *row])

            gesamt_zeilen += len(rows)
            erfolge += 1
            print(
                f"  ✓ {len(rows)} Zeile(n) angehängt "
                f"(gesamt: {gesamt_zeilen}).",
                flush=True,
            )
            f.flush()

    print("\n" + "=" * 60)
    print(f"Fertig. Erfolgreich: {erfolge}/{len(jpg_files)}")
    print(f"Insgesamt geschriebene Datenzeilen: {gesamt_zeilen}")
    print(f"Datei: {args.output_csv}")
    if fehler:
        print(f"\nFehlgeschlagen: {len(fehler)}")
        for path, msg in fehler:
            print(f"  - {path.name}: {msg}")
    print("=" * 60)

    return 0 if not fehler else 2


if __name__ == "__main__":
    sys.exit(main())
