#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR-Analyse handschriftlicher Tabellen via GPUstack der Universität Bern.

Dieses Skript verarbeitet einen Ordner mit JPG/JPEG-Dateien. Pro Bild wird
eine OCR-Anfrage an das Modell qwen3-vl-8b-instruct geschickt (Temperatur 0).
Pro JPG entsteht eine gleichnamige CSV-Datei.

Zellzuordnung:
    Das Modell wird angewiesen, die Tabelle als strukturiertes JSON
    zurückzugeben (eine Liste von Objekten, jede Zelle mit eigenem
    Schlüssel c1..c12). Das macht die Zellzuordnung deutlich robuster
    als ein Tabulator-Trennzeichen, weil das Modell für jede Zelle
    explizit antwortet – auch leere Zellen werden eindeutig markiert.

Es findet KEINE Normierung der Spaltenzahl statt: Wenn das Modell
für eine Zeile weniger Schlüssel liefert, wird die CSV-Zeile auch
entsprechend kürzer.

Voraussetzungen:
    pip install requests

Verwendung:
    export GPUSTACK_API_KEY="dein_api_key"
    python ocr_tabellen.py /pfad/zum/ordner_mit_jpgs /pfad/zum/output_ordner
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

# Anzahl der Spalten der Tabelle (nur informativ für den Prompt –
# die Ausgabe wird NICHT auf diese Zahl normiert).
NUM_COLUMNS = 12

# Wie oft die API erneut angefragt wird, wenn das JSON-Parsing fehlschlägt
MAX_RETRIES = 2

# Forschungsprompt – wörtlich wie vom Auftraggeber gewünscht.
USER_PROMPT = (
    "Führe am angehängten JPEG eine OCR-Analyse durch. Es handelt sich um "
    "eine Tabelle mit zwölf Spalten. Transkribiere die gesamte Tabelle "
    "inklusive der Header-Zeile. Behalte die Struktur dieser Tabelle bitte "
    "exakt bei. Achte insbesondere darauf, dass die Ausgaben der richtigen "
    "Zelle zugeordnet werden."
)

# Technische Formatierungsanweisung für strukturierte JSON-Ausgabe.
# Ist getrennt vom Forschungsprompt und betrifft nur das Ausgabeformat.
FORMAT_INSTRUCTION = (
    "Gib das Ergebnis ausschliesslich als gültiges JSON zurück, ohne "
    "Markdown-Codeblöcke, ohne Erklärungen, ohne zusätzlichen Text. "
    "Das JSON-Objekt hat genau einen Schlüssel \"rows\", dessen Wert "
    "eine Liste ist. Jeder Eintrag der Liste repräsentiert eine "
    "Tabellenzeile (inkl. Header-Zeile als erstem Eintrag) und ist "
    "selbst ein Objekt mit den Schlüsseln \"c1\", \"c2\", ..., \"c12\" "
    "– jeweils ein Schlüssel pro Tabellenspalte (von links nach rechts). "
    "Der Wert jedes Schlüssels ist der transkribierte Text dieser Zelle "
    "als Zeichenkette. Leere Zellen werden als leere Zeichenkette \"\" "
    "ausgegeben. Lasse keinen Schlüssel weg, auch wenn die Zelle leer "
    "ist. Beispielstruktur:\n"
    "{\"rows\": ["
    "{\"c1\": \"...\", \"c2\": \"...\", ..., \"c12\": \"...\"}, "
    "{\"c1\": \"...\", \"c2\": \"...\", ..., \"c12\": \"...\"}"
    "]}"
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
        # Bittet OpenAI-kompatible Backends, gültiges JSON zu erzwingen.
        # Wird ignoriert, falls das Backend dies nicht unterstützt.
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
    Antworttext. Entfernt Code-Fences und schneidet auf den ersten {…}-Block.
    """
    cleaned = text.strip()

    # Markdown-Codefences entfernen
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Falls noch Text vor/nach dem JSON-Objekt steht: ersten {…}-Block greifen
    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(0)

    return cleaned


def parse_model_output(raw_text: str) -> list[list[str]]:
    """Wandelt die JSON-Modellausgabe in eine Liste von Zeilen um.

    Es findet KEINE Normalisierung der Spaltenzahl statt. Pro Zeile werden
    die Werte in der Reihenfolge c1, c2, c3, ... ausgelesen, solange die
    Schlüssel lückenlos vorhanden sind. Liefert das Modell nur c1..c10,
    bekommt die Zeile zehn Spalten.
    """
    cleaned = extract_json_object(raw_text)
    parsed = json.loads(cleaned)  # wirft ValueError bei kaputtem JSON

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
            # Falls das Modell unerwarteterweise eine Liste statt eines
            # Objekts liefert, akzeptieren wir auch das.
            if isinstance(row_obj, list):
                rows.append([str(v).strip() for v in row_obj])
            continue

        # Werte in der Reihenfolge c1, c2, ... auslesen, solange lückenlos
        cells: list[str] = []
        i = 1
        while True:
            key = f"c{i}"
            if key in row_obj:
                value = row_obj[key]
                cells.append("" if value is None else str(value).strip())
                i += 1
            else:
                break

        # Wenn die c-Schlüssel ganz fehlen, fallweise alle Werte des Objekts
        # in der vorhandenen Reihenfolge übernehmen.
        if not cells and row_obj:
            cells = [
                "" if v is None else str(v).strip() for v in row_obj.values()
            ]

        rows.append(cells)

    return rows


def write_csv(rows: list[list[str]], output_path: Path) -> None:
    """Schreibt die Datenzeilen in eine CSV-Datei – ohne extra Header."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(
            f,
            delimiter=CSV_DELIMITER,
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writerows(rows)


def process_single_image(
    image_path: Path, output_dir: Path, api_key: str
) -> Path:
    """Verarbeitet ein einzelnes Bild und schreibt die zugehörige CSV.

    Bei JSON-Parsing-Fehlern wird die API bis zu MAX_RETRIES-mal erneut
    angefragt – das gleiche Bild wird neu eingereicht.
    """
    last_error: Exception | None = None

    for versuch in range(1, MAX_RETRIES + 2):  # 1 regulärer + MAX_RETRIES
        zusatz = "" if versuch == 1 else f" (Wiederholung {versuch - 1})"
        print(
            f"  → Sende {image_path.name} an die API …{zusatz}",
            flush=True,
        )
        try:
            raw_output = call_ocr_api(image_path, api_key)
            rows = parse_model_output(raw_output)
            output_path = output_dir / f"{image_path.stem}.csv"
            write_csv(rows, output_path)
            print(
                f"  ✓ {len(rows)} Zeile(n) gespeichert in "
                f"{output_path.name}",
                flush=True,
            )
            return output_path
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            print(
                f"  ! JSON-Parsing fehlgeschlagen: {exc}",
                flush=True,
            )
            time.sleep(1)  # kurze Pause vor dem nächsten Versuch

    # Alle Versuche fehlgeschlagen
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
            "Pro JPG im Eingabeordner wird eine CSV im Ausgabeordner erzeugt."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Ordner mit den JPG/JPEG-Dateien.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Ordner, in den die CSV-Dateien geschrieben werden.",
    )
    args = parser.parse_args()

    # API-Key aus Umgebungsvariable
    api_key = os.environ.get("GPUSTACK_API_KEY")
    if not api_key:
        print(
            "Fehler: Umgebungsvariable GPUSTACK_API_KEY ist nicht gesetzt.\n"
            "Setze sie z. B. mit:\n"
            "    export GPUSTACK_API_KEY=\"dein_api_key\"",
            file=sys.stderr,
        )
        return 1

    # Eingabeordner prüfen
    if not args.input_dir.is_dir():
        print(
            f"Fehler: '{args.input_dir}' ist kein gültiger Ordner.",
            file=sys.stderr,
        )
        return 1

    # Ausgabeordner anlegen
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # JPGs einsammeln (case-insensitive)
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

    print(f"Gefunden: {len(jpg_files)} Bilddatei(en).")
    print(f"Modell:    {MODEL_NAME}")
    print(f"Endpoint:  {API_ENDPOINT}")
    print(f"Ausgabe:   {args.output_dir}\n")

    erfolge = 0
    fehler: list[tuple[Path, str]] = []

    for idx, jpg in enumerate(jpg_files, start=1):
        print(f"[{idx}/{len(jpg_files)}] {jpg.name}")
        try:
            process_single_image(jpg, args.output_dir, api_key)
            erfolge += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ Fehler: {exc}", flush=True)
            fehler.append((jpg, str(exc)))

    # Zusammenfassung
    print("\n" + "=" * 60)
    print(f"Fertig. Erfolgreich: {erfolge}/{len(jpg_files)}")
    if fehler:
        print(f"Fehlgeschlagen: {len(fehler)}")
        for path, msg in fehler:
            print(f"  - {path.name}: {msg}")
    print("=" * 60)

    return 0 if not fehler else 2


if __name__ == "__main__":
    sys.exit(main())
