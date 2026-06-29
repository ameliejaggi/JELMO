#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR-Analyse handschriftlicher Tabellen via GPUstack der Universität Bern.

Dieses Skript verarbeitet einen Ordner mit JPG/JPEG-Dateien. Pro Bild wird
eine OCR-Anfrage an das Modell qwen3-vl-8b-instruct geschickt (Temperatur 0).
Pro JPG entsteht eine gleichnamige CSV-Datei mit 12 Spalten und konstanter
Kopfzeile.

Voraussetzungen:
    pip install requests

Verwendung:
    export GPUSTACK_API_KEY="dein_api_key"
    python ocr_tabellen.py /pfad/zum/ordner_mit_jpgs /pfad/zum/output_ordner

Autor: Skript für ein digitales Geschichtsprojekt
"""

import argparse
import base64
import csv
import json
import os
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# KONFIGURATION
# ---------------------------------------------------------------------------

# GPUstack-Endpoint (OpenAI-kompatibel). Falls die Uni Bern den Pfad anders
# konfiguriert hat (z. B. ohne /v1), hier anpassen.
API_BASE_URL = "https://gpustack.unibe.ch/v1"
API_ENDPOINT = f"{API_BASE_URL}/chat/completions"

# Modell
MODEL_NAME = "qwen3-vl-8b-instruct"

# Generierungsparameter
TEMPERATURE = 0

# Konstante Header der Tabelle (12 Spalten)
HEADERS = [
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

# CSV-Trennzeichen
CSV_DELIMITER = ","

# Prompt für das Modell. Wir bitten explizit um eine TSV-artige Ausgabe mit
# Tabulator als Spaltentrenner, weil die historischen Texte selbst Kommas
# enthalten können (z. B. in "Bemerkungen"). Die Konvertierung in eine
# saubere Komma-CSV übernimmt nachher der Python-csv-Writer.
PROMPT_TEXT = (
    "Führe am angehängten JPEG eine OCR-Analyse durch. Es handelt sich um "
    "eine handschriftliche Tabelle mit genau 12 Spalten. Ignoriere die "
    "Header-Zeile der Tabelle (transkribiere sie nicht). Behalte die "
    "Struktur der Tabelle exakt bei: jede Zeile der Tabelle wird zu einer "
    "Zeile in deiner Ausgabe, jede Spalte wird durch genau ein Tabulator-"
    "Zeichen (\\t) getrennt. Gib NUR die reinen Daten zurück – kein "
    "Markdown, keine Code-Fences, keine Erklärungen, keine Kopfzeile. "
    "Wenn eine Zelle leer ist, lasse sie leer (zwei Tabulatoren "
    "hintereinander). Wenn ein Wert unleserlich ist, schreibe [unleserlich]."
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

    # OpenAI-kompatibles Chat-Completion-Format mit Bild-Input
    payload = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEXT},
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
        timeout=300,  # 5 Minuten – grosse Bilder können dauern
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


def parse_model_output(raw_text: str) -> list[list[str]]:
    """Wandelt die TSV-artige Modellausgabe in eine Liste von Zeilen um.

    Jede Zeile wird auf genau 12 Spalten normalisiert:
    - Zu wenige Spalten: Mit leeren Strings auffüllen.
    - Zu viele Spalten: Letzte Spalten werden zusammengefügt (typischer Fall
      bei langen Bemerkungen, in denen versehentlich ein Tab steht).
    """
    rows: list[list[str]] = []

    # Optional: Mögliche Code-Fences entfernen, falls das Modell sie doch setzt
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        # erste und letzte Zeile (Fence) entfernen
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    for line in cleaned.splitlines():
        if not line.strip():
            continue  # leere Zeilen überspringen
        cells = line.split("\t")
        cells = [c.strip() for c in cells]

        if len(cells) < 12:
            cells = cells + [""] * (12 - len(cells))
        elif len(cells) > 12:
            # Überschüssige Zellen zur 12. Spalte zusammenfügen
            cells = cells[:11] + [" ".join(cells[11:])]

        rows.append(cells)

    return rows


def write_csv(rows: list[list[str]], output_path: Path) -> None:
    """Schreibt die Header-Zeile und die Datenzeilen in eine CSV-Datei."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(
            f,
            delimiter=CSV_DELIMITER,
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writerow(HEADERS)
        writer.writerows(rows)


def process_single_image(
    image_path: Path, output_dir: Path, api_key: str
) -> Path:
    """Verarbeitet ein einzelnes Bild und schreibt die zugehörige CSV."""
    print(f"  → Sende {image_path.name} an die API …", flush=True)
    raw_output = call_ocr_api(image_path, api_key)

    rows = parse_model_output(raw_output)
    output_path = output_dir / f"{image_path.stem}.csv"
    write_csv(rows, output_path)

    print(
        f"  ✓ {len(rows)} Zeile(n) gespeichert in {output_path.name}",
        flush=True,
    )
    return output_path


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
        except Exception as exc:  # noqa: BLE001 – wir wollen alles abfangen
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
