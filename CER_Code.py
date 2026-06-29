#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Berechnung der Character Error Rate (CER) für OCR-Transkriptionen.

Vergleicht eine OCR-Ausgabe (Hypothese) mit einer manuell erstellten
Ground-Truth-Transkription (Referenz). Beide CSV-Dateien müssen die
gleiche Struktur und gleiche Anzahl Datenzeilen haben (positionsbasierter
Vergleich, Zeile n gegen Zeile n, Zelle gegen Zelle).

Ergebnis:
    - Gesamt-CER über alle Datenzellen
    - CER pro Spalte
    - CER pro Zeile
    - Detail-CSV mit allen Zell-Paaren und ihrer CER
    - Zusammenfassungs-CSV für die Methodensektion

Die CER wird via Levenshtein-Distanz berechnet (eigene Implementierung,
ohne externe Abhängigkeiten):
    CER = (Substitutionen + Einfügungen + Löschungen) / Anzahl Zeichen GT

Konvention für leere Referenzzellen: Wenn die GT-Zelle leer ist und die
OCR-Zelle nicht, gibt es Einfügungen ohne Bezugspunkt – wir zählen die
Anzahl OCR-Zeichen als Fehler und setzen den GT-Längen-Beitrag auf 1
(damit die Division nicht durch 0 erfolgt). Solche Zellen werden in der
Detail-Auswertung markiert.

Verwendung:
    python cer_berechnen.py ground_truth.csv ocr_ausgabe.csv ausgabe_ordner/
"""

import argparse
import csv
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CER-Kernfunktion
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Berechnet die Levenshtein-Distanz zwischen zwei Strings.

    Substitution, Einfügung und Löschung kosten jeweils 1.
    Implementierung mit zeilenweiser Speicheroptimierung (O(min(len(a), len(b))).
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            substitute_cost = previous[j - 1] + (0 if char_a == char_b else 1)
            current[j] = min(insert_cost, delete_cost, substitute_cost)
        previous = current
    return previous[-1]


def cell_cer(reference: str, hypothesis: str) -> tuple[int, int]:
    """Gibt (Edit-Distanz, Referenzlänge) für eine einzelne Zelle zurück.

    Diese Rohwerte sind nötig, um später korrekt zu aggregieren:
    Gesamt-CER = sum(Edit-Distanzen) / sum(Referenzlängen).
    Eine Aggregation über die Mittelwerte einzelner Zell-CERs wäre
    statistisch fragwürdig, weil sie kurze Zellen gleich gewichtet wie
    lange.
    """
    ref = reference if reference is not None else ""
    hyp = hypothesis if hypothesis is not None else ""

    distance = levenshtein(ref, hyp)
    ref_len = len(ref)

    return distance, ref_len


# ---------------------------------------------------------------------------
# CSV-Vorverarbeitung
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[list[str]]:
    """Liest eine CSV ein und gibt sie als Liste von Zeilen zurück."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def is_meta_or_empty_row(row: list[str]) -> bool:
    """True, wenn die Zeile leer ist oder reine Strukturinformation enthält."""
    if not row:
        return True
    if all(cell.strip() == "" for cell in row):
        return True
    return False


def is_header_row(row: list[str]) -> bool:
    """Erkennt eine Header-Zeile heuristisch: Erste Zelle ist 'No' oder 'No.'
    und mindestens 'Familienname' kommt in der Zeile vor.
    """
    if not row:
        return False
    first = row[0].strip().lower()
    if first not in {"no", "no."}:
        return False
    return any("familienname" in c.strip().lower() for c in row)


def normalize_cell(value: str) -> str:
    """Entfernt führende/nachgestellte Leerzeichen und normalisiert
    Zeilenumbrüche zu einem einzelnen Leerzeichen.
    Bewusst KEINE weitergehende Normalisierung (z. B. Unicode-Kanonisierung,
    Lowercasing): die CER soll die rohe OCR-Qualität messen.
    """
    if value is None:
        return ""
    return value.replace("\n", " ").replace("\r", " ").strip()


def extract_data_rows(
    rows: list[list[str]],
) -> tuple[list[list[str]], list[str], int]:
    """Filtert die Datenzeilen aus einer geladenen CSV.

    Gibt zurück:
        - Liste der Datenzeilen (jeweils 12 Spalten)
        - Spaltennamen, falls eine Header-Zeile gefunden wurde
        - Anzahl übersprungener Header-/Meta-Zeilen
    """
    data_rows: list[list[str]] = []
    column_names: list[str] = []
    skipped = 0

    for row in rows:
        if is_meta_or_empty_row(row):
            skipped += 1
            continue
        # Titelzeile (eine einzelne Zelle mit Dateiname)
        if len(row) == 1:
            skipped += 1
            continue
        if is_header_row(row):
            if not column_names:
                column_names = [normalize_cell(c) for c in row]
            skipped += 1
            continue
        data_rows.append([normalize_cell(c) for c in row])

    return data_rows, column_names, skipped


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def evaluate(
    gt_rows: list[list[str]],
    ocr_rows: list[list[str]],
    column_names: list[str],
) -> dict:
    """Vergleicht die Datenzeilen zellenweise und gibt aggregierte
    Statistiken sowie die Detailpaare zurück.
    """
    n_pairs = min(len(gt_rows), len(ocr_rows))
    if len(gt_rows) != len(ocr_rows):
        print(
            f"Warnung: Unterschiedliche Zeilenzahl "
            f"(GT: {len(gt_rows)}, OCR: {len(ocr_rows)}). "
            f"Vergleich nur über die ersten {n_pairs} Zeilen.",
            file=sys.stderr,
        )

    n_cols = len(column_names) if column_names else 12

    # Aggregatoren
    total_edits = 0
    total_ref_len = 0
    perfect_cells = 0
    total_cells_compared = 0

    col_edits = [0] * n_cols
    col_ref_len = [0] * n_cols
    col_perfect = [0] * n_cols
    col_compared = [0] * n_cols

    detail_rows: list[dict] = []
    row_summaries: list[dict] = []

    for row_idx in range(n_pairs):
        gt = gt_rows[row_idx]
        ocr = ocr_rows[row_idx]

        # Auf gleiche Spaltenzahl bringen (mit Leerstrings auffüllen)
        if len(gt) < n_cols:
            gt = gt + [""] * (n_cols - len(gt))
        if len(ocr) < n_cols:
            ocr = ocr + [""] * (n_cols - len(ocr))

        row_edits = 0
        row_ref_len = 0

        for col_idx in range(n_cols):
            ref = gt[col_idx]
            hyp = ocr[col_idx]
            distance, ref_len = cell_cer(ref, hyp)

            # Sonderfall: GT leer, OCR nicht leer
            ref_len_effective = ref_len if ref_len > 0 else (
                1 if hyp else 0
            )

            cell_cer_value = safe_div(distance, ref_len_effective)
            is_perfect = (distance == 0)

            total_edits += distance
            total_ref_len += ref_len_effective
            total_cells_compared += 1
            if is_perfect:
                perfect_cells += 1

            col_edits[col_idx] += distance
            col_ref_len[col_idx] += ref_len_effective
            col_compared[col_idx] += 1
            if is_perfect:
                col_perfect[col_idx] += 1

            row_edits += distance
            row_ref_len += ref_len_effective

            detail_rows.append({
                "zeile_idx": row_idx + 1,
                "spalte_idx": col_idx + 1,
                "spaltenname": column_names[col_idx] if column_names else f"col{col_idx + 1}",
                "ground_truth": ref,
                "ocr_ausgabe": hyp,
                "edit_distanz": distance,
                "gt_laenge": ref_len,
                "cer_zelle": round(cell_cer_value, 4),
                "perfekt": "ja" if is_perfect else "nein",
            })

        row_summaries.append({
            "zeile_idx": row_idx + 1,
            "edit_distanz_summe": row_edits,
            "gt_zeichen_summe": row_ref_len,
            "cer_zeile": round(safe_div(row_edits, row_ref_len), 4),
        })

    return {
        "n_pairs": n_pairs,
        "total_cells": total_cells_compared,
        "perfect_cells": perfect_cells,
        "total_edits": total_edits,
        "total_ref_len": total_ref_len,
        "overall_cer": safe_div(total_edits, total_ref_len),
        "col_edits": col_edits,
        "col_ref_len": col_ref_len,
        "col_perfect": col_perfect,
        "col_compared": col_compared,
        "column_names": column_names,
        "detail_rows": detail_rows,
        "row_summaries": row_summaries,
    }


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

def write_detail_csv(detail_rows: list[dict], path: Path) -> None:
    if not detail_rows:
        return
    fieldnames = list(detail_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)


def write_column_summary_csv(result: dict, path: Path) -> None:
    rows = []
    for i, name in enumerate(result["column_names"] or [
        f"col{i + 1}" for i in range(len(result["col_edits"]))
    ]):
        cer = safe_div(result["col_edits"][i], result["col_ref_len"][i])
        rate_perfekt = safe_div(
            result["col_perfect"][i], result["col_compared"][i]
        )
        rows.append({
            "spaltenname": name,
            "zellen_verglichen": result["col_compared"][i],
            "perfekte_zellen": result["col_perfect"][i],
            "anteil_perfekt": round(rate_perfekt, 4),
            "edit_distanz_summe": result["col_edits"][i],
            "gt_zeichen_summe": result["col_ref_len"][i],
            "cer_spalte": round(cer, 4),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_row_summary_csv(row_summaries: list[dict], path: Path) -> None:
    if not row_summaries:
        return
    fieldnames = list(row_summaries[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_summaries)


def print_terminal_report(result: dict) -> None:
    print("\n" + "=" * 60)
    print("CER-AUSWERTUNG")
    print("=" * 60)
    print(f"Verglichene Datenzeilen:        {result['n_pairs']}")
    print(f"Verglichene Zellen insgesamt:   {result['total_cells']}")
    print(
        f"Perfekt transkribierte Zellen:  {result['perfect_cells']} "
        f"({safe_div(result['perfect_cells'], result['total_cells']):.1%})"
    )
    print(f"Edit-Distanz insgesamt:         {result['total_edits']}")
    print(f"GT-Zeichen insgesamt:           {result['total_ref_len']}")
    print(
        f"\n>>> Gesamt-CER: {result['overall_cer']:.4f} "
        f"({result['overall_cer']:.2%}) <<<"
    )

    print("\nCER pro Spalte (sortiert von schlecht zu gut):")
    column_data = []
    for i, name in enumerate(result["column_names"] or [
        f"col{i + 1}" for i in range(len(result["col_edits"]))
    ]):
        cer = safe_div(result["col_edits"][i], result["col_ref_len"][i])
        column_data.append((name, cer, result["col_compared"][i]))
    column_data.sort(key=lambda x: x[1], reverse=True)
    for name, cer, n in column_data:
        bar = "█" * int(cer * 50)
        print(f"  {cer:.2%}  {bar:<25}  {name}  (n={n})")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Berechnet die Character Error Rate (CER) durch Vergleich "
            "einer OCR-Ausgabe mit einer manuell erstellten Ground Truth."
        )
    )
    parser.add_argument(
        "ground_truth_csv",
        type=Path,
        help="Pfad zur Ground-Truth-CSV (Referenz).",
    )
    parser.add_argument(
        "ocr_csv",
        type=Path,
        help="Pfad zur OCR-Ausgabe-CSV (Hypothese).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Ordner für die Ergebnis-CSVs.",
    )
    args = parser.parse_args()

    if not args.ground_truth_csv.is_file():
        print(f"Fehler: GT-Datei nicht gefunden: {args.ground_truth_csv}",
              file=sys.stderr)
        return 1
    if not args.ocr_csv.is_file():
        print(f"Fehler: OCR-Datei nicht gefunden: {args.ocr_csv}",
              file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_raw = load_csv(args.ground_truth_csv)
    ocr_raw = load_csv(args.ocr_csv)

    gt_rows, gt_cols, gt_skipped = extract_data_rows(gt_raw)
    ocr_rows, ocr_cols, ocr_skipped = extract_data_rows(ocr_raw)

    print(f"Ground Truth:  {len(gt_rows)} Datenzeilen "
          f"({gt_skipped} Header/Meta-Zeilen übersprungen)")
    print(f"OCR-Ausgabe:   {len(ocr_rows)} Datenzeilen "
          f"({ocr_skipped} Header/Meta-Zeilen übersprungen)")

    # Spaltennamen bevorzugt aus der GT übernehmen
    column_names = gt_cols or ocr_cols

    result = evaluate(gt_rows, ocr_rows, column_names)

    # Ausgabedateien
    detail_path = args.output_dir / "cer_detail.csv"
    cols_path = args.output_dir / "cer_pro_spalte.csv"
    rows_path = args.output_dir / "cer_pro_zeile.csv"

    write_detail_csv(result["detail_rows"], detail_path)
    write_column_summary_csv(result, cols_path)
    write_row_summary_csv(result["row_summaries"], rows_path)

    print_terminal_report(result)

    print(f"\nGeschrieben:")
    print(f"  - Detailpaare pro Zelle:  {detail_path}")
    print(f"  - Zusammenfassung Spalte: {cols_path}")
    print(f"  - Zusammenfassung Zeile:  {rows_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
