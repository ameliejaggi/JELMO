#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Visualisierungen zur Arbeitskontrolle Bd. 3 (1910-1915)
=============================================================================

Dieses Skript erzeugt vier Diagramme aus einer CSV-Datei mit personen-
bezogenen Daten aus einer historischen Arbeitskontrolle:

    1. Kuchendiagramm:    Geschlechterverhaeltnis
    2. Saeulendiagramm:   Jahrgangsstaerke nach Geburtsjahr und Geschlecht
    3. Heatmap:           Berufe nach Geburtsjahrgang
    4. Histogramm:        Alter beim Eintritt in die Anstellung

-----------------------------------------------------------------------------
METHODISCHE HINWEISE (siehe ausfuehrliche Reflexion im Begleittext)
-----------------------------------------------------------------------------
- Fehlende Werte werden je nach Diagramm unterschiedlich behandelt; die
  jeweilige Grundgesamtheit (n) steht im Diagrammtitel.
- Die Heatmap fasst Berufe mit < 5 Personen unter "Andere" zusammen
  (Schwellenwert in der Konstante BERUF_MINDEST_N anpassbar).
- Das Alter bei Eintritt wird als Eintrittsjahr minus Geburtsjahr berechnet
  und ist somit eine Naeherung (Abweichung +/- 1 Jahr je nach Geburtstag).
- Die Farben Rosa (Frauen) und Blau (Maenner) folgen einer Konvention,
  die selbst historisch ist und reflektiert werden sollte.

-----------------------------------------------------------------------------
NUTZUNG
-----------------------------------------------------------------------------
1. Python 3.8+ und Pakete installieren:
       pip install pandas matplotlib numpy

2. Pfad zur CSV-Datei und Ausgabeordner unten anpassen (siehe KONFIGURATION).

3. Skript ausfuehren:
       python arbeitskontrolle_visualisierungen.py

Erwartete CSV-Spalten:
    No., Familienname, Vorname, Heimatsort, Land, Geburtsjahr, Beruf,
    Geschlecht, Bisheriger Wohnort, Eingetreten, Wohnung Strasse No.,
    Wohnung_Koordinaten_lat, Wohnung_Koordinaten_long, Ausgetreten

Erstellt mit Unterstuetzung von Claude (Anthropic), Mai 2026.
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path


# =============================================================================
# KONFIGURATION - hier Pfade und Parameter anpassen
# =============================================================================

# Pfad zur Eingabedatei (CSV)
CSV_PFAD = "Arbeitskontrolle_Bd3_1910-1915_1-10_v4.csv"

# Ordner fuer die Ausgabe-Bilder (wird erstellt, falls nicht vorhanden)
AUSGABE_ORDNER = "diagramme"

# Datensatzbezeichnung fuer die Diagrammtitel
TITEL_QUELLE = "Arbeitskontrolle Bd. 3 (1910-1915)"

# Mindestanzahl Personen, damit ein Beruf in der Heatmap einzeln gezeigt wird;
# alle anderen werden unter "Andere" zusammengefasst
BERUF_MINDEST_N = 5

# Farben (Rosa fuer Frauen, Blau fuer Maenner) - bewusste Setzung, vgl. Reflexion
FARBE_FRAUEN = "#D4A5A5"
FARBE_MAENNER = "#6B8CAE"

# Bildaufloesung
DPI = 200


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def lade_daten(pfad):
    """CSV einlesen, Eintrittsdatum parsen und Eintrittsalter berechnen.

    Wichtig: Das Eintrittsalter wird als Eintrittsjahr - Geburtsjahr
    berechnet. Das ist eine Naeherung, kein exaktes Alter.
    """
    df = pd.read_csv(pfad)
    df["Eintrittsdatum"] = pd.to_datetime(df["Eingetreten"], errors="coerce")
    df["Eintrittsjahr"] = df["Eintrittsdatum"].dt.year
    df["Alter_Eintritt"] = df["Eintrittsjahr"] - df["Geburtsjahr"]
    return df


def achsen_aufraeumen(ax):
    """Entfernt obere und rechte Achsenlinien fuer ein klareres Bild."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# DIAGRAMM 1: KUCHENDIAGRAMM Geschlechterverhaeltnis
# =============================================================================

def erstelle_kuchendiagramm(df, ausgabe_pfad):
    """Kuchendiagramm zum Geschlechterverhaeltnis.

    Methodischer Hinweis: Personen ohne Geschlechtsangabe (NaN) werden
    durch value_counts() automatisch ausgeschlossen.
    """
    geschlecht_counts = df["Geschlecht"].value_counts()  # ohne NaN
    labels_map = {"F": "Frauen", "M": "Maenner"}
    labels = [f"{labels_map[g]} ({n})" for g, n in geschlecht_counts.items()]
    farben = [FARBE_FRAUEN if g == "F" else FARBE_MAENNER
              for g in geschlecht_counts.index]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        geschlecht_counts.values,
        labels=labels,
        colors=farben,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 13},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(13)

    n_total = geschlecht_counts.sum()
    n_fehlend = df["Geschlecht"].isna().sum()
    untertitel = f"{TITEL_QUELLE}, n = {n_total}"
    if n_fehlend > 0:
        untertitel += f" (ohne Angabe: {n_fehlend})"
    ax.set_title(f"Geschlechterverhaeltnis\n{untertitel}",
                 fontsize=15, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(ausgabe_pfad, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()


# =============================================================================
# DIAGRAMM 2: SAEULENDIAGRAMM Jahrgangsstaerke nach Geschlecht
# =============================================================================

def erstelle_jahrgangsdiagramm(df, ausgabe_pfad):
    """Gestapelte Saeulen: Geburtsjahr x Anzahl, eingefaerbt nach Geschlecht.

    Methodischer Hinweis: Personen ohne Geburtsjahr werden ausgeschlossen.
    Jahre ohne Eintraege werden als Luecken (Hoehe 0) dargestellt.
    """
    df_v = df.dropna(subset=["Geburtsjahr"]).copy()
    df_v["Geburtsjahr"] = df_v["Geburtsjahr"].astype(int)

    # Kreuztabelle Geburtsjahr x Geschlecht
    kreuz = pd.crosstab(df_v["Geburtsjahr"], df_v["Geschlecht"])

    # Luecken in der Jahresreihe auffuellen (sonst springen die Saeulen)
    alle_jahre = range(kreuz.index.min(), kreuz.index.max() + 1)
    kreuz = kreuz.reindex(alle_jahre, fill_value=0)
    for sp in ["F", "M"]:
        if sp not in kreuz.columns:
            kreuz[sp] = 0

    frauen = kreuz["F"].values
    maenner = kreuz["M"].values
    gesamt = frauen + maenner

    fig, ax = plt.subplots(figsize=(15, 7))
    jahre_str = kreuz.index.astype(str)

    ax.bar(jahre_str, frauen, color=FARBE_FRAUEN, edgecolor="white",
           linewidth=0.8, label=f"Frauen (n = {frauen.sum()})")
    ax.bar(jahre_str, maenner, bottom=frauen, color=FARBE_MAENNER,
           edgecolor="white", linewidth=0.8,
           label=f"Maenner (n = {maenner.sum()})")

    # Gesamtsumme oberhalb jeder Saeule
    for i, t in enumerate(gesamt):
        if t > 0:
            ax.text(i, t + 0.2, str(t), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    # Teilwerte innerhalb der Segmente
    for i, (f, m) in enumerate(zip(frauen, maenner)):
        if f > 0:
            ax.text(i, f / 2, str(f), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        if m > 0:
            ax.text(i, f + m / 2, str(m), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

    ax.set_xlabel("Geburtsjahr", fontsize=12, fontweight="bold")
    ax.set_ylabel("Anzahl Personen", fontsize=12, fontweight="bold")
    fehlend_gj = df["Geburtsjahr"].isna().sum()
    ax.set_title(
        f"Jahrgangsstaerke nach Geburtsjahr und Geschlecht\n"
        f"{TITEL_QUELLE}, n = {len(df_v)} "
        f"(fehlende Geburtsjahre: {fehlend_gj})",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_yticks(np.arange(0, gesamt.max() + 2, 1))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    achsen_aufraeumen(ax)
    ax.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(ausgabe_pfad, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()


# =============================================================================
# DIAGRAMM 3: HEATMAP Berufe x Geburtsjahrgang
# =============================================================================

def erstelle_berufs_heatmap(df, ausgabe_pfad):
    """Heatmap: Beruf (Zeile) x Geburtsjahr (Spalte), Farbe = Anzahl.

    Methodischer Hinweis:
    - Personen ohne Geburtsjahr oder ohne Beruf werden ausgeschlossen.
    - Berufe mit < BERUF_MINDEST_N Personen werden unter "Andere"
      zusammengefasst; das ist eine willkuerliche Schwelle.
    - Der Farbverlauf laeuft von sehr hellem Rosa (wenige) bis dunklem
      Rotbraun (viele); kein Blau, damit keine Verwechslung mit der
      Geschlechterkodierung entsteht.
    """
    df_v = df.dropna(subset=["Geburtsjahr", "Beruf"]).copy()
    df_v["Geburtsjahr"] = df_v["Geburtsjahr"].astype(int)

    # Haeufige Berufe einzeln, seltene unter "Andere"
    beruf_counts = df_v["Beruf"].value_counts()
    haeufige_berufe = beruf_counts[beruf_counts >= BERUF_MINDEST_N].index.tolist()
    df_v["Beruf_gruppe"] = df_v["Beruf"].apply(
        lambda b: b if b in haeufige_berufe else "Andere"
    )

    # Reihenfolge: nach Haeufigkeit absteigend, "Andere" zuletzt
    reihenfolge = haeufige_berufe + ["Andere"]

    kreuz = pd.crosstab(df_v["Beruf_gruppe"], df_v["Geburtsjahr"])
    alle_jahre = range(int(df_v["Geburtsjahr"].min()),
                       int(df_v["Geburtsjahr"].max()) + 1)
    kreuz = kreuz.reindex(index=reihenfolge, columns=alle_jahre, fill_value=0)

    # Y-Labels mit Gesamtzahl pro Beruf
    labels_y = [f"{b} (n={kreuz.loc[b].sum()})" for b in kreuz.index]

    # Rosa-Verlauf
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rosa", ["#FBF5F5", "#EFD4D4", "#D4A5A5", "#A86B6B", "#6E3A3A"]
    )

    fig, ax = plt.subplots(figsize=(15, 7))
    im = ax.imshow(kreuz.values, cmap=cmap, aspect="auto", vmin=0)

    # Werte in die Zellen schreiben
    max_val = kreuz.values.max()
    for i in range(kreuz.shape[0]):
        for j in range(kreuz.shape[1]):
            v = kreuz.values[i, j]
            if v > 0:
                # Heller Text auf dunklem Hintergrund, dunkler auf hellem
                farbe = "white" if v >= max_val * 0.5 else "#333333"
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=9, color=farbe, fontweight="bold")

    ax.set_xticks(range(len(kreuz.columns)))
    ax.set_xticklabels(kreuz.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(kreuz.index)))
    ax.set_yticklabels(labels_y, fontsize=10)
    ax.set_xlabel("Geburtsjahr", fontsize=12, fontweight="bold")
    ax.set_ylabel("Beruf", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Berufe nach Geburtsjahrgang\n"
        f"{TITEL_QUELLE}, n = {len(df_v)} "
        f"(Berufe mit >= {BERUF_MINDEST_N} Personen einzeln, "
        f"Rest als <<Andere>>)",
        fontsize=13, fontweight="bold", pad=15,
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Anzahl Personen", fontsize=10)

    plt.tight_layout()
    plt.savefig(ausgabe_pfad, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()


# =============================================================================
# DIAGRAMM 4: HISTOGRAMM Alter beim Eintritt
# =============================================================================

def erstelle_eintrittsalter_histogramm(df, ausgabe_pfad):
    """Histogramm: Alter bei Eintritt (Jahre), gestapelt nach Geschlecht.

    Methodischer Hinweis:
    - Alter = Eintrittsjahr - Geburtsjahr (Naeherung, +/- 1 Jahr).
    - Personen ohne Geburtsjahr oder ohne Geschlechtsangabe werden
      ausgeschlossen.
    - Mittelwert und Median werden als gestrichelte/gepunktete Linien
      eingezeichnet.
    """
    df_alter = df.dropna(subset=["Alter_Eintritt", "Geschlecht"]).copy()
    df_alter["Alter_Eintritt"] = df_alter["Alter_Eintritt"].astype(int)

    alter_min = int(df_alter["Alter_Eintritt"].min())
    alter_max = int(df_alter["Alter_Eintritt"].max())
    alle_alter = range(alter_min, alter_max + 1)

    frauen = (df_alter[df_alter["Geschlecht"] == "F"]["Alter_Eintritt"]
              .value_counts().reindex(alle_alter, fill_value=0))
    maenner = (df_alter[df_alter["Geschlecht"] == "M"]["Alter_Eintritt"]
               .value_counts().reindex(alle_alter, fill_value=0))

    fig, ax = plt.subplots(figsize=(15, 7))
    alter_str = [str(a) for a in alle_alter]

    ax.bar(alter_str, frauen.values, color=FARBE_FRAUEN, edgecolor="white",
           linewidth=0.8, label=f"Frauen (n = {frauen.sum()})")
    ax.bar(alter_str, maenner.values, bottom=frauen.values,
           color=FARBE_MAENNER, edgecolor="white", linewidth=0.8,
           label=f"Maenner (n = {maenner.sum()})")

    # Gesamtsumme oberhalb der Saeulen
    gesamt = frauen.values + maenner.values
    for i, t in enumerate(gesamt):
        if t > 0:
            ax.text(i, t + 0.15, str(t), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    # Statistik-Linien (Position relativ zum X-Index)
    mw = df_alter["Alter_Eintritt"].mean()
    md = df_alter["Alter_Eintritt"].median()
    ax.axvline(mw - alter_min, color="#2C4A6B", linestyle="--", linewidth=1.8,
               label=f"Mittelwert: {mw:.1f} Jahre", alpha=0.8)
    ax.axvline(md - alter_min, color="#8B3A3A", linestyle=":", linewidth=1.8,
               label=f"Median: {md:.0f} Jahre", alpha=0.8)

    ax.set_xlabel("Alter bei Eintritt (in Jahren)",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel("Anzahl Personen", fontsize=12, fontweight="bold")
    fehlend = len(df) - len(df_alter)
    ax.set_title(
        f"Alter beim Eintritt in die Anstellung\n"
        f"{TITEL_QUELLE}, n = {len(df_alter)} (fehlend: {fehlend})",
        fontsize=14, fontweight="bold", pad=15,
    )
    # Schrittweite auf y-Achse je nach Datenmenge anpassen
    schritt = max(1, gesamt.max() // 15)
    ax.set_yticks(np.arange(0, gesamt.max() + 2, schritt))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.xticks(rotation=0, fontsize=10)
    achsen_aufraeumen(ax)
    ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(ausgabe_pfad, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()


# =============================================================================
# HAUPTPROGRAMM
# =============================================================================

def main():
    # Ausgabeordner anlegen
    ausgabe = Path(AUSGABE_ORDNER)
    ausgabe.mkdir(exist_ok=True)

    # Daten einlesen
    print(f"Lese Daten aus: {CSV_PFAD}")
    df = lade_daten(CSV_PFAD)
    print(f"  Eintraege gesamt:  {len(df)}")
    print(f"  Geschlecht F/M/o.A.: "
          f"{(df['Geschlecht'] == 'F').sum()} / "
          f"{(df['Geschlecht'] == 'M').sum()} / "
          f"{df['Geschlecht'].isna().sum()}")
    print(f"  Geburtsjahre:      "
          f"{int(df['Geburtsjahr'].min())}-{int(df['Geburtsjahr'].max())} "
          f"(fehlend: {df['Geburtsjahr'].isna().sum()})")
    print(f"  Berufe (versch.):  {df['Beruf'].nunique()}")
    print()

    # Diagramme erstellen
    print("Erstelle Diagramme...")
    erstelle_kuchendiagramm(df, ausgabe / "1_geschlechterverhaeltnis.png")
    print("  [1/4] Kuchendiagramm Geschlechterverhaeltnis")

    erstelle_jahrgangsdiagramm(df, ausgabe / "2_jahrgangsstaerke_geschlecht.png")
    print("  [2/4] Saeulendiagramm Jahrgangsstaerke")

    erstelle_berufs_heatmap(df, ausgabe / "3_berufe_jahrgaenge_heatmap.png")
    print("  [3/4] Heatmap Berufe x Geburtsjahr")

    erstelle_eintrittsalter_histogramm(df, ausgabe / "4_alter_eintritt.png")
    print("  [4/4] Histogramm Eintrittsalter")

    print(f"\nFertig. Alle Diagramme im Ordner: {ausgabe.resolve()}")


if __name__ == "__main__":
    main()
