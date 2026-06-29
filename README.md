# JELMO
## Die Angestellten des Warenhauses Jelmoli von 1899 bis 1945. Ein Workflow zur digitalen Aufbereitung von handschriftlichen, tabellarischen Quellen

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## Table of Contents

1. [Project Background](#1-project-background)
2. [Workflow Summary](#2-workflow-summary)
3. [Scripts — Overview](#3-scripts--overview)
4. [Pipeline — Typical Usage](#4-pipeline--typical-usage)
5. [Data Files](#5-data-files)
6. [Additional Notes](#6-additional-notes)
7. [Licence](#7-licence)
8. [Citation](#8-citation)

---

## 1. Project Background

### Archival source

The *Grands Magasins Jelmoli S.A.* was a department store founded in 1899 at Seidengasse 1 in Zurich and one of the earliest purpose-built department stores in Switzerland. The company systematically recorded all employees in handwritten registers known as *Polizeibücher. Kontrolle über die Angestellten und Arbeiter der Grands Magasins Jelmoli S.A.* (1878–1945). Since February 2025, the complete corporate archive of Jelmoli is publicly accessible at the **Stadtarchiv Zürich** (SAR), signature VII.573.:6.2.1. The eight surviving volumes contain an estimated **6,400 individual employee entries**.

The current workflow focuses on **Volume 3 (1910–1915)**, comprising 41 double pages. Each page constitutes a tabular record structured around 12 columns per employee entry:

| # | Column name (original German) |
|---|-------------------------------|
| 1 | No. |
| 2 | Familienname |
| 3 | Vorname |
| 4 | Heimatsort |
| 5 | Land |
| 6 | Geburtsjahr |
| 7 | Beruf |
| 8 | Bisheriger Wohnort |
| 9 | Eingetreten |
| 10 | Wohnung Strasse No. |
| 11 | Ausgetreten |
| 12 | Bemerkungen des Kontrollbeamten (Art. 12) |

The registers are written in **Kurrent**, a historical German cursive script common in German-speaking administrative documents of the late nineteenth and early twentieth centuries.

### Research focus

This project asks how digital methods can be usefully applied to the analysis of handwritten, tabular historical sources. The emphasis lies not only on the results, but also on how the digital methods employed can be embedded in a standardised, reproducible digital workflow — one that is accessible to historians with varying levels of digital literacy.

Substantively, the digitised data allows for the analysis of the **composition of the Jelmoli workforce**: geographic origins, occupational profiles, gender distribution, ages at entry, and periods of employment. The data also has potential for contributions to the social and demographic history of Zurich in the early twentieth century.

The project is situated within **Digital History / Digital Humanities** and is affiliated with the Digital Humanities activities at the **University of Bern**. It aims to produce FAIR-compliant, openly accessible research data in alignment with the requirements of the **Swiss National Science Foundation (SNF)**.

---

## 2. Workflow Summary

The digitisation workflow consists of four sequential steps:

```
[1] Image acquisition
    Scanned JPEGs of register pages (one JPEG per double page)
        ↓
[2] MLLM-based Handwritten Text Recognition (HTR)
    Python script → GPUstack API (qwen3-vl-8b-instruct) → structured JSON → CSV
        ↓
[3] Manual data cleaning
    Correction of transcription errors, expansion of abbreviations,
    date normalisation (YYYY-MM-DD), addition of gender and geocoordinates
        ↓
[4] Database, visualisation, and quality evaluation
    Import into Nodegoat (relational database + geographic visualisation)
    Python-based charts (gender ratio, age at entry, occupational structure)
    CER evaluation against a manually created ground truth
```

### Key technical decision: MLLM-based HTR vs. traditional HTR

Traditional HTR platforms (tested: Transkribus with a custom table model, Model ID 564709, mAP 38.9%) require a separate layout analysis and text recognition step. For tabular sources, this two-step process proved a structural weakness: segmentation errors in the layout model propagated into the text recognition output, making the total effort comparable to — or greater than — manual transcription.

As an alternative, this project uses **MLLM-based HTR**: the model `qwen3-vl-8b-instruct` (Alibaba Cloud / Qwen3-VL family), deployed on the **University of Bern GPUstack** (`https://gpustack.unibe.ch`). Unlike traditional HTR systems, this model analyses layout and text in a single step via its Vision Encoder (16×16 pixel patches) and Multimodal Rotary Position Embedding (MRoPE), which improves cell assignment accuracy in tabular sources. The column structure is enforced through explicit prompt engineering (not post-hoc normalisation), with temperature set to 0 throughout.

**Known limitation:** `qwen3-vl-8b-instruct` is a generalist vision-language model trained primarily on modern text and image data. It was not specifically trained on historical handwriting. This explains its comparatively high Character Error Rate in free-form cursive columns, particularly *Bemerkungen des Kontrollbeamten* (col. 12, CER ~92.9%). Specialist HTR tools trained on Kurrent would be expected to outperform this approach for that column specifically.

---

## 3. Scripts — Overview

| File | Purpose |
|------|---------|
| `MLLM-basedHTR_GPUStack_Qwen3-VL_v1.py` | Initial prototype: API call with TSV output; column structure not enforced; one CSV per JPEG |
| `MLLM-basedHTR_GPUStack_Qwen3-VL_v2.py` | JSON output format; header row included in transcription; one CSV per JPEG |
| `MLLM-basedHTR_GPUStack_Qwen3-VL_v3.py` | Combined output CSV for all JPEGs; source filename column added |
| `MLLM-basedHTR_GPUStack_Qwen3-VL_v4.py` | **Current version.** All 12 column names explicitly defined in prompt; robust JSON parsing; retry logic; combined output CSV |
| `Visualizations_Claude_Opus4.7.py` | Produces four coordinated charts: gender ratio (pie chart), cohort strength by year of birth and gender (bar chart), age at entry by gender (bar chart), occupational structure by year of birth (heatmap) |
| `CER_code.py` | Analyses the CER of the MLLM-based HTR per column, per row and in detail and allows a quality assesment|


All scripts require **Python 3.11 or later**.

Scripts v1–v4 are retained to document the iterative development of the prompting strategy. For new transcription runs, use **v4 only**.

**All Python scripts in this repository were produced via vibe coding** (i.e., iterative AI-assisted code generation using Claude AI, Opus 4.7) and subsequently reviewed and adapted to project requirements.

**Dependencies:**

```bash
pip install requests
```

No further third-party libraries are required for the HTR scripts. The visualisation script uses `pandas` and `matplotlib`.

---

## 4. Pipeline — Typical Usage

### Prerequisites

- Python ≥ 3.11
- VPN access to the University of Bern network
- Access to the University of Bern GPUstack (`https://gpustack.unibe.ch`)
- A valid personal GPUstack API key

### Step 1 — Set API key

```bash
export GPUSTACK_API_KEY="your_api_key_here"
```

This must be repeated each time a new terminal session is opened.

### Step 2 — Install dependency (once)

```bash
pip3 install requests
```

### Step 3 — Run HTR transcription (current version: v4)

```bash
python3 MLLM-basedHTR_GPUStack_Qwen3-VL_v4.py /path/to/jpeg_folder /path/to/output.csv
```

The script processes all `.jpg` / `.jpeg` files in the input folder alphabetically. Each file is sent to the API individually and takes approximately 30 seconds per page. The output CSV contains one row per register row, with the source filename prepended as the first column. Intermediate results are flushed to disk after each image to prevent data loss in the event of an error.

A summary log (number of files processed, rows written, any errors) is printed to the terminal on completion.

### Step 4 — Manual data cleaning

The raw CSV output requires correction before use in a database or for analysis. Recommended steps:

- Expand abbreviations that are only interpretable in the original document context (e.g., `"` for *ditto*)
- Normalise dates to ISO 8601 format (`YYYY-MM-DD`)
- Correct individual transcription errors in names, place names, and occupations
- Add columns for gender and residential geocoordinates (`lat` / `long`) if required for geographic visualisation

The *Bemerkungen* column (col. 12) was excluded from the cleaned dataset due to its very high CER and its absence in many entries.

### Step 5 — Import into Nodegoat (optional)

The cleaned CSV can be imported directly into [Nodegoat](https://nodegoat.net) by mapping CSV column headers to the corresponding object attributes. University of Bern students and staff can request a free Nodegoat domain via the Data Science Lab.

---

## 5. Data Files

| File | Description |
|------|-------------|
| `Data_Arbeitskontrolle_Bd3_1910-1915_1-10.csv` | cleaned and geocoded HTR output (v4) for pages 1–10 of Volume 3; 12 transcribed columns + source filename; approx. 200 employee entries |
| `CER_GrpundTruth_Arbeitskontrolle_Bd3_1910-1915_1-6.csv` | Ground Truth based on manual transcription for pages 1–6 of Volume 3 |
| `CER_Arbeitskontrolle_Bd3_1910-1915_Zeile.csv` | CER evaluation per row |
| `CER_Arbeitskontrolle_Bd3_1910-1915_Spalte.csv` | CER evaluation per column |
| `CER_Arbeitskontrolle_Bd3_1910-1915_Detail.csv` | CER evaluation per individual cell |


**Transcription scope:** As of June 2026, approximately 10 of 320 total pages (Volume 3) have been transcribed, corresponding to roughly 200 of an estimated 6,400 entries across all eight volumes.

---

## 6. Additional Notes

**CER results summary.** The overall CER across all columns for the first ten pages of Volume 3 was approximately 26.9%. Column-level results ranged from 2.7% (*No.*) to 92.9% (*Bemerkungen*). Columns with CER below 15% (good): *No.*, *Vorname*, *Familienname*. Columns with CER between 15–30% (moderate): *Beruf*, *Ausgetreten*, *Eingetreten*, *Geburtsjahr*, *Land*, *Heimatsort*. Columns with CER above 30% (problematic or high): *Wohnung Strasse No.*, *Bisheriger Wohnort*, *Bemerkungen des Kontrollbeamten*. The primary sources of error are historical abbreviations (e.g., ditto marks), handwritten numerals, and free-form cursive in the remarks column.

**Data cleaning approach.** Data cleaning was performed as close reading against the original source, prioritising source fidelity. Normalisation was applied to dates and common abbreviations. Occupation terms and place names were not yet standardised against controlled vocabularies (e.g., ISCO-08, GeoNames); this is a planned improvement for a future project stage.

**FAIR compliance status.** This repository is intended to be archived on [Zenodo](https://zenodo.org) with a persistent DOI issued via DataCite, satisfying FAIR principles F1 (globally unique identifier) and A2 (metadata persistence beyond the lifetime of the hosting platform). GitHub alone does not satisfy A2. A `CITATION.cff` file and a complete licence statement for the data files will be added prior to final archiving. A CC BY 4.0 licence applies to all data files; MIT applies to all code. The BORIS institutional repository (University of Bern) is the planned fallback for long-term preservation.

**CARE Principles.** The employee registers contain personal data of individuals, some of whom may have descendants. Whether the CARE Principles for Indigenous Data Governance are applicable to any entries in the dataset should be considered before public release of the full cleaned dataset.

**Reproducibility.** Temperature is set to 0 in all script versions. The full prompt text is documented within each script file. Given the same model version, API endpoint, and input images, results should be reproducible. Note that model updates on the GPUstack may affect output without prior notice.

---

## 7. Licence

- **Code** (all `.py` files): [MIT License](LICENSE)
- **Data** (all `.csv` files): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The underlying archival source (*Stadtarchiv Zürich*, VII.573.:6.2.1) remains subject to the access conditions of the Stadtarchiv Zürich. Transcriptions constitute a derived work and are released under CC BY 4.0.

---

## 8. Citation

Please cite this repository as:

> Jaggi, Amélie (2026). *JELMO: Die Angestellten des Warenhauses Jelmoli von 1899 bis 1945. Ein Workflow zur digitalen Aufbereitung von handschriftlichen, tabellarischen Quellen*. University of Bern. GitHub: https://github.com/ameliejaggi/JELMO
