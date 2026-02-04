# Knowledge Graph-Driven Price Benchmarking for API-Based Data Products: Reproduction Package

This repository provides **reproducible code and anonymized data** for analyzing and reasoning about **Knowledge Graph-Driven Price Benchmarking for API-Based Data Products** using **Knowledge Graph (KG)** techniques.

---

## 1. Project Overview

This project investigates pricing mechanisms of API-based data products via **graph-based reasoning**. By modeling relationships among **products**, **suppliers**, and **industries**, the pipeline performs utility reasoning to support **benchmarking, interpretation, and prediction** of data-product prices.

---

## 2. Repository Structure

```text
.
├── anymous/                          # Anonymized datasets
│   ├── neo4j_export/                 # Anonymized graph export from Neo4j (nodes & edges)
│   ├── name_price_anonymized.xlsx    # Anonymized price data from data exchanges (research object)
│   └── STEP0.*/                      # Intermediate variables and processed product descriptions
├── result_kg_reproduce/              # Reproduction outputs
│   ├── table_out/                    # Formatted output tables
│   ├── paper_tables_csv/             # Raw CSV tables used in the paper
│   └── paper_tables_v7_all_with_app6.xlsx  # Comprehensive summary table
├── KG_reasoning_pic_4/               # Generated reasoning visualization figures (paper-ready)
├── kg_utility_reasoning.py           # Main engine: KG-based analysis and reasoning
├── kg_utility_reasoning_reproduce.py # Reproduction runner: recreate core results
├── plt_kg_reasoning_pic.py           # Visualization runner: generate paper figures
└── requirements.txt                  # Python dependencies
```

---

## 3. Data Anonymization & Compliance

To mitigate legal and commercial risks, all sensitive content has been anonymized in accordance with:
- **Data Security Law of the People's Republic of China**
- **Personal Information Protection Law (PIPL)**

Anonymization measures include:

- **Salted Hashing**  
  Supplier names and specific product names are converted into **irreversible hash IDs**.

- **Feature Engineering for Replicability**  
  Raw product descriptions are transformed into **anonymized semantic tags** and/or **intermediate variables** (`STEP0.*`), enabling reproducibility without exposing raw business text.

> **Note:** Any attempt to de-anonymize the data is strictly prohibited.

---

## 4. Environment Setup

For consistency, we recommend **Python 3.8.2**.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 5. Reproduction Instructions

### Step 1 — Run Reasoning Reproduction

Run the reproduction script to regenerate the core analysis and pricing results:

```bash
python kg_utility_reasoning_reproduce.py
```

**Outputs**
- Generated under: `result_kg_reproduce/`
- Main summary table: `result_kg_reproduce/paper_tables_v7_all_with_app6.xlsx`

---

### Step 2 — Generate Visualizations

Generate the reasoning figures corresponding to the paper:

```bash
python plt_kg_reasoning_pic.py
```

**Outputs**
- Stored in: `KG_reasoning_pic_4/`

---

## 6. Core Modules

- **`kg_utility_reasoning.py`**  
  Main analysis engine for KG-based reasoning and result generation.

- **`kg_utility_reasoning_reproduce.py`**  
  One-click reproduction runner for core tables and summary outputs.

- **`plt_kg_reasoning_pic.py`**  
  Visualization suite that converts high-dimensional reasoning outputs into interpretable figures.

---

## 7. Disclaimer & License

- **Academic Use Only**  
  This repository is provided strictly for **scientific research and academic reproduction**.  
  Any **commercial use**, **redistribution**, or **attempt to de-anonymize** is prohibited.

- **No Warranty**  
  This code is provided **"as is"**, without warranty of any kind.

- **Compliance Responsibility**  
  Users are responsible for ensuring compliance with their **local data protection laws and regulations**.

---

## Citation & Feedback

If you find this repository useful, please consider citing our paper.

For questions, bug reports, or reproduction issues, please open an **Issue**.
