# Equity-Aware Graph Attention Networks for Breast-Cancer Patient-Reported Outcomes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20913218.svg)](https://doi.org/10.5281/zenodo.20913218)

A reproducible, **code-only** analytical pipeline that models eight EORTC
QLQ-C30/BR23 patient-reported outcome (PRO) scales with a **multi-task Graph
Attention Network (GAT)** over a patient-similarity graph, then audits the model
with **explainable AI** (SHAP, LIME, GNNExplainer, attention) and an
**algorithmic-fairness** analysis across social-determinant strata
(education, partnership, diagnosis).

This repository accompanies the manuscript *"Decoding Social Vulnerability and
Intimate Wellbeing Disparities: A Graph Attention Network and Explainable AI
Framework"* and contains **source code only**. No patient data, processed data,
model weights, figures, tables, or manuscript files are included.

---

## Pipeline overview

```
real cohort (local, private; Mendeley DOI 10.17632/wrhr5862cb.4)
        │                         └─ or: example cohort (default)
        ▼
[ preprocessing ]  load → recode → complete-case cohort → winsorise → MICE → encode → standardise
        │
        ▼
[ graph ]  Gower similarity → symmetric k-NN graph (k by silhouette)
        │
        ▼
[ model ]  multi-task GAT (7 regression + 1 binary head) + homoscedastic uncertainty weighting
        │            baselines: MLP, GCN ablation, XGBoost
        ▼
[ evaluation ]  R² / MAE / RMSE, AUC / F1 / calibration, multi-seed mean±SD, bootstrap CIs
        │
        ▼
[ explainability ]  KernelSHAP · LIME · GNNExplainer · attention · cross-method agreement
        │
        ▼
[ fairness ]  DPD / EOD / predictive parity / per-stratum calibration + reweighing mitigation
        │
        ▼
[ sensitivity ]  graph resolution k · threshold variant · cosine similarity · feature ablation
```

## Repository layout

| Path | Purpose |
|------|---------|
| `paper5_pipeline/` | Library: config, data, graph, models, training, explainability, fairness, sensitivity |
| `scripts/make_tables.py` | Build manuscript tables from `outputs/results.json` |
| `run_pipeline.py` | End-to-end orchestrator → writes `outputs/results.json` (single source of truth) |
| `config/default.yaml` | Human-readable mirror of `paper5_pipeline/config.py` defaults |
| `tests/` | Smoke tests on example data |

> The figure-generation script (`scripts/make_figures.py`) and every generated
> artefact are kept **offline** and are excluded by `.gitignore`.

## Quick start

```bash
# 1. Environment
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate an example cohort 
python data/example/generate_example.py --n 600 --out data/example/cohort.xlsx

# 3. Run the full pipeline (fast smoke run)
python run_pipeline.py --quick

# 4. Tests
pytest -q
```

To run on the **real cohort**, download it from Mendeley Data
(DOI: `10.17632/wrhr5862cb.4`), keep it outside the repository tree, and set
`PAPER5_RAW` to its path (see [`data/README.md`](data/README.md)). Such files
are excluded by `.gitignore` and must never be committed.

## Reproducibility

- Pinned dependencies: [`requirements.txt`](requirements.txt) / [`environment.yml`](environment.yml).
- Containerised runtime: [`Dockerfile`](Dockerfile).
- A single random seed (plus five replicate seeds) controls every stochastic
  step; all reported numbers are regenerated into `outputs/results.json`.

## Data and code availability

The dataset analysed in the manuscript is publicly available from Mendeley Data
(DOI: `10.17632/wrhr5862cb.4`). Processed data, intermediate artefacts, and all
generated outputs are not publicly released; they remain with the corresponding
author and are available on reasonable request, subject to the data licence.
A versioned archive of this repository is preserved on Zenodo under the citable
concept DOI [`10.5281/zenodo.20913218`](https://doi.org/10.5281/zenodo.20913218),
which always resolves to the latest release (v1.0.0:
[`10.5281/zenodo.20913219`](https://doi.org/10.5281/zenodo.20913219)).

## Citation

If you use this software, please cite the archived release:

```
The Authors (2026). Equity-Aware Graph Attention Networks for Breast-Cancer
Patient-Reported Outcomes (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.20913218
```

## Licence

Released under the MIT Licence. See [`LICENSE`](LICENSE).
