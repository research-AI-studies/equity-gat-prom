"""Central configuration: paths, seeds, feature/target schema, hyperparameters.

Everything that controls the analysis lives here so the whole study is defined
by one file and one random seed.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent
OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
TAB_DIR = OUTPUTS / "tables"
RESULTS_JSON = OUTPUTS / "results.json"

# Data source. The real cohort is the public Charite PROM Baseline deposit on
# Mendeley Data (DOI: 10.17632/wrhr5862cb.4); it is NOT redistributed here. By
# default the pipeline runs on the git-ignored example cohort that
# reproduces the analytic schema without any patient records. Point the
# environment variable PAPER5_RAW at a local copy of the real `.xlsx` to run on
# the genuine cohort.
RAW_XLSX = Path(os.environ.get(
    "PAPER5_RAW",
    str(ROOT / "data" / "example" / "cohort.xlsx"),
))

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
EXTRA_SEEDS = [43, 44, 45, 46, 47]      # for training-variance quantification
ALL_SEEDS = [SEED] + EXTRA_SEEDS

# --------------------------------------------------------------------------- #
# Feature schema. Node features blend sociodemographic/reproductive variables
# with low-missingness clinical variables (comorbidity profile, diagnosis,
# surgical/history flags). Heavily-missing tumour-pathology fields (histotype,
# grade, ER/PR/HER2, laterality; 42-65% missing) are deliberately EXCLUDED to
# avoid unreliable imputation. Financial difficulty (EORTC `fi`) is NOT a node
# feature because it is also a prediction target (would be label leakage).
# Social-determinant variables additionally serve as the equity strata.
# --------------------------------------------------------------------------- #
SOCIO = [
    "education", "marital_status", "age", "bmi", "menarche", "pregnancies",
    "menopausal", "contraceptive", "alcohol", "smoking", "bust", "cup",
]
COMORB = [
    "comorb_none", "comorb_heart", "comorb_hypertension", "comorb_paod",
    "comorb_lung", "comorb_diabetes", "comorb_kidneys", "comorb_liver",
    "comorb_stroke", "comorb_neuological", "comorb_cancerlast5years",
    "comorb_depression", "comorb_gastrointestinal", "comorb_endometriosis",
    "comorb_arthritis", "comorb_incontinence", "comorb_uti",
]
CLINICAL = ["diagnosis", "pre_op", "breastcancer_first", "cancer_breast"] + COMORB
NODE_FEATURES = SOCIO + CLINICAL

# Encoding roles on the mapped feature frame.
CONTINUOUS = ["age", "bmi", "menarche", "pregnancies", "alcohol", "smoking", "bust", "cup"]
NOMINAL = ["education", "marital_status", "contraceptive", "diagnosis"]
BINARY = ["menopausal", "pre_op", "breastcancer_first", "cancer_breast"] + COMORB

# Cohort definition (complete-case) is anchored on the sociodemographic/
# reproductive block; clinical missingness (~10%) is MICE-imputed within cohort.
COHORT_COMPLETE_CASE = SOCIO

# --------------------------------------------------------------------------- #
# Targets — 8 EORTC PRO scales (0-100). Higher = better functioning for the
# functional scales; `fi` (financial difficulties) is a symptom scale.
# --------------------------------------------------------------------------- #
TARGETS = ["brbi", "brsef", "brfu", "brhl", "brbs", "ef", "sf", "fi"]
TARGET_LABEL = {
    "brbi": "Body image", "brsef": "Sexual functioning", "brfu": "Future perspectives",
    "brhl": "Hair-loss symptoms", "brbs": "Breast symptoms",
    "ef": "Emotional functioning", "sf": "Social functioning", "fi": "Financial difficulties",
}
BR23 = ["brbi", "brsef", "brfu", "brhl", "brbs"]
C30 = ["ef", "sf", "fi"]
# 8 task heads = 7 regression + 1 binary classification head.
# The binary head is BR23 sexual functioning (clinically poor vs adequate);
# the remaining seven scales are modelled as regression heads.
BINARY_TARGET = "brsef"
BINARY_CUTOFF = 50.0   # EORTC functional scales: <50 flags clinically poor functioning
REG_TARGETS = ["brbi", "brfu", "brhl", "brbs", "ef", "sf", "fi"]

# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
# Primary construction: symmetric k-NN on Gower similarity.
K_GRID = [10, 15, 20, 25, 30, 40, 50]
# Threshold-graph variant (sensitivity analysis only).
TAU_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TAU_DEFAULT = 0.60

# --------------------------------------------------------------------------- #
# Equity strata
# --------------------------------------------------------------------------- #
EDU_STRATA = {1: "low", 2: "medium", 3: "high"}
# partnership: 1 -> partnered ; {0,2,3} -> unpartnered
def partnership(ms):
    return "partnered" if ms == 1 else "unpartnered"
DIAG_MALIGNANT = {1, 2}   # breast cancer, DCIS -> malignant ; {3,4} benign
FAIR_THRESHOLD = 0.10     # DPD/EOD action threshold

# --------------------------------------------------------------------------- #
# Split / training
# --------------------------------------------------------------------------- #
SPLIT = (0.70, 0.15, 0.15)   # train / val(test selection) / held-out test
CV_FOLDS = 5
N_BOOTSTRAP = 1000

# Pragmatic-but-honest random search (CPU). Manuscript's aspirational 50-config /
# 4xA100 search is reduced to a locally reproducible budget; the actual budget is
# recorded in results.json and reported verbatim in the manuscript.
RANDOM_SEARCH_N = int(os.environ.get("PAPER5_SEARCH_N", "15"))
MAX_EPOCHS = int(os.environ.get("PAPER5_EPOCHS", "250"))
PATIENCE = 30
GRAD_CLIP = 5.0

# Sensitivity-analysis budget (cheaper than primary; recorded in results.json).
SENS_EPOCHS = 120
MAX_SENS_EDGES = 120_000   # skip pathologically dense threshold graphs

SEARCH_SPACE = {
    "k": [10, 20, 30],
    "n_layers": [1, 2, 3],
    "hidden": [32, 64, 128],
    "heads": [2, 4, 8],
    "dropout": [0.1, 0.3, 0.5],
    "lr": [1e-4, 1e-3, 1e-2],
    "weight_decay": [0.0, 1e-5, 1e-4],
}

# Explainability budgets (kept tractable on CPU; recorded in results.json).
EXPLAIN_SUBSET = 40
SHAP_BG = 8
SHAP_NSAMPLES = 80
LIME_SAMPLES = 400
GNNEXPLAINER_EPOCHS = 80
GNNEXPLAINER_SUBSET = 40
