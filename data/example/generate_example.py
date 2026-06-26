"""Generate an example cohort (raw schema).

The example data reproduces the raw-column schema in
``data/codebook_crosswalk.csv`` and plausible marginal distributions so the
full pipeline (graph construction -> multi-task GAT -> XAI -> fairness audit)
can be exercised end-to-end without any real patient records. Output is an
``.xlsx`` written to a git-ignored path and read by ``paper5_pipeline.data``.

No real patient data is used or reproduced. Targets are generated with a mild,
example dependence on the node features purely so the learning task is
non-trivial; the numbers carry no clinical meaning.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COMORB = [
    "comorb_none", "comorb_heart", "comorb_hypertension", "comorb_paod",
    "comorb_lung", "comorb_diabetes", "comorb_kidneys", "comorb_liver",
    "comorb_stroke", "comorb_neuological", "comorb_cancerlast5years",
    "comorb_depression", "comorb_gastrointestinal", "comorb_endometriosis",
    "comorb_arthritis", "comorb_incontinence", "comorb_uti",
]
TARGETS = ["brbi", "brsef", "brfu", "brhl", "brbs", "ef", "sf", "fi"]


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    education = rng.choice([1, 2, 3], size=n, p=[0.10, 0.50, 0.40])
    marital_status = rng.choice([0, 1, 2, 3], size=n, p=[0.20, 0.67, 0.09, 0.04])
    age = np.clip(rng.normal(56.4, 12.8, n), 23, 89).round(1)
    bmi = np.clip(rng.normal(25.8, 4.9, n), 14, 55).round(1)
    menarche = np.clip(rng.normal(13.2, 1.6, n), 8, 18).round(0)
    pregnancies = rng.poisson(1.8, n).clip(0, 12)
    menopause = (age > rng.normal(51, 4, n)).astype(int)
    contraceptive = rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 888], size=n,
                               p=[0.27, 0.18, 0.12, 0.08, 0.10, 0.07, 0.06, 0.10, 0.02])
    alcohol = rng.choice([0, 1, 2, 3], size=n, p=[0.41, 0.34, 0.18, 0.07])
    smoking = rng.choice([0, 1, 2], size=n, p=[0.58, 0.18, 0.24])
    bust = np.clip(rng.normal(92, 9.5, n), 60, 150).round(0)
    cup = rng.integers(0, 6, n)
    diagnosis = rng.choice([1, 2, 3, 4], size=n, p=[0.69, 0.08, 0.13, 0.10])
    pre_op = rng.binomial(1, 0.62, n)
    bc_first = rng.binomial(1, 0.88, n)
    cancer_breast = (diagnosis <= 2).astype(int)

    df = pd.DataFrame({
        "patient_id": [f"S{idx:05d}" for idx in range(n)],
        "education": education,
        "marital_status": marital_status,
        "age": age,
        "bmi": bmi,
        "menstruation_firsttime_age": menarche,
        "pregnancy_number": pregnancies,
        "menopause_yn": menopause,
        "contraceptive_kind": contraceptive,
        "alcohol": alcohol,
        "smokingstatus": smoking,
        "bust": bust,
        "cupsize": cup,
        "diagnosis": diagnosis,
        "pre_op": pre_op,
        "breastcancer_first": bc_first,
        "cancer_breast": cancer_breast,
    })
    for c in COMORB:
        p = 0.55 if c == "comorb_none" else rng.uniform(0.03, 0.18)
        df[c] = rng.binomial(1, p, n)

    # Mild example signal: targets depend weakly on a few features + noise.
    z_age = (age - age.mean()) / age.std()
    z_edu = (education - education.mean()) / education.std()
    partnered = (marital_status == 1).astype(float)
    base = 60 - 8 * z_age + 5 * z_edu + 6 * partnered
    for t in TARGETS:
        w = rng.normal(0, 3)
        df[t] = np.clip(base + w * z_edu + rng.normal(0, 18, n), 0, 100).round(1)

    # Inject realistic missingness (MICE is exercised downstream).
    for col, frac in {
        "bmi": 0.06, "menstruation_firsttime_age": 0.10, "bust": 0.12,
        "cupsize": 0.11, "contraceptive_kind": 0.08, "pre_op": 0.09,
    }.items():
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an example cohort.")
    parser.add_argument("--n", type=int, default=600, help="Number of records.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out", type=str, default="data/example/cohort.xlsx",
        help="Output .xlsx path (git-ignored).",
    )
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False)
    print(f"Wrote {len(df)} example records to {out}")


if __name__ == "__main__":
    main()
