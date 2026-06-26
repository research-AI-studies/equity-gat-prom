"""Generate all manuscript tables (xlsx) from outputs/results.json.

Tables:
  1  Cohort baseline characteristics
  2  Patient-similarity graph topology
  3  Aggregate predictive performance (GAT vs baselines, seed-averaged)
  4  Per-head regression metrics (GAT)
  5  Global feature attribution + cross-method agreement
  6  Equity audit (DPD/EOD/PPR) + mitigation
Supplementary S1 (k grid), S2 (threshold grid), S3 (per-seed), S4 (sensitivity).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper5_pipeline import config as C

R = json.loads(Path(C.RESULTS_JSON).read_text(encoding="utf-8"))
TAB = C.TAB_DIR
TAB.mkdir(parents=True, exist_ok=True)


def save(df, name):
    df.to_excel(TAB / name, index=False)
    print("  wrote", name)


def t1_cohort():
    c = R["cohort"]
    rows = [
        ("N (analytic cohort)", f"{c['n']}"),
        ("Age, years — mean (SD)", f"{c['age_mean']:.1f} ({c['age_sd']:.1f})"),
        ("Age, years — range", f"{c['age_min']:.0f}-{c['age_max']:.0f}"),
        ("BMI, kg/m^2 — mean (SD)", f"{c['bmi_mean']:.1f} ({c['bmi_sd']:.1f})"),
        ("Age at menarche, years — mean (SD)", f"{c['menarche_mean']:.1f} ({c['menarche_sd']:.1f})"),
        ("Parity — median (IQR)", f"{c['parity_median']:.0f} ({c['parity_q1']:.0f}-{c['parity_q3']:.0f})"),
    ]
    for lab, key in [("Education: low", "low"), ("Education: medium", "medium"), ("Education: high", "high")]:
        v = c["education"].get(key, 0)
        rows.append((lab, f"{v} ({100*v/c['n']:.1f}%)"))
    for lab, key in [("Partnered", "partnered"), ("Unpartnered", "unpartnered")]:
        v = c["partnership"].get(key, 0)
        rows.append((lab, f"{v} ({100*v/c['n']:.1f}%)"))
    for lab, key in [("Malignant (breast cancer/DCIS)", "malignant"), ("Benign", "benign")]:
        v = c["diagnosis"].get(key, 0)
        rows.append((lab, f"{v} ({100*v/c['n']:.1f}%)"))
    save(pd.DataFrame(rows, columns=["Characteristic", "Value"]), "Table_1_cohort.xlsx")


def t2_topology():
    t = R["graph"]["selected_topology"]
    keys = [("k (nearest neighbours)", "k", "{:.0f}"), ("Nodes", "n_nodes", "{:.0f}"),
            ("Edges", "n_edges", "{:.0f}"), ("Density", "density", "{:.4f}"),
            ("Mean degree", "mean_degree", "{:.1f}"), ("Median degree", "median_degree", "{:.0f}"),
            ("Degree IQR", None, None), ("Max degree", "max_degree", "{:.0f}"),
            ("Isolated nodes", "n_isolated", "{:.0f}"),
            ("Largest component (%)", "lcc_frac", "{:.3f}"),
            ("Diameter (LCC)", "lcc_diameter", "{:.0f}"),
            ("Mean shortest path (LCC)", "lcc_mean_path", "{:.2f}"),
            ("Clustering coefficient", "clustering_coef", "{:.3f}"),
            ("Edge-weight median", "edge_w_median", "{:.3f}"),
            ("Spectral-clustering silhouette", "silhouette", "{:.3f}")]
    rows = []
    for lab, k, fmt in keys:
        if k is None:
            rows.append((lab, f"{t['deg_q1']:.0f}-{t['deg_q3']:.0f}"))
        elif k in t:
            rows.append((lab, fmt.format(t[k])))
    save(pd.DataFrame(rows, columns=["Topology statistic", "Value"]), "Table_2_graph_topology.xlsx")


def t3_performance():
    ss = R["performance"]["seed_summary"]
    rows = []
    name = {"gat": "GAT (proposed)", "gcn": "GCN ablation", "mlp": "MLP", "xgb": "XGBoost"}
    for m in ["gat", "gcn", "mlp", "xgb"]:
        rows.append((name[m],
                     f"{ss[m]['macro_r2']['mean']:.3f} ({ss[m]['macro_r2']['sd']:.3f})",
                     f"{ss[m]['auc']['mean']:.3f} ({ss[m]['auc']['sd']:.3f})"))
    save(pd.DataFrame(rows, columns=["Model", "Macro-R2 mean (SD)", "AUC mean (SD)"]),
         "Table_3_performance.xlsx")


def t4_per_head():
    ph = R["performance"]["primary_detail"]["gat"]["per_head"]
    phm = R["performance"]["per_head_r2_seedmean"]
    rows = []
    for nm in C.REG_TARGETS:
        if nm not in ph:
            continue
        d = ph[nm]
        ci = f"({d.get('r2_lo', float('nan')):.3f}-{d.get('r2_hi', float('nan')):.3f})" if "r2_lo" in d else ""
        rows.append((C.TARGET_LABEL[nm], f"{phm.get(nm, {}).get('mean', d['r2']):.3f}",
                     ci, f"{d['mae']:.1f}", f"{d['rmse']:.1f}", f"{d['calibration_slope']:.2f}"))
    save(pd.DataFrame(rows, columns=["PRO scale", "R2 (seed-mean)", "R2 95% CI (seed42)",
                                     "MAE", "RMSE", "Calibration slope"]),
         "Table_4_per_head.xlsx")


def t5_explain():
    e = R.get("explainability", {})
    rows = [("SHAP-LIME per-patient mean rho", f"{e.get('shap_lime_per_patient_mean', float('nan')):.3f}"),
            ("SHAP-LIME fraction >= 0.5", f"{e.get('shap_lime_frac_above_0.5', float('nan')):.3f}")]
    for k, v in e.get("pairwise_spearman", {}).items():
        rows.append((f"Global ranking Spearman: {k}", f"{v:.3f}"))
    ah = e.get("attention_homophily", {})
    if ah.get("share_education_top_neighbour") is not None:
        rows.append(("Top-attended neighbour shares education", f"{ah['share_education_top_neighbour']:.3f}"))
        rows.append(("Top-attended neighbour shares partnership", f"{ah['share_partnership_top_neighbour']:.3f}"))
    save(pd.DataFrame(rows, columns=["Explainability metric", "Value"]), "Table_5_explainability.xlsx")
    # top features
    tf = e.get("top_features", [])
    save(pd.DataFrame(tf, columns=["Feature", "Mean |SHAP|"]), "Table_5b_top_features.xlsx")


def t6_fairness():
    un = R["fairness"]["unmitigated"]
    rows = []
    for col in ["education", "partnership", "diagnosis"]:
        d = un.get(col, {})
        if "dpd" in d:
            rows.append((col, f"{d['compare'][0]} vs {d['compare'][1]}",
                         f"{d['dpd']:.3f}", f"{d['eod']:.3f}",
                         f"{d['ppr']:.3f}" if d.get("ppr") else "n/a",
                         "yes" if d["triggered"] else "no"))
    df = pd.DataFrame(rows, columns=["Stratum", "Comparison", "DPD", "EOD", "PPR", "Triggered (>0.10)"])
    save(df, "Table_6_fairness.xlsx")
    # mitigation
    rw = R["fairness"].get("reweighed", {})
    if rw:
        mrows = []
        for col in ["education", "partnership", "diagnosis"]:
            if col in un and "dpd" in un[col] and col in rw and "dpd" in rw[col]:
                mrows.append((col, f"{un[col]['dpd']:.3f}", f"{rw[col]['dpd']:.3f}"))
        save(pd.DataFrame(mrows, columns=["Stratum", "DPD unmitigated", "DPD reweighed"]),
             "Table_6b_mitigation.xlsx")


def supp():
    save(pd.DataFrame(R["graph"]["k_grid"]), "Table_S1_k_grid.xlsx")
    save(pd.DataFrame(R["graph"]["threshold_grid"]), "Table_S2_threshold_grid.xlsx")
    # per-seed
    ss = R["performance"]["seed_summary"]
    seeds = R["meta"]["seeds"]
    rows = []
    for i, s in enumerate(seeds):
        rows.append((s, ss["gat"]["macro_r2"]["values"][i], ss["gat"]["auc"]["values"][i]))
    save(pd.DataFrame(rows, columns=["seed", "GAT macro-R2", "GAT AUC"]), "Table_S3_per_seed.xlsx")
    sens = R.get("sensitivity", {})
    if "graph_resolution_k" in sens:
        save(pd.DataFrame(sens["graph_resolution_k"]), "Table_S4_sensitivity_k.xlsx")
    if "threshold_variant" in sens:
        save(pd.DataFrame(sens["threshold_variant"]), "Table_S5_sensitivity_threshold.xlsx")


def main():
    print("Generating tables ->", TAB)
    t1_cohort(); t2_topology(); t3_performance(); t4_per_head()
    t5_explain(); t6_fairness(); supp()
    print("Tables done.")


if __name__ == "__main__":
    main()
