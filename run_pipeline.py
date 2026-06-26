"""Paper 5 — end-to-end orchestrator.

Runs the complete seeded analysis and writes outputs/results.json, the single
source of truth for every number, figure, and table in the manuscript.

Usage:
    python run_pipeline.py            # full run
    python run_pipeline.py --quick    # smaller budgets for a fast smoke test
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np

from paper5_pipeline import config as C
from paper5_pipeline import data, graph, train, explain, fairness, sensitivity


def jdefault(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(str(type(o)))


def cohort_descriptives(co):
    raw, feat, strata = co.raw, co.raw_features, co.strata
    num = lambda c: __import__("pandas").to_numeric(raw[c], errors="coerce")
    age = num("age")
    bmi = feat["bmi"].astype(float)
    men = feat["menarche"].astype(float)
    parity = feat["pregnancies"].astype(float)
    d = {
        "n": int(len(raw)),
        "age_mean": float(age.mean()), "age_sd": float(age.std()),
        "age_min": float(age.min()), "age_max": float(age.max()),
        "bmi_mean": float(bmi.mean()), "bmi_sd": float(bmi.std()),
        "menarche_mean": float(men.mean()), "menarche_sd": float(men.std()),
        "parity_median": float(parity.median()),
        "parity_q1": float(parity.quantile(.25)), "parity_q3": float(parity.quantile(.75)),
        "education": strata["education"].value_counts().to_dict(),
        "partnership": strata["partnership"].value_counts().to_dict(),
        "diagnosis": strata["diagnosis"].value_counts().to_dict(),
        "diagnosis_code": num("diagnosis").value_counts().sort_index().to_dict(),
        "feature_missingness_pct": co.feat_missing,
        "target_missingness_pct": {t: float(co.Y[t].isna().mean() * 100) for t in C.TARGETS},
        "binary_positive_rate": float(np.nanmean(co.y_bin)),
    }
    return d


def main(quick=False):
    t0 = time.time()
    if quick:
        C.RANDOM_SEARCH_N = 4
        C.MAX_EPOCHS = 80
        C.EXPLAIN_SUBSET = 12
        C.GNNEXPLAINER_SUBSET = 15
        C.ALL_SEEDS_RUN = [42, 43]
    seeds = getattr(C, "ALL_SEEDS_RUN", C.ALL_SEEDS)

    R = {"meta": {
        "generated": datetime.now(timezone.utc).isoformat(),
        "seed": C.SEED, "seeds": seeds,
        "random_search_n": C.RANDOM_SEARCH_N, "max_epochs": C.MAX_EPOCHS,
        "python": platform.python_version(),
        "note": "Single source of truth for Paper 5. All manuscript numbers are "
                "injected from this file.",
    }}

    print("[1/8] loading cohort ..."); co = data.load_cohort()
    R["cohort"] = cohort_descriptives(co)
    R["features"] = {"encoded_dim": int(co.X.shape[1]), "names": co.feat_names,
                     "node_feature_blocks": {"sociodemographic_reproductive": C.SOCIO,
                                             "clinical": C.CLINICAL}}

    print("[2/8] Gower similarity + topology ..."); S = graph.gower_matrix(co.raw_features)
    sil_k, k_grid = graph.select_k(S)
    R["graph"] = {"silhouette_selected_k": int(sil_k), "k_grid": k_grid,
                  "threshold_grid": graph.threshold_grid(S)}

    print("[3/8] hyperparameter random search (graph-aware CV) ...")
    tr0, va0, te0 = train.make_splits(co, C.SEED)
    train_pool = np.sort(np.concatenate([tr0, va0]))  # 85% for CV pool
    best_cfg, search_hist = train.random_search_graph(
        co, S, np.sort(tr0), C.SEED, n=C.RANDOM_SEARCH_N, max_epochs=100)
    R["model_selection"] = {"best_config": best_cfg,
                            "search_history": search_hist,
                            "search_space": C.SEARCH_SPACE}
    best_k = best_cfg["k"]
    A = graph.knn_adjacency(S, best_k)
    R["graph"]["selected_topology"] = graph.full_topology(S, A, "k", best_k)
    ei, ew = graph.build_edges(A, S)

    print("[4/8] multi-seed final training (GAT/GCN/MLP/XGB) ...")
    per_seed = {"gat": [], "gcn": [], "mlp": [], "xgb": []}
    per_head_acc = {nm: [] for nm in C.REG_TARGETS}
    margins = {"gat_minus_mlp_r2": [], "gat_minus_gcn_r2": [],
               "gat_minus_mlp_auc": [], "gat_minus_gcn_auc": []}
    primary = {}
    for s in seeds:
        tr, va, te = train.make_splits(co, s)
        Tn = train.make_tensors(co, ei, ew, tr)
        gat, _ = train.train_model("gat", Tn, tr, va, best_cfg, s, max_epochs=C.MAX_EPOCHS)
        gcn, _ = train.train_model("gcn", Tn, tr, va, best_cfg, s, max_epochs=C.MAX_EPOCHS)
        mlp, _ = train.train_model("mlp", Tn, tr, va, best_cfg, s, max_epochs=C.MAX_EPOCHS)
        rg = train.evaluate(gat, Tn, te, n_boot=C.N_BOOTSTRAP if s == C.SEED else 0, seed=s)
        rc = train.evaluate(gcn, Tn, te, n_boot=0, seed=s)
        rm = train.evaluate(mlp, Tn, te, n_boot=0, seed=s)
        rx = train.xgboost_baseline(co, tr, te, seed=s)
        per_seed["gat"].append({"macro_r2": rg["macro_r2"], "auc": rg["binary"]["auc"]})
        per_seed["gcn"].append({"macro_r2": rc["macro_r2"], "auc": rc["binary"]["auc"]})
        per_seed["mlp"].append({"macro_r2": rm["macro_r2"], "auc": rm["binary"]["auc"]})
        per_seed["xgb"].append({"macro_r2": rx["macro_r2"], "auc": rx["binary"]["auc"]})
        for nm in C.REG_TARGETS:
            if nm in rg["per_head"]:
                per_head_acc[nm].append(rg["per_head"][nm]["r2"])
        margins["gat_minus_mlp_r2"].append(rg["macro_r2"] - rm["macro_r2"])
        margins["gat_minus_gcn_r2"].append(rg["macro_r2"] - rc["macro_r2"])
        margins["gat_minus_mlp_auc"].append(rg["binary"]["auc"] - rm["binary"]["auc"])
        margins["gat_minus_gcn_auc"].append(rg["binary"]["auc"] - rc["binary"]["auc"])
        if s == C.SEED:
            primary = {"gat": rg, "gcn": rc, "mlp": rm, "xgb": rx,
                       "split": {"train": len(tr), "val": len(va), "test": len(te)}}
            primary_models = {"gat": gat, "Tn": Tn, "tr": tr, "te": te, "va": va}

    def agg(lst, key):
        v = np.array([x[key] for x in lst])
        return {"mean": float(v.mean()), "sd": float(v.std()), "values": v.tolist()}

    R["performance"] = {
        "primary_seed": C.SEED,
        "seed_summary": {m: {"macro_r2": agg(per_seed[m], "macro_r2"),
                             "auc": agg(per_seed[m], "auc")} for m in per_seed},
        "per_head_r2_seedmean": {nm: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                                 for nm, v in per_head_acc.items() if v},
        "margins": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                        "values": v} for k, v in margins.items()},
        "primary_detail": primary,
    }

    print("[5/8] explainability (seed 42) ...")
    gat = primary_models["gat"]; Tn = primary_models["Tn"]
    tr, te = primary_models["tr"], primary_models["te"]
    rng = np.random.default_rng(C.SEED)
    test_obs = [i for i in te if not np.isnan(co.y_bin[i])]
    subset = list(rng.choice(test_obs, min(C.EXPLAIN_SUBSET, len(test_obs)), replace=False))
    gsub = list(rng.choice(test_obs, min(C.GNNEXPLAINER_SUBSET, len(test_obs)), replace=False))
    try:
        shap_mat = explain.shap_attributions(gat, Tn, tr, subset)
        lime_mat = explain.lime_attributions(gat, Tn, tr, subset, co.feat_names)
        gnn_glob = explain.gnnexplainer_global(gat, Tn, gsub)
        R["explainability"] = explain.cross_method(shap_mat, lime_mat, gnn_glob, co.feat_names)
        R["explainability"]["attention_homophily"] = explain.attention_homophily(
            gat, Tn, co.strata, te)
    except Exception as e:
        R["explainability"] = {"error": repr(e)}

    print("[6/8] fairness audit + mitigation (seed 42) ...")
    _, prob = train.predict(gat, Tn)
    R["fairness"] = {"unmitigated": fairness.audit_all(
        prob, co.y_bin, Tn.ybin_mask.numpy(), co.strata, te)}
    try:
        mit = fairness.train_binary_reweighed(co, Tn, tr, primary_models["va"],
                                              best_cfg, C.SEED, "education")
        _, prob_m = train.predict(mit, Tn)
        R["fairness"]["reweighed"] = fairness.audit_all(
            prob_m, co.y_bin, Tn.ybin_mask.numpy(), co.strata, te)
        rmit = train.evaluate(mit, Tn, te, n_boot=0)
        R["fairness"]["reweighed_auc"] = rmit["binary"]["auc"]
    except Exception as e:
        R["fairness"]["mitigation_error"] = repr(e)

    # checkpoint core results BEFORE the (slower, supplementary) sensitivity stage
    C.OUTPUTS.mkdir(exist_ok=True)
    with open(C.RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, default=jdefault)
    print("    checkpoint written (core results) ->", C.RESULTS_JSON)

    print("[7/8] sensitivity analyses (seed 42) ...")
    R["sensitivity"] = {}
    try:
        R["sensitivity"]["graph_resolution_k"] = sensitivity.graph_resolution(co, S, best_cfg, C.ALL_SEEDS)
        R["sensitivity"]["threshold_variant"] = sensitivity.threshold_variant(co, S, best_cfg, C.ALL_SEEDS)
        R["sensitivity"]["cosine_variant"] = sensitivity.cosine_variant(co, best_cfg, C.ALL_SEEDS, best_k)
        R["sensitivity"]["feature_ablation"] = sensitivity.feature_ablation(co, S, best_cfg, C.ALL_SEEDS, best_k)
        R["sensitivity"]["_note"] = (
            "All trainings aggregated over ALL_SEEDS (mean+/-SD), matched to the "
            f"headline. Gower-threshold graphs skipped where edges > {C.MAX_SENS_EDGES} "
            "(uniformly dense; motivates symmetric k-NN sparsification).")
    except Exception as e:
        R["sensitivity"]["error"] = repr(e)

    R["meta"]["runtime_sec"] = round(time.time() - t0, 1)
    print("[8/8] writing results.json ...")
    C.OUTPUTS.mkdir(exist_ok=True)
    with open(C.RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, default=jdefault)
    print(f"DONE in {R['meta']['runtime_sec']}s -> {C.RESULTS_JSON}")
    print("GAT macro-R2 (seed-mean): %.4f +/- %.4f | AUC: %.3f +/- %.3f" % (
        R["performance"]["seed_summary"]["gat"]["macro_r2"]["mean"],
        R["performance"]["seed_summary"]["gat"]["macro_r2"]["sd"],
        R["performance"]["seed_summary"]["gat"]["auc"]["mean"],
        R["performance"]["seed_summary"]["gat"]["auc"]["sd"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(quick=args.quick)
