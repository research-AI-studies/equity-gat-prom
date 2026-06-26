"""Smoke tests on example data only (no real patient records).

These exercise the data-loading and graph-construction stages end-to-end on a
small example cohort. Heavy model training (PyTorch Geometric) is
intentionally not invoked here so the test runs quickly in CI.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def example_cohort(tmp_path_factory):
    import sys
    sys.path.insert(0, str(REPO))
    from data.example.generate_example import generate

    out = tmp_path_factory.mktemp("syn") / "cohort.xlsx"
    df = generate(n=120, seed=42)
    df.to_excel(out, index=False)
    os.environ["PAPER5_RAW"] = str(out)
    return out


def test_data_loads(example_cohort):
    from paper5_pipeline import data
    co = data.load_cohort()
    assert co.X.shape[0] == len(co.raw)
    assert co.X.shape[0] > 0
    assert co.X.shape[1] == len(co.feat_names)
    assert co.Y.shape[1] == 8
    assert set(co.strata.columns) == {"education", "partnership", "diagnosis"}
    assert not np.isnan(co.X).any()


def test_graph_builds(example_cohort):
    from paper5_pipeline import data, graph
    co = data.load_cohort()
    S = graph.gower_matrix(co.raw_features)
    assert S.shape == (len(co.raw), len(co.raw))
    A = graph.knn_adjacency(S, k=10)
    assert A.shape == S.shape
    assert (A == A.T).all()  # symmetric
    ei, ew = graph.build_edges(A, S)
    assert ei.shape[0] == 2
    assert ei.shape[1] == ew.shape[0]
