"""
Tests for src/eval/coverage.py: the measurements C4-C8 report, checked on
synthetic scores where the right answer is known.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.eval.coverage import (
    calibrate_and_evaluate,
    expected_recall,
    recall_spread,
    summarize_trials,
    threshold_outcome,
)


def test_threshold_outcome_counts():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 0, 1, 0])
    out = threshold_outcome(scores, labels, tau=0.5)
    assert out["recall"] == 0.5
    assert out["flag_rate"] == 0.5
    assert out["precision"] == 0.5
    assert out["accepted_hallucination_rate"] == 0.5


def test_flag_everything_when_tau_is_minus_infinity():
    out = threshold_outcome([0.1, 0.2], [1, 0], tau=-np.inf)
    assert out["recall"] == 1.0
    assert out["flag_rate"] == 1.0
    assert math.isnan(out["accepted_hallucination_rate"])


def test_expected_recall_and_spread_follow_the_formula():
    assert expected_recall(19, 0.05) == pytest.approx(0.95)
    assert expected_recall(50, 0.05) == pytest.approx(1 - 2 / 51)
    assert expected_recall(10, 0.05) == 1.0
    assert recall_spread(10, 0.05) == 0.0
    # Beta(n + 1 - k, k) with n = 1000, k = 50
    assert recall_spread(1000, 0.05) == pytest.approx(0.0069, abs=2e-4)


def test_recall_spread_matches_simulation():
    rng = np.random.default_rng(0)
    n, alpha = 200, 0.05
    recalls = []
    for _ in range(3000):
        cal = rng.normal(size=n)
        fresh = rng.normal(size=2000)
        out = calibrate_and_evaluate(cal, np.ones(n), fresh, np.ones(2000), alpha)
        recalls.append(out["recall"])
    assert np.mean(recalls) == pytest.approx(expected_recall(n, alpha), abs=0.003)
    assert np.std(recalls) == pytest.approx(recall_spread(n, alpha), rel=0.1)


def test_calibrate_and_evaluate_records_the_setup():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 400)
    scores = labels + rng.normal(0, 1, 400)
    out = calibrate_and_evaluate(scores[:200], labels[:200], scores[200:], labels[200:], 0.1)
    assert out["target_recall"] == pytest.approx(0.9)
    assert out["n_cal"] == 200
    assert out["n_cal_pos"] == int(labels[:200].sum())
    assert out["coverage_gap"] == pytest.approx(out["recall"] - 0.9)


def test_summarize_trials():
    trials = pd.DataFrame({
        "score": ["a"] * 4,
        "target_recall": [0.9] * 4,
        "recall": [0.85, 0.95, 0.92, 0.88],
        "flag_rate": [0.5] * 4,
        "accepted_hallucination_rate": [0.1] * 4,
        "n_cal_pos": [100] * 4,
    })
    row = summarize_trials(trials, ["score"]).iloc[0]
    assert row["trials"] == 4
    assert row["mean_recall"] == pytest.approx(0.9)
    assert row["frac_below_target"] == pytest.approx(0.5)
    assert row["mean_coverage_gap"] == pytest.approx(0.0)


def test_summarize_trials_can_group_by_target():
    trials = pd.DataFrame({
        "score": ["a"] * 4,
        "target_recall": [0.9, 0.9, 0.95, 0.95],
        "recall": [0.85, 0.95, 0.96, 0.94],
        "flag_rate": [0.5] * 4,
        "accepted_hallucination_rate": [0.1] * 4,
        "n_cal_pos": [100] * 4,
    })
    summary = summarize_trials(trials, ["score", "target_recall"])
    assert summary["target_recall"].tolist() == [0.9, 0.95]
    assert summary["mean_coverage_gap"].tolist() == pytest.approx([0.0, 0.0])
