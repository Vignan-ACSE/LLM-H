"""
Guarantee validation for the conformal recall threshold (C4-C8).

C1 (src/calibration/conformal.py) turns calibration scores into a threshold
tau. This module measures what that threshold does on data it was not set
on, and what the theory says it should do:

    threshold_outcome       recall, flag rate, ... of one tau on one set
    calibrate_and_evaluate  set tau on one set, measure it on another
    expected_recall         mean recall over calibration draws, 1 - k / (n + 1)
    recall_spread           SD of the recall one calibration set gives
    summarize_trials        repeated runs -> mean, spread, share below target

Convention, as in C1: higher score = more likely hallucination, and an
answer is flagged when score >= tau.

Theory, for n calibration positives and k = floor(alpha * (n + 1)): the
recall that one calibration set gives on fresh positives follows
Beta(n + 1 - k, k) when scores have no ties, so its mean is 1 - k / (n + 1).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.calibration.conformal import _FLOOR_TOL, calibrate_recall_threshold


def _k(n_pos: int, alpha: float) -> int:
    return int(math.floor(alpha * (n_pos + 1) + _FLOOR_TOL))


def expected_recall(n_pos: int, alpha: float) -> float:
    """Mean recall over calibration draws; 1 when n_pos is too small (k = 0)."""
    k = _k(n_pos, alpha)
    return 1.0 if k == 0 else 1.0 - k / (n_pos + 1)


def recall_spread(n_pos: int, alpha: float) -> float:
    """SD of the recall a single calibration set gives (0 when k = 0)."""
    k = _k(n_pos, alpha)
    if k == 0:
        return 0.0
    a, b = n_pos + 1 - k, k
    return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))


def threshold_outcome(scores, labels, tau: float) -> dict:
    """
    What flagging ``score >= tau`` does on one labelled set.

    recall                       share of hallucinations flagged
    flag_rate                    share of all answers flagged (escalated)
    precision                    share of flagged answers that are hallucinations
    accepted_hallucination_rate  share of accepted (unflagged) answers that are
                                 hallucinations: the risk that remains
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    flagged = scores >= tau
    positive = labels == 1
    accepted = ~flagged
    return {
        "n": int(len(labels)),
        "n_pos": int(positive.sum()),
        "recall": float(flagged[positive].mean()) if positive.any() else float("nan"),
        "flag_rate": float(flagged.mean()) if len(labels) else float("nan"),
        "precision": float(labels[flagged].mean()) if flagged.any() else float("nan"),
        "accepted_hallucination_rate": (
            float(labels[accepted].mean()) if accepted.any() else float("nan")
        ),
    }


def calibrate_and_evaluate(cal_scores, cal_labels, eval_scores, eval_labels, alpha: float) -> dict:
    """Set tau on the calibration set (C1), then measure it on the eval set."""
    tau = calibrate_recall_threshold(cal_scores, cal_labels, alpha)
    n_cal_pos = int((np.asarray(cal_labels) == 1).sum())
    out = {
        "target_recall": 1.0 - alpha,
        "tau": tau,
        "n_cal": int(len(cal_labels)),
        "n_cal_pos": n_cal_pos,
        "expected_recall": expected_recall(n_cal_pos, alpha),
    }
    out.update(threshold_outcome(eval_scores, eval_labels, tau))
    out["coverage_gap"] = out["recall"] - out["target_recall"]
    return out


def summarize_trials(trials: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """
    Repeated calibrate/evaluate runs grouped by ``by``.

    frac_below_target is expected to be well above 0: the guarantee is on
    the mean over calibration sets, not on every set. A mean_recall clearly
    below target_recall is the real warning sign.
    """
    frame = trials.assign(below=trials["recall"] < trials["target_recall"] - 1e-12)
    target = {} if "target_recall" in by else {"target_recall": ("target_recall", "first")}
    summary = frame.groupby(by, sort=False).agg(
        trials=("recall", "size"),
        **target,
        mean_recall=("recall", "mean"),
        sd_recall=("recall", "std"),
        min_recall=("recall", "min"),
        frac_below_target=("below", "mean"),
        mean_flag_rate=("flag_rate", "mean"),
        mean_accepted_hallucination_rate=("accepted_hallucination_rate", "mean"),
        mean_n_cal_pos=("n_cal_pos", "mean"),
    ).reset_index()
    summary["mean_coverage_gap"] = summary["mean_recall"] - summary["target_recall"]
    return summary
