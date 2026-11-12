"""C2 - synthetic test harness for src/calibration/conformal.py.

Workbook C2: scores with a known, controllable relationship to labels; assert
that empirical coverage (recall) converges to the target as calibration size
grows. This must pass before any calibration touches real data.

The generator has two dials:
  pos_rate   - share of hallucinated (label 1) examples
  separation - how far the hallucinated scores sit above the others.
               0 = the score carries no information about the label.
               Scores are N(0, 1) for label 0 and N(separation, 1) for label 1,
               so AUROC = Phi(separation / sqrt(2)):
                 weak   0.5 -> AUROC 0.64
                 medium 1.5 -> AUROC 0.86
                 strong 3.0 -> AUROC 0.98
               (0.0 -> 0.50 is the useless score used in a separate test.)

Every test calibrates for a 95% recall target (alpha = 0.05) on one set and
measures recall on a fresh set drawn the same way, repeated many times.
"""

import numpy as np
import pytest

from src.calibration.conformal import calibrate_recall_threshold

ALPHA = 0.05
TARGET = 1 - ALPHA
TOL = 0.004  # allowance for Monte-Carlo noise in averaged recall


def make_labeled_scores(rng, n, pos_rate, separation):
    """n examples with exactly round(n * pos_rate) positives, shuffled."""
    n_pos = int(round(n * pos_rate))
    labels = np.concatenate([np.ones(n_pos, dtype=int),
                             np.zeros(n - n_pos, dtype=int)])
    scores = rng.normal(0.0, 1.0, n) + separation * labels
    order = rng.permutation(n)
    return scores[order], labels[order]


def run_trials(n_cal, pos_rate, separation, trials=600, n_test=3000, seed=0):
    """Calibrate on n_cal rows, test on fresh rows; return the mean recall and
    mean flag rate (share of ALL answers flagged) over many repetitions."""
    rng = np.random.default_rng(seed)
    recalls = np.empty(trials)
    flag_rates = np.empty(trials)
    for t in range(trials):
        cal_s, cal_y = make_labeled_scores(rng, n_cal, pos_rate, separation)
        tau = calibrate_recall_threshold(cal_s, cal_y, ALPHA)
        test_s, test_y = make_labeled_scores(rng, n_test, pos_rate, separation)
        recalls[t] = np.mean(test_s[test_y == 1] >= tau)
        flag_rates[t] = np.mean(test_s >= tau)
    return recalls.mean(), flag_rates.mean()


# ---------- the generator itself ----------

def test_generator_dials_do_what_they_say():
    rng = np.random.default_rng(0)
    scores, labels = make_labeled_scores(rng, 20000, pos_rate=0.89, separation=2.0)
    assert labels.sum() == 17800                      # exact share of positives
    gap = scores[labels == 1].mean() - scores[labels == 0].mean()
    assert abs(gap - 2.0) < 0.1                       # separation is the mean gap
    scores0, labels0 = make_labeled_scores(rng, 20000, 0.5, separation=0.0)
    gap0 = scores0[labels0 == 1].mean() - scores0[labels0 == 0].mean()
    assert abs(gap0) < 0.1                            # separation 0 = no signal


# ---------- coverage converges to the target as calibration size grows ----------

@pytest.mark.parametrize("separation", [0.5, 1.5, 3.0], ids=["weak", "medium", "strong"])
def test_recall_reaches_target_and_converges(separation):
    # Calibration sizes 20 -> 2000 rows at 50% positives (10 -> 1000 positives).
    sizes = [20, 100, 500, 2000]
    overshoot = []
    for n_cal in sizes:
        recall, _ = run_trials(n_cal, pos_rate=0.5, separation=separation)
        assert recall >= TARGET - TOL, (separation, n_cal, recall)
        overshoot.append(recall - TARGET)
    # Small sets are conservative (flag more than needed); the excess shrinks
    # as the calibration set grows, and is tiny at 2000 rows.
    for earlier, later in zip(overshoot, overshoot[1:]):
        assert later <= earlier + 0.0005, (separation, overshoot)
    assert overshoot[-1] < 0.002, (separation, overshoot)


# ---------- the guarantee does not depend on how good the score is ----------

def test_useless_score_still_reaches_target_but_flags_almost_everything():
    recall, flag_rate = run_trials(2000, pos_rate=0.89, separation=0.0)
    assert recall >= TARGET - TOL
    # With no signal, flagging 95% of hallucinations means flagging ~95% of all
    # answers: the recall guarantee is met but nothing is saved.
    assert abs(flag_rate - recall) < 0.01
    assert flag_rate > 0.9


def test_good_score_flags_fewer_answers_at_the_same_recall():
    useless_recall, useless_flags = run_trials(2000, 0.89, separation=0.0)
    good_recall, good_flags = run_trials(2000, 0.89, separation=3.0)
    assert good_recall >= TARGET - TOL
    assert useless_recall >= TARGET - TOL
    assert useless_flags - good_flags > 0.05          # same recall, fewer flags


# ---------- the guarantee does not depend on the share of hallucinations ----------

@pytest.mark.parametrize("pos_rate", [0.89, 0.5, 0.3, 0.1])
def test_recall_holds_for_any_hallucination_rate(pos_rate):
    n_cal = 1000
    n_pos = int(round(n_cal * pos_rate))
    recall, _ = run_trials(n_cal, pos_rate=pos_rate, separation=1.5)
    assert recall >= TARGET - TOL, (pos_rate, recall)
    # not wasteful either: the excess is at most about one rank in n_pos + 1
    assert recall <= TARGET + 1 / (n_pos + 1) + TOL, (pos_rate, recall)
