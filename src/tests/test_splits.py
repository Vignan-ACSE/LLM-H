"""C3 - tests for src/calibration/splits.py (synthetic data only).

Frames mimic the real one: four subsets, very uneven label balance, and a
norm_key that is the normalized text of the example (as in src/data/loaders.py),
made of random words so unrelated texts are never similar while "twins" (one
word changed) are. Some rows share a norm_key, i.e. have identical text.

If your other tests import differently, adjust the import below to match.
"""

import numpy as np
import pandas as pd
import pytest

from src.calibration.splits import (
    PART_NAMES,
    SplitLeakageError,
    assert_no_leakage,
    find_near_duplicates,
    make_splits,
    split_summary,
)

SUBSETS = {"qa": 0.80, "dialogue": 0.98, "summarization": 0.82, "truthfulqa": 0.82}
VOCAB = [f"w{i}" for i in range(6000)]


def _prompt(rng, n_words=30):
    return " ".join(rng.choice(VOCAB, n_words, replace=False))


def make_frame(n_per_subset=600, seed=0, share_key_frac=0.1):
    """Synthetic frame with the columns make_splits expects."""
    rng = np.random.default_rng(seed)
    rows = []
    for subset, pos_rate in SUBSETS.items():
        n_pos = int(round(n_per_subset * pos_rate))
        labels = np.array([1] * n_pos + [0] * (n_per_subset - n_pos))
        rng.shuffle(labels)
        for label in labels:
            rows.append({"subset": subset, "hallucinated": int(label),
                         "prompt": _prompt(rng)})
    frame = pd.DataFrame(rows)
    frame.insert(0, "uid", [f"u{i}" for i in range(len(frame))])
    # a share of rows repeat an earlier row's text exactly (groups of 2-3 rows)
    texts = frame["prompt"].tolist()
    for i in rng.choice(len(frame) - 1, int(share_key_frac * len(frame)), replace=False):
        texts[i + 1] = texts[i]
    frame["prompt"] = texts
    frame["norm_key"] = texts  # in loaders.py norm_key is the normalized text
    return frame


def add_twins(frame, n_twins, seed=1):
    """Append near-copies: same text with one word changed, new uid."""
    rng = np.random.default_rng(seed)
    originals = rng.choice(len(frame), n_twins, replace=False)
    twins = frame.iloc[originals].copy()
    new_prompts = []
    for text in twins["prompt"]:
        words = text.split()
        words[rng.integers(len(words))] = "changedword"
        new_prompts.append(" ".join(words))
    twins["prompt"] = new_prompts
    twins["norm_key"] = new_prompts
    twins["uid"] = [f"twin{i}" for i in range(n_twins)]
    pairs = list(zip(frame["uid"].iloc[originals], twins["uid"]))
    return pd.concat([frame, twins], ignore_index=True), pairs


# ---------- the split itself ----------

def test_every_row_gets_exactly_one_part_and_no_leakage():
    frame = make_frame()
    out, info = make_splits(frame)
    assert len(out) == len(frame)
    assert set(out["split"]) == set(PART_NAMES)
    assert out["uid"].is_unique
    assert_no_leakage(out)  # uid, norm_key, group_id, near-duplicates
    assert sum(info["rows_per_split"].values()) == len(frame)


def test_shared_norm_key_rows_stay_together():
    # Checked with near-duplicate merging both off and on: with it off, only the
    # norm_key grouping keeps identical rows together.
    frame = make_frame(share_key_frac=0.3)
    for near_dup in (False, True):
        out, _ = make_splits(frame, near_dup=near_dup)
        assert out.groupby("norm_key")["split"].nunique().max() == 1, near_dup


def test_sizes_and_label_balance_follow_the_request():
    frame = make_frame()
    out, _ = make_splits(frame, fractions=(0.4, 0.3, 0.3))
    for subset in SUBSETS:
        sub = out[out["subset"] == subset]
        overall_pos = sub["hallucinated"].mean()
        for part, want in zip(PART_NAMES, (0.4, 0.3, 0.3)):
            in_part = sub[sub["split"] == part]
            assert abs(len(in_part) / len(sub) - want) < 0.04, (subset, part)
            # stratified: each part keeps about the subset's share of positives
            assert abs(in_part["hallucinated"].mean() - overall_pos) < 0.06, (subset, part)


def test_same_seed_same_split_different_seed_different_split():
    frame = make_frame()
    a, _ = make_splits(frame, seed=7)
    b, _ = make_splits(frame, seed=7)
    c, _ = make_splits(frame, seed=8)
    assert (a["split"].to_numpy() == b["split"].to_numpy()).all()
    assert (a["split"].to_numpy() != c["split"].to_numpy()).any()


def test_row_order_of_input_does_not_leak_into_result_columns():
    frame = make_frame()
    out, _ = make_splits(frame)
    # extra columns of the input are carried through unchanged
    assert (out["uid"].to_numpy() == frame["uid"].to_numpy()).all()
    assert (out["prompt"].to_numpy() == frame["prompt"].to_numpy()).all()


def test_zero_fit_fraction_gives_two_parts():
    # e.g. C4 (no fit needed) or C5 (split the rows the classifier never saw)
    frame = make_frame()
    out, info = make_splits(frame, fractions=(0.0, 0.5, 0.5))
    assert (out["split"] == "fit").sum() == 0
    assert info["rows_per_split"]["fit"] == 0
    assert abs((out["split"] == "calibrate").mean() - 0.5) < 0.03
    assert_no_leakage(out)


def test_info_records_what_was_done():
    out, info = make_splits(make_frame(), seed=3, sim_threshold=0.85)
    assert info["seed"] == 3 and info["sim_threshold"] == 0.85
    assert info["n_rows"] == len(out)
    assert info["n_groups_after_merge"] <= info["n_groups_exact"]
    assert info["positives_per_split"]["calibrate"] > 0


def test_split_summary_counts_add_up():
    out, _ = make_splits(make_frame())
    table = split_summary(out)
    assert table["n"].sum() == len(out)
    assert list(table.columns) == ["n_label0", "n_label1", "n", "frac_label1"]


# ---------- near-duplicates ----------

def test_find_near_duplicates_basics():
    rng = np.random.default_rng(0)
    base = _prompt(rng)
    words = base.split()
    words[5] = "changedword"
    texts = [base, base, " ".join(words), _prompt(rng), _prompt(rng)]
    pairs = {tuple(p) for p in find_near_duplicates(texts, threshold=0.9)}
    assert pairs == {(0, 1), (0, 2), (1, 2)}  # the three copies, nothing else
    assert len(find_near_duplicates(texts, threshold=0.999)) == 1  # only the exact copy


def test_near_duplicate_twins_are_kept_in_the_same_part():
    frame, pairs = add_twins(make_frame(), n_twins=200)
    out, info = make_splits(frame)
    part = out.set_index("uid")["split"]
    assert all(part[a] == part[b] for a, b in pairs)
    assert info["n_near_dup_pairs"] >= 200
    assert_no_leakage(out)


def test_without_merging_twins_leak_and_the_check_catches_it():
    frame, pairs = add_twins(make_frame(), n_twins=200)
    out, _ = make_splits(frame, near_dup=False)  # norm_keys differ, so no exact match
    part = out.set_index("uid")["split"]
    assert any(part[a] != part[b] for a, b in pairs)
    with pytest.raises(SplitLeakageError):
        assert_no_leakage(out)


# ---------- the leakage check itself ----------

def test_check_detects_a_shared_norm_key():
    out, _ = make_splits(make_frame(share_key_frac=0.3))
    key = out.groupby("norm_key").size().loc[lambda s: s > 1].index[0]
    rows = out.index[out["norm_key"] == key]
    other = next(p for p in PART_NAMES if p != out.loc[rows[0], "split"])
    out.loc[rows[0], "split"] = other
    with pytest.raises(SplitLeakageError):
        assert_no_leakage(out, check_near_dup=False)


def test_check_detects_a_duplicated_uid():
    out, _ = make_splits(make_frame())
    dup = pd.concat([out, out.iloc[[0]]], ignore_index=True)
    with pytest.raises(SplitLeakageError):
        assert_no_leakage(dup, check_near_dup=False)


def test_check_detects_a_missing_assignment():
    out, _ = make_splits(make_frame())
    out.loc[0, "split"] = np.nan
    with pytest.raises(SplitLeakageError):
        assert_no_leakage(out, check_near_dup=False)


# ---------- input validation ----------

def test_rejects_missing_label():
    frame = make_frame()
    frame["hallucinated"] = frame["hallucinated"].astype(float)
    frame.loc[3, "hallucinated"] = np.nan
    with pytest.raises(ValueError):
        make_splits(frame)


def test_rejects_non_binary_label():
    frame = make_frame()
    frame.loc[3, "hallucinated"] = 2
    with pytest.raises(ValueError):
        make_splits(frame)


def test_rejects_bad_fractions():
    frame = make_frame()
    for bad in [(0.5, 0.5, 0.5), (0.33, 0.33, 0.34), (-0.1, 0.6, 0.5), (0.5, 0.5)]:
        with pytest.raises(ValueError):
            make_splits(frame, fractions=bad)


def test_rejects_duplicate_uid_and_missing_column():
    frame = make_frame()
    dup = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        make_splits(dup)
    with pytest.raises(ValueError):
        make_splits(frame.drop(columns=["norm_key"]))


def test_rejects_a_giant_near_duplicate_cluster():
    # If the text column is the same for every row, everything chains into one
    # group and no split can be made.
    frame = make_frame(n_per_subset=100)
    frame["norm_key"] = "summarize the following document"
    with pytest.raises(ValueError):
        make_splits(frame)


def test_rejects_missing_near_duplicate_text():
    frame = make_frame()
    frame.loc[0, "norm_key"] = np.nan
    with pytest.raises(ValueError, match="norm_key"):
        make_splits(frame)


def test_summarization_style_constant_prompt_is_fine_on_norm_key():
    # Real data: the summarization prompt is the same fixed sentence in every
    # row and the document lives in norm_key. Comparing norm_key works; comparing
    # the constant prompt column is the trap and must fail loudly.
    frame = make_frame(n_per_subset=100)
    frame["prompt"] = "Summarize the following document."
    out, _ = make_splits(frame)  # default text_col = norm_key
    assert_no_leakage(out)
    with pytest.raises(ValueError):
        make_splits(frame, text_col="prompt")


def test_near_duplicate_detector_rejects_invalid_settings():
    texts = ["a question about cats", "a question about dogs"]
    for kwargs in (
        {"threshold": -0.1},
        {"threshold": 1.1},
        {"chunk_size": 0},
        {"max_df": 0.0},
        {"max_df": 1.1},
    ):
        with pytest.raises(ValueError):
            find_near_duplicates(texts, **kwargs)
