"""
C3 - splitting utilities for conformal calibration.

Splits a labelled frame into three parts:

    fit        data a detector may be trained on (e.g. the B8 classifier)
    calibrate  data used ONLY to set the threshold tau (C1)
    eval       data used ONLY to measure recall / flag rate afterwards

The same row must never appear in two parts, and neither may a near-copy of
it: the recall guarantee needs the calibration scores to be exchangeable with
fresh data, and a detector that has seen a row (or its twin) scores it
optimistically.

How the split is made
---------------------
1. Rows that share a ``norm_key`` form one group (exact duplicates).
2. Rows whose texts are near-duplicates (TF-IDF cosine similarity >=
   ``sim_threshold``) are merged into the same group, so a group never
   straddles two parts. ``norm_key`` alone only catches exact matches.
   The text compared is the ``norm_key`` column itself: in src/data/loaders.py
   it is the normalized text that identifies a prompt (QA question, dialogue
   history, summarization DOCUMENT, TruthfulQA question). The ``prompt``
   column is not used because for summarization it is the same fixed
   instruction in every row.
3. Groups are assigned to parts with ``StratifiedGroupKFold`` (same idea as
   ``grouped_folds`` in src/signals/probes.py), stratified on subset x label
   so every part keeps a similar share of positives in every subset. Recall is
   measured on positives, so each part needs enough of them.
4. ``assert_no_leakage`` re-checks the result independently.

Fractions are realised by giving whole folds to parts, so they must be
multiples of ``1 / n_folds`` (5% steps with the default 20 folds). A part
with fraction 0 is simply empty, e.g. ``(0, 0.5, 0.5)`` for a detector that
needs no fit part (C4), or for splitting a held-out set into calibrate/eval
(C5: pass only the rows the classifier never trained on).

Rows with a missing label (e.g. refusals) must be dropped before splitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PART_NAMES = ("fit", "calibrate", "eval")


class SplitLeakageError(AssertionError):
    """Raised when two parts share a row, a norm_key, a group or a near-copy."""



# near-duplicate detection
def find_near_duplicates(texts, threshold: float = 0.9, chunk_size: int = 1000,
                         max_df: float = 0.5):
    """Return an (m, 2) int array of index pairs (i < j) whose TF-IDF cosine
    similarity is >= ``threshold``. Exact duplicates (similarity 1) are included.

    Words that occur in more than ``max_df`` of all texts (template wording,
    "the", ...) are ignored: they carry no information about whether two texts
    are copies and they make the pairwise search much slower. The cut-off is
    never below 20 texts, so small inputs are not pruned, and if pruning would
    leave no words at all (e.g. every text is identical) the words are kept.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not 0.0 < max_df <= 1.0:
        raise ValueError("max_df must be in (0, 1]")

    texts = [str(t) for t in texts]
    n = len(texts)
    if n < 2:
        return np.empty((0, 2), dtype=int)
    try:
        cutoff = max(int(max_df * n), 20)  # absolute document count
        matrix = TfidfVectorizer(lowercase=True, max_df=cutoff).fit_transform(texts)
    except ValueError:  # pruning left an empty vocabulary
        matrix = TfidfVectorizer(lowercase=True).fit_transform(texts)
    matrix = matrix.tocsr()
    matrix_t = matrix.T.tocsc()
    found = []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sims = (matrix[start:stop] @ matrix_t).tocoo()
        keep = (sims.data >= threshold) & (sims.col > sims.row + start)
        if keep.any():
            found.append(np.column_stack([sims.row[keep] + start, sims.col[keep]]))
    if not found:
        return np.empty((0, 2), dtype=int)
    return np.vstack(found)


class _UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# the split
def _check_frame(frame, uid_col, group_col, stratify_col, label_col, text_col,
                 need_text):
    needed = [uid_col, group_col, stratify_col, label_col]
    if need_text:
        needed.append(text_col)
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")
    if frame[uid_col].duplicated().any():
        raise ValueError(f"{uid_col!r} must be unique")
    for col in (group_col, stratify_col, label_col):
        if frame[col].isna().any():
            raise ValueError(
                f"{col!r} has missing values; drop unlabeled rows (e.g. "
                f"refusals) before splitting")
    if need_text and frame[text_col].isna().any():
        raise ValueError(f"{text_col!r} has missing values")
    if not np.isin(frame[label_col].to_numpy(), (0, 1)).all():
        raise ValueError(f"{label_col!r} must contain only 0 and 1")


def make_splits(
    frame: pd.DataFrame,
    fractions=(0.4, 0.3, 0.3),
    *,
    seed: int = 42,
    n_folds: int = 20,
    uid_col: str = "uid",
    group_col: str = "norm_key",
    stratify_col: str = "subset",
    label_col: str = "hallucinated",
    text_col: str = "norm_key",
    near_dup: bool = True,
    sim_threshold: float = 0.9,
    max_group_frac: float = 0.05,
):
    """Split ``frame`` into fit / calibrate / eval.

    Returns ``(split_frame, info)``. ``split_frame`` is a copy of ``frame``
    with two added columns: ``group_id`` (norm_key groups merged across
    near-duplicates) and ``split`` (one of "fit", "calibrate", "eval").
    ``info`` is a dict of settings and counts, suitable for logging.
    """
    if (
        len(fractions) != 3
        or not np.isfinite(np.asarray(fractions, dtype=float)).all()
        or min(fractions) < 0
        or abs(sum(fractions) - 1) > 1e-9
    ):
        raise ValueError("fractions must be three non-negative numbers summing to 1")
    if int(n_folds) != n_folds or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least 2")
    if not 0.0 < max_group_frac <= 1.0:
        raise ValueError("max_group_frac must be in (0, 1]")
    fold_counts = np.array(fractions, dtype=float) * n_folds
    if not np.allclose(fold_counts, np.round(fold_counts), atol=1e-6):
        raise ValueError(
            f"fractions must be multiples of 1/{n_folds} ({100 / n_folds:g}%); "
            f"got {tuple(fractions)}")
    fold_counts = np.round(fold_counts).astype(int)

    _check_frame(frame, uid_col, group_col, stratify_col, label_col, text_col,
                 need_text=near_dup)
    if len(frame) < n_folds:
        raise ValueError(f"need at least n_folds={n_folds} rows, got {len(frame)}")

    out = frame.reset_index(drop=True).copy()
    out[label_col] = out[label_col].astype(int)

    # groups: exact duplicates share a norm_key ...
    codes, _ = pd.factorize(out[group_col])
    n_groups_exact = int(codes.max()) + 1
    union = _UnionFind(n_groups_exact)
    n_pairs = 0
    
    # and near-duplicate prompts are merged into the same group
    if near_dup:
        pairs = find_near_duplicates(out[text_col], threshold=sim_threshold)
        n_pairs = len(pairs)
        for i, j in pairs:
            union.union(codes[i], codes[j])
    roots = np.array([union.find(c) for c in codes])
    out["group_id"] = ["g" + str(r) for r in roots]
    sizes = out["group_id"].value_counts()
    if sizes.iloc[0] > max_group_frac * len(out):
        raise ValueError(
            f"largest group has {sizes.iloc[0]} rows "
            f"({sizes.iloc[0] / len(out):.1%} of the data, limit "
            f"{max_group_frac:.0%}). Near-duplicate merging has probably "
            f"chained unrelated rows together (or {text_col!r} is not the text "
            f"that distinguishes examples). Raise sim_threshold or check the column.")

    # assign whole groups to folds, stratified on subset x label
    from sklearn.model_selection import StratifiedGroupKFold

    strat = out[stratify_col].astype(str) + "|" + out[label_col].astype(str)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of_row = np.full(len(out), -1)
    for fold, (_, test_idx) in enumerate(
            splitter.split(out, strat, groups=out["group_id"])):
        fold_of_row[test_idx] = fold
    part_of_fold = np.repeat(np.array(PART_NAMES), fold_counts)
    out["split"] = part_of_fold[fold_of_row]

    # Enforce row and exact-key separation before returning the split.
    # Near-duplicate separation is guaranteed by assigning each merged group
    # wholly to a fold; assert_no_leakage can independently recheck it.
    assert_no_leakage(
        out,
        uid_col=uid_col,
        group_col=group_col,
        text_col=text_col,
        check_near_dup=False,
    )

    # C1 needs positives in every part that is meant to exist
    for name, frac in zip(PART_NAMES, fractions):
        if frac > 0:
            positives = int(((out["split"] == name) & (out[label_col] == 1)).sum())
            if positives == 0:
                raise ValueError(f"part {name!r} received no positive examples")

    info = {
        "seed": seed, "fractions": tuple(fractions), "n_folds": n_folds,
        "near_dup": near_dup, "sim_threshold": sim_threshold, "text_col": text_col,
        "n_rows": len(out), "n_groups_exact": n_groups_exact,
        "n_groups_after_merge": int(out["group_id"].nunique()),
        "n_near_dup_pairs": int(n_pairs), "largest_group": int(sizes.iloc[0]),
        "rows_per_split": out["split"].value_counts().reindex(PART_NAMES, fill_value=0).to_dict(),
        "positives_per_split": out[out[label_col] == 1]["split"]
            .value_counts().reindex(PART_NAMES, fill_value=0).to_dict(),
    }
    return out, info



# leakage checks and reporting
def assert_no_leakage(
    frame: pd.DataFrame,
    *,
    split_col: str = "split",
    uid_col: str = "uid",
    group_col: str = "norm_key",
    text_col: str = "norm_key",
    check_near_dup: bool = True,
    sim_threshold: float = 0.9,
) -> None:
    """Raise ``SplitLeakageError`` unless the parts are fully separate.

    Checks, independently of how the split was made: every uid is unique,
    no uid / norm_key / group_id appears in two parts, and (optionally) no pair
    of near-duplicate texts straddles two parts.
    """
    required = [split_col, uid_col, group_col]
    if check_near_dup:
        required.append(text_col)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")
    if frame[uid_col].duplicated().any():
        raise SplitLeakageError(f"{uid_col!r} is not unique")
    if frame[split_col].isna().any():
        raise SplitLeakageError("some rows have no split assignment")
    if not set(frame[split_col].unique()).issubset(PART_NAMES):
        raise SplitLeakageError(
            f"{split_col!r} contains values outside {PART_NAMES}")
    if frame[group_col].isna().any():
        raise SplitLeakageError(f"{group_col!r} contains missing values")

    keys = [uid_col, group_col] + (["group_id"] if "group_id" in frame.columns else [])
    for key in keys:
        parts_per_key = frame.groupby(key)[split_col].nunique()
        shared = parts_per_key[parts_per_key > 1]
        if len(shared):
            raise SplitLeakageError(
                f"{len(shared)} {key} value(s) appear in more than one part, "
                f"e.g. {list(shared.index[:3])}")

    if check_near_dup:
        pairs = find_near_duplicates(frame[text_col], threshold=sim_threshold)
        if len(pairs):
            parts = frame[split_col].to_numpy()
            crossing = pairs[parts[pairs[:, 0]] != parts[pairs[:, 1]]]
            if len(crossing):
                i, j = crossing[0]
                raise SplitLeakageError(
                    f"{len(crossing)} near-duplicate pair(s) straddle two parts, "
                    f"e.g. rows {i} and {j}")


def split_summary(
    frame: pd.DataFrame,
    *,
    split_col: str = "split",
    stratify_col: str = "subset",
    label_col: str = "hallucinated",
) -> pd.DataFrame:
    """Counts of examples per (split, subset) with columns for label 0 / 1 and
    the share of positives. Log this next to every calibration run."""
    table = pd.crosstab([frame[split_col], frame[stratify_col]], frame[label_col])
    table = table.reindex(columns=[0, 1], fill_value=0)
    table.columns = ["n_label0", "n_label1"]
    table["n"] = table.sum(axis=1)
    table["frac_label1"] = table["n_label1"] / table["n"]
    return table
