"""
Block C, steps C4-C8: calibrate detector scores with the C1 threshold and
check the recall guarantee.

Every run splits the labelled calibration-split answers into fit / calibrate /
eval (C3, src/calibration/splits.py), grouped so no prompt or near-copy of a
prompt lands in two parts. Classifiers train on fit, tau is set on calibrate,
and recall is measured on eval. Scores that need no training ignore fit, so
both families are calibrated and evaluated on the same rows.

    C4  sampling-based scores, one run on the seed-42 split
    C5  single-pass scores, the same run
    C6  the run repeated on many splits: does mean recall reach the target?
    C7  calibration size sweep on the seed-42 split
    C8  calibrate on one subset, evaluate on another (exchangeability)

C4 reproduces the conformal-abstention setup (Abbasi-Yadkori et al. 2024,
arXiv:2405.01563) with its match-count score: how many of the 15 samples
mean the same as the main answer, from the B9 clustering. SelfCheck-NLI is
the main sampling score and is used when B3 has saved it.

Outputs, each folder with config.json:
    data/c4/results.csv            data/c5/results.csv
    data/c6/trials.parquet, summary.csv, figures/
    data/c7/trials.parquet, summary.csv, figures/
    data/c8/summary.csv, figures/

Labels are provisional, so these numbers test the pipeline, not the paper.

Run from the repository root:
    python scripts/run_calibration.py
    python scripts/run_calibration.py --trials 10 --size-repeats 20   # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration.splits import make_splits  # noqa: E402
from src.eval import plots  # noqa: E402
from src.eval.coverage import (  # noqa: E402
    calibrate_and_evaluate,
    expected_recall,
    recall_spread,
    summarize_trials,
)
from src.signals.common import (  # noqa: E402
    CALIBRATION_PATH,
    DATA_DIR,
    LABELS_PATH,
    git_commit_sha,
)


TARGETS = (0.90, 0.95)
FRACTIONS = (0.4, 0.3, 0.3)  # fit, calibrate, eval
SEED = 42
SUBSETS = ("dialogue", "qa", "summarization", "truthfulqa")
SIZES = (10, 20, 50, 100, 200, 500, 1000, 2000)  # calibration positives, C7
B3_SCORES = DATA_DIR / "b3" / "selfcheck_scores.parquet"

# name -> (family, needs a fitted model, short note for the tables)
SCORES = {
    "selfcheck_nli": ("sampling", False, "B3 SelfCheck-NLI"),
    "match_count": ("sampling", False, "conformal-abstention match count, strict clusters"),
    "match_count_nonstrict": ("sampling", False, "match count, non-strict clusters"),
    "semantic_entropy": ("sampling", False, "B4"),
    "naive_entropy": ("sampling", False, "B4, no clustering"),
    "b8_gradient_boosting": ("single-pass", True, "B8 features, trained on fit"),
    "b8_logistic_regression": ("single-pass", True, "B8 features, trained on fit"),
    "logprob_neg_mean": ("single-pass", False, "B6"),
    "accprobe_tbg": ("single-pass", False, "B5, layer chosen with these labels: optimistic"),
}
MAIN = {"sampling": ["selfcheck_nli", "match_count"], "single-pass": ["b8_gradient_boosting"]}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200, help="C6/C8 splits")
    parser.add_argument("--size-repeats", type=int, default=200, help="C7 draws per size")
    parser.add_argument("--sim-threshold", type=float, default=0.9)
    parser.add_argument("--no-near-dup", action="store_true",
                        help="Group by norm_key only (skip the C3 near-duplicate merge).")
    parser.add_argument("--output-root", type=Path, default=DATA_DIR,
                        help="Folder that gets c4/ ... c8/ (default: data/).")
    return parser.parse_args()


# -------------------------------------------------------------------
# Data
# -------------------------------------------------------------------

def match_count_score(cluster_ids) -> float:
    """1 - share of the samples in the main answer's cluster (main answer first)."""
    ids = np.asarray(cluster_ids)
    return 1.0 - float((ids[1:] == ids[0]).sum()) / (len(ids) - 1)


def load_frame() -> tuple[pd.DataFrame, list[str]]:
    labels = pd.read_parquet(LABELS_PATH, columns=["uid", "subset", "hallucinated"])
    frame = labels[labels["hallucinated"].notna()].copy()
    frame["hallucinated"] = frame["hallucinated"].astype(int)
    frame = frame.merge(pd.read_parquet(CALIBRATION_PATH, columns=["uid", "norm_key"]), on="uid")

    b4 = pd.read_parquet(DATA_DIR / "b4" / "scores.parquet",
                         columns=["uid", "semantic_entropy", "naive_entropy"])
    b9 = pd.read_parquet(DATA_DIR / "b9" / "scores.parquet",
                         columns=["uid", "cluster_ids", "cluster_ids_nonstrict"])
    b9["match_count"] = b9["cluster_ids"].map(match_count_score)
    b9["match_count_nonstrict"] = b9["cluster_ids_nonstrict"].map(match_count_score)
    b6 = pd.read_parquet(DATA_DIR / "b6" / "scores.parquet", columns=["uid", "logprob_neg_mean"])
    b5 = pd.read_parquet(DATA_DIR / "b5" / "scores.parquet", columns=["uid", "accprobe_tbg"])
    features = pd.read_parquet(DATA_DIR / "b7" / "feature_table.parquet")
    feature_names = [c for c in features.columns if c.startswith(("b1_", "ling_"))]

    for part in (b4, b9[["uid", "match_count", "match_count_nonstrict"]], b6, b5, features):
        frame = frame.merge(part, on="uid", how="left")

    if B3_SCORES.exists():
        b3 = pd.read_parquet(B3_SCORES, columns=["uid", "SelfCheck-NLI"])
        frame = frame.merge(b3.rename(columns={"SelfCheck-NLI": "selfcheck_nli"}), on="uid", how="left")
    else:
        print(f"[WARNING] {B3_SCORES} not found: C4 runs without SelfCheck-NLI. "
              "Rerun B3 (slurm/b3.sh) to create it.")

    stored = [s for s, (_, fit, _) in SCORES.items() if not fit and s in frame]
    missing = frame[stored + feature_names].isna().any()
    if missing.any():
        raise ValueError(f"Scores missing for some labelled answers: {list(missing[missing].index)}")
    return frame, feature_names


def fit_single_pass(train: pd.DataFrame, features: list[str]) -> dict:
    """Same pipelines and settings as src/eval/b8_single_pass_classifier.py."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight

    X, y = train[features].to_numpy(), train["hallucinated"].to_numpy()
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ])
    boosted = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(random_state=SEED)),
    ])
    logistic.fit(X, y)
    boosted.fit(X, y, model__sample_weight=compute_sample_weight("balanced", y))
    return {"b8_logistic_regression": logistic, "b8_gradient_boosting": boosted}


def split_and_score(grouped: pd.DataFrame, seed: int, features: list[str]) -> pd.DataFrame:
    """One fit/calibrate/eval split, with the classifier scores filled in."""
    split, _ = make_splits(grouped, FRACTIONS, seed=seed, near_dup=False, group_col="group_id")
    models = fit_single_pass(split[split["split"] == "fit"], features)
    held = split[split["split"] != "fit"].copy()
    for name, model in models.items():
        held[name] = model.predict_proba(held[features].to_numpy())[:, 1]
    return held


# -------------------------------------------------------------------
# Experiments
# -------------------------------------------------------------------

def run_split(held: pd.DataFrame, scores: list[str], seed: int) -> list[dict]:
    """
    Every calibrate -> evaluate pairing C4-C6 and C8 need, for one split:
    all -> all, all -> each subset, subset -> subset, and all but one -> that one.
    """
    cal = held[held["split"] == "calibrate"]
    ev = held[held["split"] == "eval"]
    cal_sets = {"all": cal, **{s: cal[cal["subset"] == s] for s in SUBSETS},
                **{f"not {s}": cal[cal["subset"] != s] for s in SUBSETS}}
    pairs = [("all", "all")] + [("all", s) for s in SUBSETS]
    pairs += [(a, b) for a in SUBSETS for b in SUBSETS] + [(f"not {s}", s) for s in SUBSETS]

    rows = []
    for score in scores:
        family = SCORES[score][0]
        for target in TARGETS:
            for cal_on, eval_on in pairs:
                c = cal_sets[cal_on]
                e = ev if eval_on == "all" else ev[ev["subset"] == eval_on]
                out = calibrate_and_evaluate(c[score], c["hallucinated"], e[score],
                                             e["hallucinated"], 1 - target)
                rows.append({"seed": seed, "score": score, "family": family,
                             "calibrate_on": cal_on, "evaluate_on": eval_on, **out,
                             "target_recall": target})  # exact, not 1 - (1 - target)
    return rows


def size_sweep(held: pd.DataFrame, scores: list[str], repeats: int) -> pd.DataFrame:
    """C7: draw n calibration positives (all negatives kept), evaluate on all of eval."""
    cal = held[held["split"] == "calibrate"]
    ev = held[held["split"] == "eval"]
    positives = cal[cal["hallucinated"] == 1]
    negatives = cal[cal["hallucinated"] == 0]
    sizes = [n for n in SIZES if n < len(positives)] + [len(positives)]
    rng = np.random.default_rng(SEED)
    rows = []
    for n in sizes:
        for repeat in range(repeats if n < len(positives) else 1):
            sample = pd.concat([positives.iloc[rng.choice(len(positives), n, replace=False)], negatives])
            for score in scores:
                for target in TARGETS:
                    out = calibrate_and_evaluate(sample[score], sample["hallucinated"], ev[score],
                                                 ev["hallucinated"], 1 - target)
                    rows.append({"score": score, "family": SCORES[score][0], "size": n,
                                 "repeat": repeat, **out, "target_recall": target})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Figures
# -------------------------------------------------------------------

def figure_c6(summary: pd.DataFrame, out: Path):
    plt = plots.setup(headless=True)
    table = summary[(summary["calibrate_on"] == "all") & (summary["evaluate_on"] == "all")]
    scores = list(dict.fromkeys(table["score"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(scores) + 1.6), sharey=True)
    for ax, metric, label in [(axes[0], "mean_recall", "Recall on eval"),
                              (axes[1], "mean_flag_rate", "Flag rate on eval")]:
        for i, target in enumerate(TARGETS):
            rows = table[table["target_recall"] == target].set_index("score").reindex(scores)
            y = np.arange(len(scores)) + (i - 0.5) * 0.3
            xerr = rows["sd_recall"] if metric == "mean_recall" else None
            ax.errorbar(rows[metric], y, xerr=xerr, fmt="o", color=plots.PALETTE[i],
                        label=f"target {target:.0%}", capsize=2)
            if metric == "mean_recall":
                ax.axvline(target, color=plots.PALETTE[i], ls="--", lw=1)
        ax.set_yticks(np.arange(len(scores)), scores)
        ax.set_xlabel(label)
    axes[0].legend(loc="best")
    fig.suptitle(f"C6: mean ± SD over {int(table['trials'].iloc[0])} splits", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out / "recall_and_flag_rate.png")
    plt.close(fig)


def figure_c7(summary: pd.DataFrame, scores: list[str], out: Path):
    plt = plots.setup(headless=True)
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(5.5 * len(TARGETS), 4), sharey=True)
    for ax, target in zip(np.atleast_1d(axes), TARGETS):
        rows = summary[summary["target_recall"] == target]
        sizes = np.array(sorted(rows["size"].unique()))
        mean = np.array([expected_recall(n, 1 - target) for n in sizes])
        sd = np.array([recall_spread(n, 1 - target) for n in sizes])
        ax.fill_between(sizes, mean - sd, np.minimum(mean + sd, 1), color=plots.NEUTRAL,
                        alpha=0.3, label="theory, mean ± SD")
        for i, score in enumerate(scores):
            r = rows[rows["score"] == score].sort_values("size")
            ax.errorbar(r["size"], r["mean_recall"], yerr=r["sd_recall"], marker="o",
                        color=plots.PALETTE[i], label=score, capsize=2)
        ax.axhline(target, color=plots.TEXT_MUTED, ls="--", lw=1)
        ax.set(xscale="log", xlabel="Calibration positives", title=f"target {target:.0%}")
    np.atleast_1d(axes)[0].set_ylabel("Recall on eval")
    np.atleast_1d(axes)[0].legend(loc="best")
    fig.suptitle("C7: recall vs calibration size", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out / "recall_vs_size.png")
    plt.close(fig)


def figure_c8(summary: pd.DataFrame, scores: list[str], out: Path, target: float = 0.95):
    plt = plots.setup(headless=True)
    rows = summary[(summary["target_recall"] == target)
                   & summary["calibrate_on"].isin(SUBSETS) & summary["evaluate_on"].isin(SUBSETS)]
    fig, axes = plt.subplots(1, len(scores), figsize=(4.2 * len(scores) + 1.6, 4.4),
                             squeeze=False, constrained_layout=True)
    for ax, score in zip(axes[0], scores):
        grid = (rows[rows["score"] == score]
                .pivot(index="calibrate_on", columns="evaluate_on", values="mean_recall")
                .reindex(index=list(SUBSETS), columns=list(SUBSETS)))
        image = ax.imshow(grid.to_numpy(), vmin=0.5, vmax=1.0, cmap="RdYlGn")
        for (i, j), value in np.ndenumerate(grid.to_numpy()):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(SUBSETS)), SUBSETS, rotation=30, ha="right")
        ax.set_yticks(range(len(SUBSETS)), SUBSETS if ax is axes[0][0] else [])
        ax.set(xlabel="evaluate on", ylabel="calibrate on" if ax is axes[0][0] else "", title=score)
        ax.grid(False)
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label="mean recall")
    fig.suptitle(f"C8: recall at a {target:.0%} target when the subset changes", fontweight="bold")
    plots.save(fig, out / "recall_by_subset_pair.png")
    plt.close(fig)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def write_config(directory: Path, **extra):
    directory.mkdir(parents=True, exist_ok=True)
    config = {"targets": list(TARGETS), "fractions": list(FRACTIONS), "seed": SEED,
              "git_commit_sha": git_commit_sha(), "labels": "provisional", **extra}
    (directory / "config.json").write_text(json.dumps(config, indent=2, default=str))


def main():
    args = parse_args()
    started = time.time()

    frame, features = load_frame()
    scores = [s for s in SCORES if SCORES[s][1] or s in frame]
    print(f"[INFO] Labelled answers: {len(frame):,}   scores: {scores}")

    # C3: the near-duplicate merge runs once; every split reuses its groups.
    grouped, info = make_splits(frame, FRACTIONS, seed=SEED, near_dup=not args.no_near_dup,
                                sim_threshold=args.sim_threshold)
    grouped = grouped.drop(columns="split")
    print(f"[INFO] Groups: {info['n_groups_exact']:,} norm_keys -> "
          f"{info['n_groups_after_merge']:,} after merging {info['n_near_dup_pairs']:,} "
          f"near-duplicate pairs (largest group {info['largest_group']})")
    split_info = {k: info[k] for k in ("near_dup", "sim_threshold", "n_groups_exact",
                                       "n_groups_after_merge", "n_near_dup_pairs", "largest_group")}

    seeds = [SEED + i for i in range(args.trials)]
    rows, main_split = [], None
    for i, seed in enumerate(seeds):
        held = split_and_score(grouped, seed, features)
        if seed == SEED:
            main_split = held
        rows += run_split(held, scores, seed)
        if (i + 1) % 10 == 0 or i + 1 == len(seeds):
            print(f"[INFO] splits {i + 1}/{len(seeds)}  ({(time.time() - started) / 60:.1f} min)")
    trials = pd.DataFrame(rows)

    # C4 / C5: the seed-42 split, tau set on all calibration rows.
    main = trials[(trials["seed"] == SEED) & (trials["calibrate_on"] == "all")]
    columns = ["score", "evaluate_on", "target_recall", "tau", "n_cal_pos", "recall", "flag_rate",
               "precision", "accepted_hallucination_rate", "coverage_gap", "n", "n_pos"]
    for step, family in (("c4", "sampling"), ("c5", "single-pass")):
        out = args.output_root / step
        table = main[main["family"] == family][columns].copy()
        table["note"] = table["score"].map(lambda s: SCORES[s][2])
        write_config(out, family=family, scores=sorted(set(table["score"])), split=split_info,
                     rows_per_part=info["rows_per_split"])
        table.to_csv(out / "results.csv", index=False)

    # C6: tau from all calibration rows, recall overall and per subset.
    c6 = trials[trials["calibrate_on"] == "all"]
    out = args.output_root / "c6"
    write_config(out, trials=args.trials, scores=scores, split=split_info)
    c6.to_parquet(out / "trials.parquet", index=False)
    c6_summary = summarize_trials(c6, ["score", "family", "calibrate_on", "evaluate_on", "target_recall"])
    c6_summary.to_csv(out / "summary.csv", index=False)
    figure_c6(c6_summary, out / "figures")

    # C7: calibration size, on the seed-42 split.
    sweep = size_sweep(main_split, scores, args.size_repeats)
    out = args.output_root / "c7"
    write_config(out, size_repeats=args.size_repeats, sizes=sorted(sweep["size"].unique()),
                 scores=scores, split=split_info)
    sweep.to_parquet(out / "trials.parquet", index=False)
    c7_summary = summarize_trials(sweep, ["score", "family", "size", "target_recall"])
    c7_summary["theory_recall"] = [expected_recall(n, 1 - t) for n, t in
                                   zip(c7_summary["size"], c7_summary["target_recall"])]
    c7_summary["theory_sd"] = [recall_spread(n, 1 - t) for n, t in
                               zip(c7_summary["size"], c7_summary["target_recall"])]
    c7_summary.to_csv(out / "summary.csv", index=False)
    main_scores = [s for family in MAIN.values() for s in family if s in scores]
    figure_c7(c7_summary, main_scores, out / "figures")

    # C8: tau from one subset (or all but one), recall on another.
    c8 = trials[trials["calibrate_on"] != "all"]
    out = args.output_root / "c8"
    write_config(out, trials=args.trials, scores=scores, split=split_info)
    c8_summary = summarize_trials(c8, ["score", "family", "calibrate_on", "evaluate_on", "target_recall"])
    c8_summary["in_domain"] = c8_summary["calibrate_on"] == c8_summary["evaluate_on"]
    c8_summary.to_csv(out / "summary.csv", index=False)
    figure_c8(c8_summary, main_scores, out / "figures")

    overall = c6_summary[c6_summary["evaluate_on"] == "all"]
    print(overall[["score", "target_recall", "mean_recall", "sd_recall", "frac_below_target",
                   "mean_flag_rate"]].round(3).to_string(index=False))
    print(f"[SUCCESS] C4-C8 written to {args.output_root}/c4 ... c8 in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
