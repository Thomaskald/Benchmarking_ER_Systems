#!/usr/bin/env python3
"""
Statistical + scalability analysis for the 7-framework ER comparison.

Produces the two artefacts the professor asked for, plus a supporting
scalability curve:

    1. Nemenyi critical-difference (CD) diagrams          -> STATS/cd_*.png
       Demsar (2006): Friedman omnibus test across datasets, then the
       Nemenyi post-hoc.  Two mean-rank axes are drawn (pairwise F1 and
       B-Cubed F1), with frameworks whose ranks differ by less than the
       critical difference joined by a bar (statistically tied).

    2. Pareto fronts (quality vs. runtime)               -> STATS/pareto_*.png
       Mean F1 (maximise) vs. mean wall-clock time in seconds (minimise,
       log axis).  The non-dominated frontier is highlighted.

    3. Scalability curves (runtime vs. dataset size)     -> STATS/scalability_runtime.png

All 7 frameworks share the 8 CCER datasets D2..D9, so the analysis runs on
that complete 7x8 block.  (Magellan, RecordLinkage, LinkTransformer never ran
CORA/CDDB, so those two DER datasets cannot enter a 7-way Friedman test.)

Run:  python3 stats_analysis.py
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- paths -----
HOME = "/home/thomas"
OUT_ROOT = "/home/thomas/ER_PLOTS_AND_DIAGRAMS"
STATS_DIR = os.path.join(OUT_ROOT, "STATS")

FRAMEWORK_FILES = {
    "Dedupe":          [f"{HOME}/DEDUPE/results/dedupe_bestconfig_eval.csv"],
    "pyJedAI":         [f"{HOME}/PYJEDAI/results/pyjedai_bestconfig_eval.csv"],
    "Zingg":           [f"{HOME}/ZINGG/results/zingg_bestconfig_eval.csv"],
    "Splink":          [f"{HOME}/SPLINK/results/splink_bestconfig_eval.csv"],
    "Magellan":        [f"{HOME}/MAGELLAN/magellan_bestconfig_eval.csv"],
    "RecordLinkage":   [f"{HOME}/RECORDLINKAGE/results/recordlinkage_bestconfig_eval.csv"],
    "LinkTransformer": sorted(glob.glob(
        f"{HOME}/LINKTRANSFORMER/results/linktransformer_bestconfig_evalD*.csv")),
}

FRAMEWORK_ORDER = ["Dedupe", "Splink", "Zingg", "Magellan",
                   "pyJedAI", "LinkTransformer", "RecordLinkage"]
FRAMEWORK_COLORS = {
    "Dedupe":          "#2a78d6",
    "Splink":          "#1baf7a",
    "Zingg":           "#eda100",
    "Magellan":        "#008300",
    "pyJedAI":         "#4a3aa7",
    "LinkTransformer": "#e34948",
    "RecordLinkage":   "#e87ba4",
}

# All 7 frameworks share exactly these 8 datasets -> complete block.
CCER_DATASETS = ["D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
# Only these 4 frameworks ran the 2 DER (dirty) datasets.
DER_DATASETS = ["CORA", "CDDB"]
DER_FRAMEWORKS = ["Dedupe", "Splink", "Zingg", "pyJedAI"]

VALUE_COLS = ["pairwise_f1", "bcubed_f1", "time_sec", "n_entities"]

# Nemenyi critical values q_alpha (studentized range / sqrt(2)), df=inf.
# Index = number of models k.  Source: Demsar (2006), Table 5(b).
NEMENYI_Q = {
    0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268},
    0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693,
           8: 2.780, 9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030},
}


# --------------------------------------------------------------- loading ----
def load_all():
    frames = []
    for fw, files in FRAMEWORK_FILES.items():
        for path in files:
            if not os.path.exists(path):
                print(f"  [skip] missing file for {fw}: {path}")
                continue
            df = pd.read_csv(path)
            cols = ["dataset"] + [c for c in VALUE_COLS if c in df.columns]
            df = df[cols].copy()
            df.insert(0, "framework", fw)
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["dataset"] = combined["dataset"].astype(str).str.upper()
    return combined


# ------------------------------------------------------ nemenyi machinery ---
def critical_difference(k, n, alpha):
    q = NEMENYI_Q[alpha][k]
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def rank_matrix(combined, metric, datasets, frameworks):
    """Return a datasets x frameworks matrix of scores (higher = better)."""
    M = pd.DataFrame(index=datasets, columns=frameworks, dtype=float)
    for ds in datasets:
        for fw in frameworks:
            row = combined[(combined["framework"] == fw) &
                           (combined["dataset"] == ds)]
            if len(row):
                M.loc[ds, fw] = float(row.iloc[0][metric])
    return M


def average_ranks(score_matrix):
    """Rank frameworks per dataset (rank 1 = best), average over datasets."""
    # rank descending: highest score -> rank 1.  ties share the mean rank.
    ranks = score_matrix.rank(axis=1, ascending=False, method="average")
    return ranks.mean(axis=0)


# ----------------------------------------------------------- CD diagram -----
def plot_cd_diagram(avg_ranks, cd, k, n, title, out_path, alpha=0.05):
    """Critical-difference diagram, decluttered.

    Layout keeps the two visually-noisy elements apart:
      - "tied" significance bars sit ABOVE the rank axis;
      - the label connectors droop BELOW it, as thin neutral lines,
        with each framework's identity carried by a coloured dot ON the axis.
    """
    order = avg_ranks.sort_values().index.tolist()  # best (lowest rank) first
    ranks = avg_ranks[order].values

    # --- cliques: maximal runs (in rank order) whose span <= CD ---------
    bars = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and (ranks[j + 1] - ranks[i]) <= cd:
            j += 1
        if j > i:
            bars.append((ranks[i], ranks[j]))
        i += 1
    bars = [b for b in bars
            if not any(o != b and o[0] <= b[0] and o[1] >= b[1] for o in bars)]

    lo, hi = 1, k
    left, right = 0.16, 0.84
    def x(r):
        return left + (r - lo) / (hi - lo) * (right - left)

    n_left = (k + 1) // 2
    row_h = 0.42                                      # vertical gap between labels
    CONN = "#b3b3b3"                                  # neutral connector grey
    axis_y = 0.0

    # geometry: CD reference bar sits above the axis; the tied-group bars
    # now sit just BELOW the axis, inside the body of the diagram, with the
    # label connectors dropping below them.
    cd_y = 0.55                                       # CD reference bar height
    bar_y0 = -0.30                                    # first tied-bar (below axis)
    bar_step = 0.30
    tie_bottom = bar_y0 - (len(bars) - 1) * bar_step if bars else bar_y0
    labels_top = tie_bottom - 0.42                    # labels start below bars
    bottom = labels_top - (max(n_left, k - n_left) - 1) * row_h - 0.30

    fig, ax = plt.subplots(figsize=(10, 3.7 + max(n_left, k - n_left) * 0.32))
    ax.set_xlim(0, 1)

    # rank axis with integer ticks (numbers above the axis)
    ax.plot([x(lo), x(hi)], [axis_y, axis_y], "k-", lw=1.6, zorder=2)
    for r in range(lo, hi + 1):
        ax.plot([x(r), x(r)], [axis_y, axis_y + 0.10], "k-", lw=1.1, zorder=2)
        ax.text(x(r), axis_y + 0.17, str(r), ha="center", va="bottom",
                fontsize=11)

    # CD reference bar (above the axis)
    ax.plot([x(lo), x(lo + cd)], [cd_y, cd_y], "k-", lw=2.4)
    for xe in (lo, lo + cd):
        ax.plot([x(xe), x(xe)], [cd_y - 0.05, cd_y + 0.05], "k-", lw=2.4)
    ax.text(x(lo + cd / 2), cd_y + 0.09, f"CD = {cd:.2f}  (alpha={alpha})",
            ha="center", va="bottom", fontsize=11)

    # connectors from the axis down to each side label + coloured dot on axis
    for i, fw in enumerate(order):
        r = avg_ranks[fw]
        on_left = i < n_left
        row = i if on_left else i - n_left
        depth = labels_top - row * row_h
        xlab = (left - 0.03) if on_left else (right + 0.03)
        ha = "right" if on_left else "left"
        ax.plot([x(r), x(r)], [axis_y, depth], color=CONN, lw=1.1, zorder=1)
        ax.plot([x(r), xlab], [depth, depth], color=CONN, lw=1.1, zorder=1)
        ax.scatter([x(r)], [axis_y], s=70, color=FRAMEWORK_COLORS[fw],
                   edgecolor="white", linewidth=1.0, zorder=5)
        ax.text(xlab, depth, f"{fw}   {r:.2f}", ha=ha, va="center",
                fontsize=11.5, color=FRAMEWORK_COLORS[fw], fontweight="bold")

    # tied-group bars just BELOW the axis, drawn on top of the connectors
    for idx, (a, b) in enumerate(bars):
        by = bar_y0 - idx * bar_step
        ax.plot([x(a), x(b)], [by, by], "-", color="#333", lw=4.5,
                solid_capstyle="round", zorder=4)

    ax.set_ylim(bottom, cd_y + 0.35)
    ax.axis("off")
    ax.set_title(f"{title}\n(k={k} frameworks, N={n} datasets,  "
                 f"1 = best; bars join groups with no significant difference)",
                 fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mean_ranks_bar(avg_ranks, k, n, title, out_path, note=None):
    """Descriptive horizontal mean-rank bar chart (no significance claim).

    Used where a formal CD diagram is not meaningful (e.g. N=2 datasets)."""
    order = avg_ranks.sort_values().index.tolist()   # best first
    vals = [avg_ranks[fw] for fw in order]
    colors = [FRAMEWORK_COLORS[fw] for fw in order]

    fig, ax = plt.subplots(figsize=(8, 0.7 * len(order) + 2))
    ys = range(len(order))
    ax.barh(list(ys), vals, color=colors, edgecolor="white", height=0.62)
    for y, v in zip(ys, vals):
        ax.text(v + 0.05, y, f"{v:.2f}", va="center", ha="left", fontsize=11,
                fontweight="bold")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(order, fontsize=11)
    ax.invert_yaxis()                                # best framework on top
    ax.set_xlim(0, k + 0.6)
    ax.set_xlabel("Mean rank  (1 = best)")
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    if note:
        ax.text(0.5, -0.22, note, transform=ax.transAxes, ha="center",
                va="top", fontsize=9, style="italic", color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------- pareto machinery --
def pareto_front(points):
    """points: list of (x_cost_min, y_quality_max, label).  Return set of
    labels that are non-dominated (lower cost AND higher quality wins)."""
    front = []
    for i, (xi, yi, li) in enumerate(points):
        dominated = False
        for j, (xj, yj, lj) in enumerate(points):
            if j == i:
                continue
            if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                dominated = True
                break
        if not dominated:
            front.append(li)
    return set(front)


def plot_pareto(agg, metric, title, out_path):
    """agg: DataFrame indexed by framework with columns 'f1' and 'time'."""
    pts = [(agg.loc[fw, "time"], agg.loc[fw, metric], fw) for fw in agg.index]
    front = pareto_front([(t, q, fw) for (t, q, fw) in pts])

    fig, ax = plt.subplots(figsize=(8.4, 6))
    # draw the frontier staircase (sorted by time)
    fp = sorted([(agg.loc[fw, "time"], agg.loc[fw, metric])
                 for fw in front])
    if len(fp) > 1:
        fx, fy = zip(*fp)
        ax.plot(fx, fy, color="#888", lw=1.4, ls="--", zorder=1,
                label="Pareto frontier")

    for fw in agg.index:
        t, q = agg.loc[fw, "time"], agg.loc[fw, metric]
        is_front = fw in front
        ax.scatter([t], [q], s=190 if is_front else 110,
                   color=FRAMEWORK_COLORS[fw],
                   edgecolor="black" if is_front else "white",
                   linewidth=1.8 if is_front else 0.8,
                   marker="*" if is_front else "o", zorder=3)
        ax.annotate(fw, (t, q), textcoords="offset points",
                    xytext=(10, 7), fontsize=9,
                    fontweight="bold" if is_front else "normal",
                    color=FRAMEWORK_COLORS[fw])

    ax.set_xscale("log")
    ax.set_xlabel("Mean runtime per dataset  [s]  (log scale, lower = better)")
    ax.set_ylabel(f"Mean {metric.replace('_', ' ')}  (higher = better)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    # proxy legend: star = Pareto-optimal, circle = dominated
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#888", lw=1.4, ls="--", label="Pareto frontier"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#555",
               markeredgecolor="black", markersize=15,
               label="Pareto-optimal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#999",
               markersize=9, label="dominated"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9,
              framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------- scalability curve ----
def plot_scalability(combined, datasets, out_path):
    sizes = (combined[combined["dataset"].isin(datasets)]
             .groupby("dataset")["n_entities"].first())
    fig, ax = plt.subplots(figsize=(8.4, 6))
    for fw in FRAMEWORK_ORDER:
        xs, ys = [], []
        for ds in datasets:
            row = combined[(combined["framework"] == fw) &
                           (combined["dataset"] == ds)]
            if len(row) and pd.notna(row.iloc[0]["time_sec"]):
                xs.append(sizes[ds])
                ys.append(row.iloc[0]["time_sec"])
        if not xs:
            continue
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]
        ax.plot(xs, ys, marker="o", color=FRAMEWORK_COLORS[fw], label=fw,
                lw=1.8, ms=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size  [# entities]  (log scale)")
    ax.set_ylabel("Runtime  [s]  (log scale)")
    ax.set_title("Scalability: runtime vs. dataset size (CCER, D2–D9)")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


METRICS = [("pairwise_f1", "Pairwise F1"), ("bcubed_f1", "B-Cubed F1")]


def pareto_section(combined, datasets, frameworks, family, log):
    """Pareto front (quality vs runtime) for one dataset family."""
    sub = combined[combined["dataset"].isin(datasets)]
    agg = sub.groupby("framework").agg(
        pairwise_f1=("pairwise_f1", "mean"),
        bcubed_f1=("bcubed_f1", "mean"),
        time=("time_sec", "mean")).loc[frameworks]
    log(f"\nMean over {family} datasets ({', '.join(datasets)}):")
    log(agg.round(3).to_string())
    for metric, nice in METRICS:
        front = pareto_front([(agg.loc[fw, "time"], agg.loc[fw, metric], fw)
                              for fw in agg.index])
        log(f"Pareto-optimal ({family}, {nice} vs runtime): "
            f"{', '.join(sorted(front))}")
        plot_pareto(agg, metric,
                    f"Pareto front — {nice} vs. runtime ({family})",
                    os.path.join(STATS_DIR, f"pareto_{family}_{metric}.png"))
    agg.round(4).to_csv(os.path.join(STATS_DIR, f"pareto_summary_{family}.csv"))


# ------------------------------------------------------------------- main ----
def main():
    os.makedirs(STATS_DIR, exist_ok=True)
    # start clean so stale/renamed artefacts don't linger
    for f in glob.glob(os.path.join(STATS_DIR, "*")):
        os.remove(f)
    combined = load_all()

    summary_lines = []
    def log(s=""):
        print(s)
        summary_lines.append(s)

    # ===================== CCER: full Friedman + Nemenyi ================
    log("=" * 70)
    log("CCER  —  7 frameworks x 8 datasets (D2-D9)")
    log("Friedman omnibus test + Nemenyi post-hoc (critical-difference)")
    log("=" * 70)
    ccer_fw = [fw for fw in FRAMEWORK_ORDER if fw in set(combined["framework"])]
    for metric, nice in METRICS:
        M = rank_matrix(combined, metric, CCER_DATASETS, ccer_fw)
        if M.isna().any().any():
            log(f"[warn] {metric}: incomplete block, skipping")
            continue
        k, n = len(ccer_fw), len(CCER_DATASETS)
        stat, p = friedmanchisquare(*[M[fw].values for fw in ccer_fw])
        avg = average_ranks(M)
        cd05 = critical_difference(k, n, 0.05)
        log(f"\n--- {nice} ---")
        log(f"Friedman chi2 = {stat:.3f},  p = {p:.4g}  "
            f"({'significant' if p < 0.05 else 'not significant'} at 0.05)")
        log(f"CD(0.05) = {cd05:.3f}    mean ranks (1 = best):")
        for fw in avg.sort_values().index:
            log(f"    {fw:<16} {avg[fw]:.3f}")
        plot_cd_diagram(avg, cd05, k, n,
                        f"Critical-difference diagram — {nice}  (CCER)",
                        os.path.join(STATS_DIR, f"cd_CCER_{metric}.png"))
        M.to_csv(os.path.join(STATS_DIR, f"scores_CCER_{metric}.csv"))
        pd.DataFrame({"mean_rank": avg}).sort_values("mean_rank").to_csv(
            os.path.join(STATS_DIR, f"meanranks_CCER_{metric}.csv"))

    # ===================== DER: descriptive only (N=2) ==================
    log("\n" + "=" * 70)
    log("DER  —  4 frameworks x 2 datasets (CORA, CDDB)")
    log("NOTE: with only N=2 datasets a Friedman/Nemenyi test is under-")
    log("powered and a CD diagram is not meaningful. Reporting the test")
    log("for completeness, but the ranking below is DESCRIPTIVE only.")
    log("=" * 70)
    der_fw = [fw for fw in FRAMEWORK_ORDER if fw in DER_FRAMEWORKS]
    for metric, nice in METRICS:
        M = rank_matrix(combined, metric, DER_DATASETS, der_fw)
        if M.isna().any().any():
            log(f"[warn] {metric}: incomplete DER block, skipping")
            continue
        k, n = len(der_fw), len(DER_DATASETS)
        stat, p = friedmanchisquare(*[M[fw].values for fw in der_fw])
        avg = average_ranks(M)
        log(f"\n--- {nice} ---")
        log(f"Friedman chi2 = {stat:.3f},  p = {p:.4g}  "
            f"(under-powered at N=2 — interpret with caution)")
        log("Mean ranks (1 = best, descriptive):")
        for fw in avg.sort_values().index:
            log(f"    {fw:<16} {avg[fw]:.3f}")
        plot_mean_ranks_bar(
            avg, k, n, f"Mean ranks — {nice}  (DER: CORA, CDDB)",
            os.path.join(STATS_DIR, f"meanranks_DER_{metric}.png"),
            note="Descriptive ranking only — N=2 datasets is too few for a "
                 "valid Friedman/Nemenyi significance test.")
        M.to_csv(os.path.join(STATS_DIR, f"scores_DER_{metric}.csv"))

    # ===================== Pareto fronts (both families) ================
    log("\n" + "=" * 70)
    log("PARETO FRONTS  (quality vs. runtime)")
    log("=" * 70)
    pareto_section(combined, CCER_DATASETS, ccer_fw, "CCER", log)
    pareto_section(combined, DER_DATASETS, der_fw, "DER", log)

    # ===================== scalability curve (CCER) =====================
    plot_scalability(combined, CCER_DATASETS,
                     os.path.join(STATS_DIR, "scalability_CCER.png"))

    with open(os.path.join(STATS_DIR, "stats_summary.txt"), "w") as fh:
        fh.write("\n".join(summary_lines) + "\n")
    log(f"\nDone. Artefacts written to {STATS_DIR}")


if __name__ == "__main__":
    main()
