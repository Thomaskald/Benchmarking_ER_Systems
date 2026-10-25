#!/usr/bin/env python3
"""
Reproducible benchmark plots for the ER framework comparison.

Reads each framework's `bestconfig_eval` CSV(s), normalizes the columns to a
common schema, and produces, per dataset, four plots:

    1. pairwise  (precision, recall, f1)   -> grouped bars per framework
    2. bcubed    (precision, recall, f1)   -> grouped bars per framework
    3. pairwise f1 only                    -> one bar per framework
    4. bcubed   f1 only                    -> one bar per framework

A framework with no result on a dataset (Sudowoodo, where no configuration
finished within the time limits) gets no bar, an "n/a" marker and a footnote --
never a zero bar.

Output is split into two folders by ER family:
    CCER/   -> datasets D2..D9   (8 clean-clean ER datasets, shown as D1..D8)
    DER/    -> datasets CORA, CDDB (2 dirty ER datasets)

A merged long-format table is also written to `combined_metrics.csv`.

Run:  python3 make_plots.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from er_style import (HERE, REPO, FRAMEWORK_ORDER, FRAMEWORK_COLORS,
                      FRAMEWORK_HATCH, CCER_DATASETS, DER_DATASETS, disp,
                      TIME_LIMIT_NOTE)

# ---------------------------------------------------------------- paths -----
HOME = REPO
OUT_ROOT = HERE
CCER_DIR = os.path.join(OUT_ROOT, "CCER")
DER_DIR = os.path.join(OUT_ROOT, "DER")

# Where each framework's eval CSV(s) live. LinkTransformer is one file per
# dataset, so it gets a glob; everyone else has a single combined file.
FRAMEWORK_FILES = {
    "Dedupe":         [f"{HOME}/DEDUPE/results/dedupe_bestconfig_eval.csv"],
    "pyJedAI-syn":    [f"{HOME}/PYJEDAI/results/pyjedai_syn_bestconfig_eval.csv"],
    "pyJedAI-sem":    [f"{HOME}/PYJEDAI/results/pyjedai_sem_bestconfig_eval.csv"],
    "Zingg":          [f"{HOME}/ZINGG/results/zingg_bestconfig_eval.csv"],
    "Splink":         [f"{HOME}/SPLINK/results/splink_bestconfig_eval.csv"],
    "Magellan":       [f"{HOME}/MAGELLAN/results/magellan_bestconfig_eval.csv"],
    "RecordLinkage":  [f"{HOME}/RECORDLINKAGE/results/recordlinkage_bestconfig_eval.csv"],
    "LinkTransformer": sorted(glob.glob(
        f"{HOME}/LINKTRANSFORMER/results/linktransformer_bestconfig_evalD*.csv")),
    "DeepER":         [f"{HOME}/DEEPER/results/deeper_bestconfig_eval.csv"],
    "Sudowoodo":      [f"{HOME}/SUDOWOODO/results/sudowoodo_bestconfig_eval.csv"],
}

METRIC_COLS = [
    "pairwise_precision", "pairwise_recall", "pairwise_f1",
    "bcubed_precision", "bcubed_recall", "bcubed_f1",
]


# --------------------------------------------------------------- loading ----
def load_all():
    """Load every framework's eval CSV into one long dataframe."""
    frames = []
    for fw, files in FRAMEWORK_FILES.items():
        for path in files:
            if not os.path.exists(path):
                print(f"  [skip] missing file for {fw}: {path}")
                continue
            df = pd.read_csv(path)
            missing = [c for c in ["dataset"] + METRIC_COLS if c not in df.columns]
            if missing:
                print(f"  [warn] {fw} {os.path.basename(path)} missing {missing}")
                continue
            df = df[["dataset"] + METRIC_COLS].copy()
            df.insert(0, "framework", fw)
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    # Keep only OK-looking numeric rows.
    combined["dataset"] = combined["dataset"].astype(str).str.upper()
    return combined


# --------------------------------------------------------------- plotting ---
def _frameworks_for(df_ds):
    """Frameworks with a result for a dataset, in the fixed global order."""
    present = set(df_ds.dropna(subset=METRIC_COLS)["framework"])
    return [fw for fw in FRAMEWORK_ORDER if fw in present]


def _missing_for(df_ds):
    """Frameworks with no result (no OK config) on any of these rows."""
    na = set(df_ds[df_ds[METRIC_COLS].isna().any(axis=1)]["framework"])
    return [fw for fw in FRAMEWORK_ORDER if fw in na]


def _note_missing(fig, missing, y=-0.01, ax=None, fontsize=8.5):
    """Footnote explaining which frameworks have no result.

    Placed in figure coords, or in `ax` coords when given (so it sits a fixed
    distance under an axes-anchored legend)."""
    if not missing:
        return
    text = TIME_LIMIT_NOTE.replace("Sudowoodo", ", ".join(missing))
    kw = dict(ha="center", va="top", fontsize=fontsize, style="italic",
              color="#555555")
    if ax is not None:
        ax.text(0.5, y, text, transform=ax.transAxes, **kw)
    else:
        fig.text(0.5, y, text, **kw)


def _mark_na(ax, x, y=0.012, fontsize=6.5):
    """Visible marker where a framework has no bar."""
    ax.text(x, y, "n/a", ha="center", va="bottom", fontsize=fontsize,
            rotation=90, color="#555555")


def plot_grouped(df_ds, dataset, metric_prefix, title_metric, out_path):
    """Grouped bars: precision / recall / f1 for one metric family."""
    frameworks = _frameworks_for(df_ds)
    metrics = [f"{metric_prefix}_precision",
               f"{metric_prefix}_recall",
               f"{metric_prefix}_f1"]
    metric_labels = ["Precision", "Recall", "F1"]

    n_groups = len(metrics)
    n_fw = len(frameworks)
    group_w = 0.8
    bar_w = group_w / max(n_fw, 1)

    fig, ax = plt.subplots(figsize=(max(7, n_fw * 1.3), 5))
    for i, fw in enumerate(frameworks):
        row = df_ds[df_ds["framework"] == fw].iloc[0]
        vals = [row[m] for m in metrics]
        xs = [g - group_w / 2 + bar_w * (i + 0.5) for g in range(n_groups)]
        ax.bar(xs, vals, width=bar_w, label=fw,
               color=FRAMEWORK_COLORS[fw], edgecolor="white", linewidth=0.5,
               hatch=FRAMEWORK_HATCH[fw])

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{disp(dataset)} — {title_metric} metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    _note_missing(fig, _missing_for(df_ds))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_f1_only(df_ds, dataset, metric_prefix, title_metric, out_path):
    """Single bar per framework for the F1 of one metric family."""
    frameworks = _frameworks_for(df_ds)
    col = f"{metric_prefix}_f1"
    vals = [df_ds[df_ds["framework"] == fw].iloc[0][col] for fw in frameworks]
    colors = [FRAMEWORK_COLORS[fw] for fw in frameworks]

    fig, ax = plt.subplots(figsize=(max(6, len(frameworks) * 1.1), 5))
    xs = range(len(frameworks))
    bars = ax.bar(xs, vals, color=colors, edgecolor="white", linewidth=0.5)
    for bar, fw in zip(bars, frameworks):
        bar.set_hatch(FRAMEWORK_HATCH[fw])
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(frameworks, rotation=30, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1")
    ax.set_title(f"{disp(dataset)} — {title_metric} F1")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _note_missing(fig, _missing_for(df_ds))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overview_pr(combined, datasets, metric_prefix, title_metric,
                     family_label, out_path, style):
    """Same layout as plot_overview_f1, but precision and recall.

    Two stacked panels sharing one x axis and one legend: precision on top,
    recall below, so the trade-off behind each F1 is readable per dataset.
    """
    sub = combined[combined["dataset"].isin(datasets)]
    datasets = [d for d in datasets if d in set(sub["dataset"])]
    frameworks = [fw for fw in FRAMEWORK_ORDER if fw in set(sub["framework"])]

    n_groups = len(datasets)
    n_fw = len(frameworks)
    group_w = 0.82
    bar_w = group_w / max(n_fw, 1)

    fs = style["font"]
    fig, axes = plt.subplots(
        2, 1, sharex=True, figsize=(style["width"], style["pr_height"]))
    for ax, part in zip(axes, ["precision", "recall"]):
        col = f"{metric_prefix}_{part}"
        for i, fw in enumerate(frameworks):
            vals = []
            for ds in datasets:
                row = sub[(sub["framework"] == fw) & (sub["dataset"] == ds)]
                vals.append(row.iloc[0][col] if len(row) else np.nan)
            xs = [g - group_w / 2 + bar_w * (i + 0.5) for g in range(n_groups)]
            ax.bar(xs, vals, width=bar_w, label=fw,
                   color=FRAMEWORK_COLORS[fw], edgecolor="white", linewidth=0.6,
                   hatch=FRAMEWORK_HATCH[fw])
            for xv, v in zip(xs, vals):
                if pd.isna(v):
                    _mark_na(ax, xv, fontsize=fs * 0.75)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"{title_metric} {part}", fontsize=fs)
        ax.tick_params(axis="both", labelsize=fs)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    axes[-1].set_xticks(range(n_groups))
    axes[-1].set_xticklabels([disp(d) for d in datasets])
    axes[-1].set_xlabel("Dataset", fontsize=fs)
    fig.suptitle(f"{family_label} — {title_metric} precision and recall "
                 f"across datasets", fontsize=fs * 1.1)
    handles, labels = axes[0].get_legend_handles_labels()
    ncol = style["ncol"] or n_fw
    fig.legend(handles, labels, fontsize=fs * 0.95, ncol=ncol, framealpha=0.9,
               loc="upper center", bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.0, handlelength=1.6, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _note_missing(fig, _missing_for(sub), y=style["pr_note_y"],
                  fontsize=fs * 0.8)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overview_f1(combined, datasets, metric_prefix, title_metric,
                     family_label, out_path, style):
    """One figure: x = dataset, grouped bars = frameworks, y = F1.

    Gives a single-glance comparison of every framework across a whole family.
    """
    col = f"{metric_prefix}_f1"
    sub = combined[combined["dataset"].isin(datasets)]
    datasets = [d for d in datasets if d in set(sub["dataset"])]
    frameworks = [fw for fw in FRAMEWORK_ORDER if fw in set(sub["framework"])]

    n_groups = len(datasets)
    n_fw = len(frameworks)
    group_w = 0.82
    bar_w = group_w / max(n_fw, 1)

    fs = style["font"]
    fig, ax = plt.subplots(figsize=(style["width"], style["f1_height"]))
    for i, fw in enumerate(frameworks):
        vals = []
        for ds in datasets:
            row = sub[(sub["framework"] == fw) & (sub["dataset"] == ds)]
            vals.append(row.iloc[0][col] if len(row) else np.nan)
        xs = [g - group_w / 2 + bar_w * (i + 0.5) for g in range(n_groups)]
        ax.bar(xs, vals, width=bar_w, label=fw,
               color=FRAMEWORK_COLORS[fw], edgecolor="white", linewidth=0.6,
               hatch=FRAMEWORK_HATCH[fw])
        for xv, v in zip(xs, vals):
            if pd.isna(v):
                _mark_na(ax, xv, fontsize=fs * 0.75)

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([disp(d) for d in datasets])
    ax.tick_params(axis="both", labelsize=fs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1", fontsize=fs)
    ax.set_xlabel("Dataset", fontsize=fs)
    ax.set_title(f"{family_label} — {title_metric} F1 across datasets",
                 fontsize=fs * 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ncol = style["ncol"] or n_fw
    ax.legend(fontsize=fs * 0.95, ncol=ncol, framealpha=0.9,
              loc="upper center", bbox_to_anchor=(0.5, style["f1_legend_y"]),
              columnspacing=1.0, handlelength=1.6, handletextpad=0.4)
    fig.tight_layout()
    _note_missing(fig, _missing_for(sub), y=style["f1_note_y"], ax=ax,
                  fontsize=fs * 0.8)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overview_stacked(combined, datasets, out_path, style):
    """Paper figure: F1, precision and recall as three equal stacked panels.

    The panels share one x axis (dataset labels on the bottom panel only) and
    one legend at the top; no title and no footnote, so the caption carries
    the explanation.
    """
    sub = combined[combined["dataset"].isin(datasets)]
    datasets = [d for d in datasets if d in set(sub["dataset"])]
    frameworks = [fw for fw in FRAMEWORK_ORDER if fw in set(sub["framework"])]

    n_groups = len(datasets)
    n_fw = len(frameworks)
    group_w = 0.82
    bar_w = group_w / max(n_fw, 1)

    fs = style["font"]
    fig, axes = plt.subplots(3, 1, sharex=True,
                             figsize=(style["width"], style["stack_height"]))
    for ax, part, ylabel in zip(axes, ["f1", "precision", "recall"],
                                ["F1", "Precision", "Recall"]):
        col = f"pairwise_{part}"
        for i, fw in enumerate(frameworks):
            vals = []
            for ds in datasets:
                row = sub[(sub["framework"] == fw) & (sub["dataset"] == ds)]
                vals.append(row.iloc[0][col] if len(row) else np.nan)
            xs = [g - group_w / 2 + bar_w * (i + 0.5) for g in range(n_groups)]
            ax.bar(xs, vals, width=bar_w, label=fw,
                   color=FRAMEWORK_COLORS[fw], edgecolor="white", linewidth=0.6,
                   hatch=FRAMEWORK_HATCH[fw])
            for xv, v in zip(xs, vals):
                if pd.isna(v):
                    _mark_na(ax, xv, fontsize=fs * 0.75)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel, fontsize=fs)
        ax.tick_params(axis="both", labelsize=fs)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    axes[-1].set_xticks(range(n_groups))
    axes[-1].set_xticklabels([disp(d) for d in datasets])
    axes[-1].set_xlabel("Dataset", fontsize=fs)
    handles, labels = axes[0].get_legend_handles_labels()
    ncol = style["ncol"] or n_fw
    fig.legend(handles, labels, fontsize=fs * 0.95, ncol=ncol, framealpha=0.9,
               loc="lower center", bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.0, handlelength=1.6, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# Overview figure geometry per family. The paper places the CCER overviews at
# full text width and the DER ones in a single column, so each family gets a
# canvas and font size that stay legible at that printed width.  ncol=None
# puts the whole legend on one line.
OVERVIEW_STYLE = {
    "CCER": dict(width=15, f1_height=4.6, pr_height=7.6, font=15, ncol=None,
                 f1_legend_y=-0.17, f1_note_y=-0.42, pr_note_y=-0.075,
                 stack_height=10.5),
    "DER":  dict(width=7, f1_height=4.6, pr_height=7.6, font=15, ncol=4,
                 f1_legend_y=-0.19, f1_note_y=-0.58, pr_note_y=-0.13,
                 stack_height=10.5),
}

# The stacked F1/precision/recall figures the paper includes (Figures 1 and 4),
# written next to this script under the names the paper uses.
STACKED_FIGURES = {"CCER": "results.pdf", "DER": "DERresults.pdf"}


def make_dataset_plots(combined, dataset, out_dir):
    df_ds = combined[combined["dataset"] == dataset]
    if df_ds.empty:
        print(f"  [skip] no data for {dataset}")
        return 0
    name = disp(dataset)
    plot_grouped(df_ds, dataset, "pairwise", "Pairwise",
                 os.path.join(out_dir, f"{name}_pairwise.png"))
    plot_grouped(df_ds, dataset, "bcubed", "B-Cubed",
                 os.path.join(out_dir, f"{name}_bcubed.png"))
    plot_f1_only(df_ds, dataset, "pairwise", "Pairwise",
                 os.path.join(out_dir, f"{name}_pairwise_f1.png"))
    plot_f1_only(df_ds, dataset, "bcubed", "B-Cubed",
                 os.path.join(out_dir, f"{name}_bcubed_f1.png"))
    print(f"  [ok] {dataset} -> {name}: 4 plots ({len(df_ds)} frameworks)")
    return 4


# ------------------------------------------------------------------- main ----
def main():
    os.makedirs(CCER_DIR, exist_ok=True)
    os.makedirs(DER_DIR, exist_ok=True)

    print("Loading eval CSVs ...")
    combined = load_all()
    combined.to_csv(os.path.join(OUT_ROOT, "combined_metrics.csv"), index=False)
    print(f"Combined table: {len(combined)} rows -> combined_metrics.csv\n")

    total = 0
    print("CCER plots ->", CCER_DIR)
    for ds in CCER_DATASETS:
        total += make_dataset_plots(combined, ds, CCER_DIR)
    print("\nDER plots ->", DER_DIR)
    for ds in DER_DATASETS:
        total += make_dataset_plots(combined, ds, DER_DIR)

    print("\nOverview figures (all datasets at a glance)")
    for prefix, tm in [("pairwise", "Pairwise"), ("bcubed", "B-Cubed")]:
        for family, datasets, out_dir in [("CCER", CCER_DATASETS, CCER_DIR),
                                          ("DER", DER_DATASETS, DER_DIR)]:
            style = OVERVIEW_STYLE[family]
            plot_overview_f1(combined, datasets, prefix, tm, family,
                             os.path.join(out_dir, f"overview_{prefix}_f1.png"),
                             style)
            plot_overview_pr(combined, datasets, prefix, tm, family,
                             os.path.join(out_dir, f"overview_{prefix}_pr.png"),
                             style)
            total += 2
    print("  [ok] 4 overview F1 figures + 4 precision/recall overviews "
          "(2 of each per family)")

    for family, datasets in [("CCER", CCER_DATASETS), ("DER", DER_DATASETS)]:
        plot_overview_stacked(combined, datasets,
                              os.path.join(OUT_ROOT, STACKED_FIGURES[family]),
                              OVERVIEW_STYLE[family])
        total += 1
    print(f"  [ok] paper figures: {', '.join(STACKED_FIGURES.values())}")

    print(f"\nDone. {total} plots written.")


if __name__ == "__main__":
    main()
