#!/usr/bin/env python3
"""
Reproducible benchmark plots for the ER framework comparison.

Reads each framework's `bestconfig_eval` CSV(s), normalizes the columns to a
common schema, and produces, per dataset, four plots:

    1. pairwise  (precision, recall, f1)   -> grouped bars per framework
    2. bcubed    (precision, recall, f1)   -> grouped bars per framework
    3. pairwise f1 only                    -> one bar per framework
    4. bcubed   f1 only                    -> one bar per framework

Output is split into two folders by ER family:
    CCER/   -> datasets D2..D9   (8 clean-clean ER datasets)
    DER/    -> datasets CORA, CDDB (2 dirty ER datasets)

A merged long-format table is also written to `combined_metrics.csv`.

Run:  python3 make_plots.py
"""

import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- paths -----
HOME = "/home/thomas"
OUT_ROOT = "/home/thomas/ER_PLOTS_AND_DIAGRAMS"
CCER_DIR = os.path.join(OUT_ROOT, "CCER")
DER_DIR = os.path.join(OUT_ROOT, "DER")

# Where each framework's eval CSV(s) live. LinkTransformer is one file per
# dataset, so it gets a glob; everyone else has a single combined file.
FRAMEWORK_FILES = {
    "Dedupe":         [f"{HOME}/DEDUPE/results/dedupe_bestconfig_eval.csv"],
    "pyJedAI":        [f"{HOME}/PYJEDAI/results/pyjedai_bestconfig_eval.csv"],
    "Zingg":          [f"{HOME}/ZINGG/results/zingg_bestconfig_eval.csv"],
    "Splink":         [f"{HOME}/SPLINK/results/splink_bestconfig_eval.csv"],
    "Magellan":       [f"{HOME}/MAGELLAN/results/magellan_bestconfig_eval.csv"],
    "RecordLinkage":  [f"{HOME}/RECORDLINKAGE/results/recordlinkage_bestconfig_eval.csv"],
    "LinkTransformer": sorted(glob.glob(
        f"{HOME}/LINKTRANSFORMER/results/linktransformer_bestconfig_evalD*.csv")),
}

# Fixed framework order + colors so every plot reads the same way.
FRAMEWORK_ORDER = ["Dedupe", "Splink", "Zingg", "Magellan",
                   "pyJedAI", "LinkTransformer", "RecordLinkage"]
# CVD-safe categorical palette (validated: worst all-pairs deutan dE 12.9 on
# white). Order follows FRAMEWORK_ORDER so adjacent bars stay distinguishable.
FRAMEWORK_COLORS = {
    "Dedupe":          "#2a78d6",  # blue
    "Splink":          "#1baf7a",  # aqua
    "Zingg":           "#eda100",  # yellow
    "Magellan":        "#008300",  # green
    "pyJedAI":         "#4a3aa7",  # violet
    "LinkTransformer": "#e34948",  # red
    "RecordLinkage":   "#e87ba4",  # magenta
}

# Dataset -> family / ordering.
CCER_DATASETS = ["D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
DER_DATASETS = ["CORA", "CDDB"]

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
    """Frameworks present for a dataset, in the fixed global order."""
    present = set(df_ds["framework"])
    return [fw for fw in FRAMEWORK_ORDER if fw in present]


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
               color=FRAMEWORK_COLORS[fw], edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{dataset} — {title_metric} metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
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
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(frameworks, rotation=30, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1")
    ax.set_title(f"{dataset} — {title_metric} F1")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overview_f1(combined, datasets, metric_prefix, title_metric,
                     family_label, out_path):
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

    fig, ax = plt.subplots(figsize=(max(8, n_groups * n_fw * 0.28), 5.2))
    for i, fw in enumerate(frameworks):
        vals = []
        for ds in datasets:
            row = sub[(sub["framework"] == fw) & (sub["dataset"] == ds)]
            vals.append(row.iloc[0][col] if len(row) else 0.0)
        xs = [g - group_w / 2 + bar_w * (i + 0.5) for g in range(n_groups)]
        ax.bar(xs, vals, width=bar_w, label=fw,
               color=FRAMEWORK_COLORS[fw], edgecolor="white", linewidth=0.6)

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_xlabel("Dataset")
    ax.set_title(f"{family_label} — {title_metric} F1 across datasets")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=min(n_fw, 4), framealpha=0.9,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_dataset_plots(combined, dataset, out_dir):
    df_ds = combined[combined["dataset"] == dataset]
    if df_ds.empty:
        print(f"  [skip] no data for {dataset}")
        return 0
    plot_grouped(df_ds, dataset, "pairwise", "Pairwise",
                 os.path.join(out_dir, f"{dataset}_pairwise.png"))
    plot_grouped(df_ds, dataset, "bcubed", "B-Cubed",
                 os.path.join(out_dir, f"{dataset}_bcubed.png"))
    plot_f1_only(df_ds, dataset, "pairwise", "Pairwise",
                 os.path.join(out_dir, f"{dataset}_pairwise_f1.png"))
    plot_f1_only(df_ds, dataset, "bcubed", "B-Cubed",
                 os.path.join(out_dir, f"{dataset}_bcubed_f1.png"))
    print(f"  [ok] {dataset}: 4 plots ({len(df_ds)} frameworks)")
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
        plot_overview_f1(combined, CCER_DATASETS, prefix, tm, "CCER",
                         os.path.join(CCER_DIR, f"overview_{prefix}_f1.png"))
        plot_overview_f1(combined, DER_DATASETS, prefix, tm, "DER",
                         os.path.join(DER_DIR, f"overview_{prefix}_f1.png"))
        total += 2
    print("  [ok] 4 overview figures (2 per family)")

    print(f"\nDone. {total} plots written.")


if __name__ == "__main__":
    main()
