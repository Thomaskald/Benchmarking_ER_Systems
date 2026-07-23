#!/usr/bin/env python3
"""
Scalability comparison plots for the 4 DER frameworks
(Dedupe, Splink, Zingg, pyJedAI).

Produces four figures in ./SCALABILITY :

    1. effectiveness_vs_time.png   -> F1 vs wall-clock time   (cost-quality)
    2. effectiveness_vs_memory.png -> F1 vs peak memory       (cost-quality)
    3. effectiveness_vs_scale.png  -> F1 vs dataset size       (quality retention)
    4. runtime_vs_scale.png        -> runtime vs dataset size  (cost growth)
    5. memory_vs_scale.png         -> peak memory vs dataset size (cost growth)
    6. how_far_each_went.png       -> max dataset size reached (reach)

The "best config" scalability sweep starts reporting at 50K. Dedupe could not
complete 50K (timed out under the SLURM wall-clock limit) and only ever ran up
to 10K, so it has no points in plots 1-3 -- it is still kept in every legend and
gets its own bar in plot 4.

Source numbers live in /home/thomas/scalability_analysis/*_scalability_results.csv
Palette + framework order are inherited from the main thesis plots (CVD-safe).

Run:  /home/thomas/miniconda3/bin/python make_scalability_plots.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# ----------------------------------------------------------------- style ----
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12.5,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.8,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SCALABILITY")
os.makedirs(OUT_DIR, exist_ok=True)

# Fixed DER framework order + colors (shared with the main thesis plots).
FRAMEWORK_ORDER = ["Dedupe", "Splink", "Zingg", "pyJedAI"]
FRAMEWORK_COLORS = {
    "Dedupe":  "#2a78d6",  # blue
    "Splink":  "#1baf7a",  # aqua
    "Zingg":   "#eda100",  # yellow
    "pyJedAI": "#4a3aa7",  # violet
}
MARKERS = {"Dedupe": "D", "Splink": "o", "Zingg": "s", "pyJedAI": "^"}
# Line style + hatch give a second, non-colour channel (greyscale / B&W print).
LINESTYLES = {"Dedupe": (0, (1, 1)), "Splink": "-",
              "Zingg": (0, (5, 2)), "pyJedAI": (0, (3, 1, 1, 1))}
HATCHES = {"Dedupe": "//", "Splink": "\\\\", "Zingg": "xx", "pyJedAI": "++"}

# ---- scalability sweep data (best config, from 50K up) ---------------------
# time = load + workflow wall-clock seconds; mem = peak RSS in MB.
DATA = {
    "Splink": {
        "labels": ["50K", "100K", "200K", "300K", "1M"],
        "n":    [50_000, 100_000, 200_000, 300_000, 1_000_000],
        "f1":   [0.9653, 0.9619, 0.9589, 0.9573, 0.9517],
        "time": [1.47 + 18.37, 3.24 + 63.09, 6.74 + 210.29,
                 11.07 + 487.53, 49.48 + 10238.17],
        "mem":  [1147.1, 3584.2, 13586.7, 29983.0, 50122.7],
    },
    "pyJedAI": {
        "labels": ["50K", "100K", "200K", "300K"],
        "n":    [50_000, 100_000, 200_000, 300_000],
        "f1":   [0.6206, 0.6023, 0.5900, 0.6047],
        "time": [0.61 + 1137.28, 1.30 + 2335.79, 2.72 + 4840.92, 4.71 + 7539.19],
        "mem":  [1149.7, 1498.3, 2165.3, 2896.8],
    },
    "Zingg": {
        "labels": ["50K", "100K"],
        "n":    [50_000, 100_000],
        "f1":   [0.8548, 0.8484],
        "time": [2693.72, 6708.93],
        "mem":  [497.9, 836.2],
    },
    # Dedupe: no completed run >= 50K (timed out at 50K); max reached = 10K.
    "Dedupe": {"labels": [], "n": [], "f1": [], "time": [], "mem": []},
}

# max dataset size each framework successfully completed
MAX_REACHED = {"Splink": 1_000_000, "pyJedAI": 300_000,
               "Zingg": 100_000, "Dedupe": 10_000}

DEDUPE_NOTE = ("Dedupe not shown: did not scale past 10K\n"
               "(timed out at 50K, the first reported size)")


def human(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:g}M"
    if n >= 1_000:
        return f"{n/1_000:g}K"
    return str(int(n))


def full_legend(ax):
    """Legend listing all 4 frameworks; Dedupe flagged as <=10K only."""
    handles = []
    for fw in FRAMEWORK_ORDER:
        label = fw if fw != "Dedupe" else "Dedupe  (≤10K only)"
        handles.append(Line2D([0], [0], color=FRAMEWORK_COLORS[fw],
                              marker=MARKERS[fw], markersize=8, linewidth=2.2,
                              linestyle=LINESTYLES[fw],
                              markeredgecolor="white", markeredgewidth=0.8,
                              label=label))
    ax.legend(handles=handles, frameon=True, framealpha=0.95,
              edgecolor="#cccccc", loc="best", fontsize=11)


def cost_quality_plot(metric, xlabel, title, fname, logx=True):
    """Generic F1-vs-cost line plot (metric in {'time','mem'})."""
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for fw in FRAMEWORK_ORDER:
        d = DATA[fw]
        if not d["n"]:
            continue
        c = FRAMEWORK_COLORS[fw]
        ax.plot(d[metric], d["f1"], color=c, marker=MARKERS[fw],
                linestyle=LINESTYLES[fw],
                markersize=9, linewidth=2.2, markeredgecolor="white",
                markeredgewidth=0.9, zorder=3)
        # size label next to each marker
        for x, y, lab in zip(d[metric], d["f1"], d["labels"]):
            ax.annotate(lab, (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8.5,
                        color="#555555")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Effectiveness  (pairwise F1)")
    ax.set_title(title)
    ax.set_ylim(0.55, 1.0)
    ax.text(0.02, 0.03, DEDUPE_NOTE, transform=ax.transAxes, fontsize=9,
            style="italic", color="#777777", va="bottom", ha="left")
    full_legend(ax)
    fig.savefig(os.path.join(OUT_DIR, fname))
    plt.close(fig)
    print("wrote", fname)


def scale_plot():
    """F1 vs dataset size -- quality retention as data grows."""
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for fw in FRAMEWORK_ORDER:
        d = DATA[fw]
        if not d["n"]:
            continue
        c = FRAMEWORK_COLORS[fw]
        ax.plot(d["n"], d["f1"], color=c, marker=MARKERS[fw], markersize=9,
                linestyle=LINESTYLES[fw],
                linewidth=2.2, markeredgecolor="white", markeredgewidth=0.9,
                zorder=3)
        # label framework at the far (largest-N) end of its line
        ax.annotate(fw, (d["n"][-1], d["f1"][-1]), textcoords="offset points",
                    xytext=(8, 0), ha="left", va="center", fontsize=10.5,
                    fontweight="bold", color=c)
    ax.set_xscale("log")
    ax.set_xlabel("Dataset size  (number of records)")
    ax.set_ylabel("Effectiveness  (pairwise F1)")
    ax.set_title("Effectiveness vs. scalability")
    ax.set_ylim(0.55, 1.0)
    ax.set_xticks([50_000, 100_000, 200_000, 300_000, 1_000_000])
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: human(x)))
    ax.text(0.02, 0.03, DEDUPE_NOTE, transform=ax.transAxes, fontsize=9,
            style="italic", color="#777777", va="bottom", ha="left")
    full_legend(ax)
    fig.savefig(os.path.join(OUT_DIR, "effectiveness_vs_scale.png"))
    plt.close(fig)
    print("wrote effectiveness_vs_scale.png")


def cost_vs_scale_plot(metric, ylabel, title, fname, fmt):
    """Cost on the vertical axis, dataset size on the horizontal one.

    Same layout as effectiveness_vs_scale, but the y axis carries what the
    sweep actually costs: wall-clock time (metric='time') or peak memory
    (metric='mem').  Both grow fast, so y is log-scaled like x.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for fw in FRAMEWORK_ORDER:
        d = DATA[fw]
        if not d["n"]:
            continue
        c = FRAMEWORK_COLORS[fw]
        ax.plot(d["n"], d[metric], color=c, marker=MARKERS[fw], markersize=9,
                linestyle=LINESTYLES[fw],
                linewidth=2.2, markeredgecolor="white", markeredgewidth=0.9,
                zorder=3)
        # label framework at the far (largest-N) end of its line
        ax.annotate(fw, (d["n"][-1], d[metric][-1]), textcoords="offset points",
                    xytext=(8, 0), ha="left", va="center", fontsize=10.5,
                    fontweight="bold", color=c)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size  (number of records)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([50_000, 100_000, 200_000, 300_000, 1_000_000])
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: human(x)))
    ax.get_yaxis().set_major_formatter(FuncFormatter(fmt))
    ax.set_xlim(4e4, 2.2e6)
    # below the axes: the curves fill the plotting area, so an in-axes note
    # would sit on top of a series label
    ax.text(0.5, -0.17, DEDUPE_NOTE.replace("\n", " "), transform=ax.transAxes,
            fontsize=9, style="italic", color="#777777", va="top", ha="center")
    full_legend(ax)
    fig.savefig(os.path.join(OUT_DIR, fname))
    plt.close(fig)
    print("wrote", fname)


def reach_plot():
    """Horizontal bars: max dataset size each framework completed."""
    order = sorted(FRAMEWORK_ORDER, key=lambda fw: MAX_REACHED[fw])
    vals = [MAX_REACHED[fw] for fw in order]
    colors = [FRAMEWORK_COLORS[fw] for fw in order]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ys = range(len(order))
    bars = ax.barh(list(ys), vals, color=colors, edgecolor="white", height=0.62,
                   zorder=3, log=True)
    for bar, fw in zip(bars, order):
        bar.set_hatch(HATCHES[fw])
    for y, fw, v in zip(ys, order, vals):
        ax.text(v * 1.12, y, human(v), va="center", ha="left",
                fontsize=11.5, fontweight="bold", color=FRAMEWORK_COLORS[fw])
    ax.set_yticks(list(ys))
    ax.set_yticklabels(order, fontsize=12)
    ax.set_xlabel("Largest dataset completed  (records, log scale)")
    ax.set_title("How far each framework scaled")
    ax.set_xlim(5_000, 3_000_000)
    ax.grid(axis="y", visible=False)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: human(x)))
    ax.text(0.98, 0.06,
            "Dedupe stopped at 10K; the 50K+ sweep it could not complete.",
            transform=ax.transAxes, fontsize=9, style="italic",
            color="#777777", va="bottom", ha="right")
    fig.savefig(os.path.join(OUT_DIR, "how_far_each_went.png"))
    plt.close(fig)
    print("wrote how_far_each_went.png")


if __name__ == "__main__":
    cost_quality_plot("time", "Wall-clock time  (seconds, log scale)",
                      "Effectiveness vs. time", "effectiveness_vs_time.png")
    cost_quality_plot("mem", "Peak memory  (MB, log scale)",
                      "Effectiveness vs. memory", "effectiveness_vs_memory.png")
    scale_plot()
    cost_vs_scale_plot("time", "Wall-clock time  (seconds, log scale)",
                       "Runtime vs. dataset size", "runtime_vs_scale.png",
                       lambda y, _: f"{y:g}")
    cost_vs_scale_plot("mem", "Peak memory  (MB, log scale)",
                       "Memory consumption vs. dataset size",
                       "memory_vs_scale.png", lambda y, _: f"{y:g}")
    reach_plot()
    print("\nAll plots written to", OUT_DIR)
