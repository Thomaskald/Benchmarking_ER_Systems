"""
Shared framework identity for every plotting script: order, colours, hatches,
dataset naming.  Imported by make_plots.py, stats_analysis.py and
make_scalability_plots.py so all figures read the same way.
"""

import os

# Repo root (this file lives in <repo>/ER_PLOTS_AND_DIAGRAMS/ER_PLOTS_AND_DIAGRAMS/).
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# Fixed order, grouped by family: probabilistic / active learning, classic
# pair classifiers, pyJedAI's two workflows, deep learning.
FRAMEWORK_ORDER = ["Dedupe", "Splink", "Zingg", "Magellan", "RecordLinkage",
                   "pyJedAI-syn", "pyJedAI-sem",
                   "DeepER", "Sudowoodo", "LinkTransformer"]

# CVD-safe categorical palette, validated in FRAMEWORK_ORDER (adjacent pairs:
# worst CVD dE 9.1, worst normal-vision dE 18.7 on white).  The original seven
# keep their colours; pyJedAI-sem is a lighter step of pyJedAI's violet, and
# Sudowoodo a darker step of LinkTransformer's red (both transformer models).
FRAMEWORK_COLORS = {
    "Dedupe":          "#2a78d6",  # blue
    "Splink":          "#1baf7a",  # aqua
    "Zingg":           "#eda100",  # yellow
    "Magellan":        "#008300",  # green
    "RecordLinkage":   "#e87ba4",  # magenta
    "pyJedAI-syn":     "#4a3aa7",  # violet
    "pyJedAI-sem":     "#9085e9",  # light violet
    "DeepER":          "#eb6834",  # orange
    "Sudowoodo":       "#8f2d2c",  # dark red
    "LinkTransformer": "#e34948",  # red
}

# A second, redundant channel next to colour so bars survive greyscale / B&W
# print.  Single-character patterns; scripts may repeat them for density.
FRAMEWORK_HATCH = {
    "Dedupe":          "/",
    "Splink":          "\\",
    "Zingg":           "x",
    "Magellan":        ".",
    "RecordLinkage":   "-",
    "pyJedAI-syn":     "+",
    "pyJedAI-sem":     "o",
    "DeepER":          "*",
    "Sudowoodo":       "O",
    "LinkTransformer": "|",
}

CCER_DATASETS = ["D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
DER_DATASETS = ["CORA", "CDDB"]
# Frameworks that ran the two DER (dirty) datasets.  Magellan, RecordLinkage
# and LinkTransformer are two-source only.
DER_FRAMEWORKS = ["Dedupe", "Splink", "Zingg", "pyJedAI-syn", "pyJedAI-sem",
                  "DeepER", "Sudowoodo"]

# The CCER datasets are presented as D1..D8 (D2 -> D1, ... D9 -> D8).  Only
# naming changes; the underlying data is untouched.
DISPLAY_NAME = {f"D{i}": f"D{i - 1}" for i in range(2, 10)}


def disp(dataset):
    """Name a dataset is shown/saved under."""
    return DISPLAY_NAME.get(dataset, dataset)


# Why a framework has no result on a dataset (shown wherever a bar is missing).
TIME_LIMIT_NOTE = ("Sudowoodo: no configuration finished within the time limits "
                   "(90-min per-config cap, 5-h job walltime)")
