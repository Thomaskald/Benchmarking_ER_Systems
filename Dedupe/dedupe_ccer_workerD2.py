"""
dedupe_ccer_workerD2.py
-------------------------------
Runs ONE Dedupe configuration on D2 (Abt-Buy, two-source -> RecordLink).

Searched params (sampled by the harness):
    neg_ratio          negatives per positive in the labelled training set
                       (Dedupe's dominant effectiveness lever; controls only
                        how Dedupe consumes the SHARED train split)
    recall             train(): proportion of true-dupe pairs the blocking
                       predicates must cover
    index_predicates   train(): whether to consider index predicates

The decision threshold is SWEPT here (not sampled): Dedupe outputs a match
score per pair; we sweep the cutoff to trace the full precision/recall curve.

sample_size / blocked_proportion are NOT searched: a pilot showed they have
no effect on the result in supervised mode (identical output across values).

Fairness: train / valid / test splits are identical across all frameworks.
neg_ratio only reshapes how Dedupe uses the TRAIN split; valid/test untouched.

Output: a single 'RESULT_JSON:' line on stdout.
"""

import sys
import io
import json
import time
import re
import random
import resource
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
from unidecode import unidecode
from sklearn.metrics import precision_score, recall_score, f1_score
import dedupe

# -------------------------------------------------------
# PATHS  (D2 Abt-Buy)
# -------------------------------------------------------
ABT_PATH   = "/home/thomas/pyJedAI/data/ccer/D2/abt.csv"
BUY_PATH   = "/home/thomas/pyJedAI/data/ccer/D2/buy.csv"
TRAIN_PATH = "/home/thomas/train_test_valid_datasets/db2/train_set.csv"
VALID_PATH = "/home/thomas/train_test_valid_datasets/db2/valid_set.csv"
TEST_PATH  = "/home/thomas/train_test_valid_datasets/db2/test_set.csv"

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def clean_text(text):
    if text is None:
        return None
    text = unidecode(str(text))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text if text else None


def preprocess_price(price):
    if not price:
        return None
    try:
        return float(str(price).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None


def load_csv(path):
    import csv
    data = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            data[int(row["id"])] = {
                "name":        clean_text(row.get("name")),
                "description": clean_text(row.get("description")),
                "price":       preprocess_price(row.get("price")),
            }
    return data


def load_split_pairs(path, abt, buy):
    pairs, labels = [], []
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        lid, rid = int(row["left_id"]), int(row["right_id"])
        if abt.get(lid) is None or buy.get(rid) is None:
            continue
        pairs.append((lid, rid))
        labels.append(int(row["label"]))
    return pairs, labels


def main():
    cfg = json.loads(sys.argv[1])
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    t_start = time.time()

    abt = load_csv(ABT_PATH)
    buy = load_csv(BUY_PATH)
    train_pairs, train_labels = load_split_pairs(TRAIN_PATH, abt, buy)
    valid_pairs, valid_labels = load_split_pairs(VALID_PATH, abt, buy)
    test_pairs,  test_labels  = load_split_pairs(TEST_PATH,  abt, buy)

    fields = [
        dedupe.variables.String("name"),
        dedupe.variables.Text("description"),
        dedupe.variables.Price("price", has_missing=True),
    ]
    deduper = dedupe.RecordLink(fields)

    # Build match / distinct from the SHARED train split
    matches, distinct = [], []
    for (lid, rid), label in zip(train_pairs, train_labels):
        if label == 1:
            matches.append((abt[lid], buy[rid]))
        else:
            distinct.append((abt[lid], buy[rid]))

    # neg_ratio: keep at most neg_ratio negatives per positive (searched knob)
    neg_ratio = float(cfg["neg_ratio"])
    rng = random.Random(seed)
    n_keep = int(min(len(distinct), round(len(matches) * neg_ratio)))
    if 0 < n_keep < len(distinct):
        distinct = rng.sample(distinct, n_keep)

    training_file = io.StringIO()
    json.dump({"match": matches, "distinct": distinct}, training_file)
    training_file.seek(0)

    deduper.prepare_training(abt, buy, training_file=training_file)
    deduper.train(
        recall=float(cfg["recall"]),
        index_predicates=bool(cfg["index_predicates"]),
    )

    # Score VALID and TEST pairs
    def score_pairs(pairs):
        rec = [((lid, abt[lid]), (rid, buy[rid])) for (lid, rid) in pairs]
        scored = deduper.score(rec)
        sm = {(int(l), int(r)): float(s) for (l, r), s in scored}
        return [sm.get((lid, rid), 0.0) for (lid, rid) in pairs]

    valid_scores = score_pairs(valid_pairs)
    test_scores  = score_pairs(test_pairs)

    # 1) choose threshold on VALIDATION
    best_t, best_valid_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        preds = [1 if s >= t else 0 for s in valid_scores]
        f1 = f1_score(valid_labels, preds, zero_division=0)
        if f1 > best_valid_f1:
            best_valid_f1, best_t = f1, float(t)

    # 2) report TEST at that fixed threshold
    preds_test = [1 if s >= best_t else 0 for s in test_scores]
    test_p = precision_score(test_labels, preds_test, zero_division=0)
    test_r = recall_score(test_labels, preds_test, zero_division=0)
    test_f1 = f1_score(test_labels, preds_test, zero_division=0)

    # 3) full TEST curve for the frontier
    curve = []
    for t in THRESHOLD_GRID:
        preds = [1 if s >= t else 0 for s in test_scores]
        curve.append({"t": round(float(t), 3),
                      "precision": round(precision_score(test_labels, preds, zero_division=0), 6),
                      "recall": round(recall_score(test_labels, preds, zero_division=0), 6),
                      "f1": round(f1_score(test_labels, preds, zero_division=0), 6)})

    result = {
        "config_id": cfg.get("config_id"),
        "params": {k: cfg[k] for k in ["neg_ratio", "recall", "index_predicates"]},
        "status": "OK",
        "chosen_threshold": round(best_t, 3),
        "valid_f1_at_threshold": round(best_valid_f1, 6),
        "test_point": {"t": round(best_t, 3), "precision": round(test_p, 6),
                       "recall": round(test_r, 6), "f1": round(test_f1, 6)},
        "pr_curve": curve,
        "time_sec": round(time.time() - t_start, 2),
        "peak_mem_mb": round(peak_mem_mb(), 1),
    }
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()