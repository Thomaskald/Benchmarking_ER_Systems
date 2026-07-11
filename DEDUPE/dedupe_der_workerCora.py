import sys
import io
import csv
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
import zope
zope.__path__.append("/home/it2022025/.local/lib/python3.10/site-packages/zope")
import dedupe

CORA_PATH  = "/home/it2022025/er_scalability/datasets/cora/cora.csv"
TRAIN_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/cora/train_set.csv"
VALID_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/cora/valid_set.csv"
TEST_PATH  = "/home/it2022025/er_scalability/train_validation_test_sets/cora/test_set.csv"

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)

# Cora training-sample caps (from the laptop script). Cora is small (~1295
# records) so prepare_training uses the FULL records dict (no record subsampling).
MAX_MATCHES  = 2500
MAX_DISTINCT = 7500


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def clean_text(text):
    if text is None:
        return None
    text = unidecode(str(text))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text if text else None


def load_cora(path):
    records = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")   # Cora uses |
        for row in reader:
            try:
                rid = int(row["Entity Id"])
            except (ValueError, TypeError):
                continue
            records[rid] = {
                "title":     clean_text(row.get("title")),
                "author":    clean_text(row.get("author")),
                "venue":     clean_text(row.get("venue")),
                "publisher": clean_text(row.get("publisher")),
                "year":      clean_text(row.get("year")),
            }
    return records


def load_split_pairs(path, records):
    pairs, labels = [], []
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        lid, rid = int(row["left_id"]), int(row["right_id"])
        if lid not in records or rid not in records:
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

    records = load_cora(CORA_PATH)
    train_pairs, train_labels = load_split_pairs(TRAIN_PATH, records)
    valid_pairs, valid_labels = load_split_pairs(VALID_PATH, records)
    test_pairs,  test_labels  = load_split_pairs(TEST_PATH,  records)

    fields = [
        dedupe.variables.Text("title"),
        dedupe.variables.Text("author"),
        dedupe.variables.String("venue", has_missing=True),
        dedupe.variables.String("publisher", has_missing=True),
        dedupe.variables.String("year", has_missing=True),
    ]
    deduper = dedupe.Dedupe(fields)   # single-source dirty ER

    matches, distinct = [], []
    for (id1, id2), label in zip(train_pairs, train_labels):
        if label == 1:
            matches.append((records[id1], records[id2]))
        else:
            distinct.append((records[id1], records[id2]))

    rng = random.Random(seed)
    # neg_ratio is searched; distinct count = neg_ratio * matches, but capped by
    # the CDDB OOM ceilings (MAX_MATCHES / MAX_DISTINCT).
    if len(matches) > MAX_MATCHES:
        matches = rng.sample(matches, MAX_MATCHES)
    neg_ratio = float(cfg["neg_ratio"])
    n_keep = int(round(len(matches) * neg_ratio))
    n_keep = min(n_keep, MAX_DISTINCT, len(distinct))
    if 0 < n_keep < len(distinct):
        distinct = rng.sample(distinct, n_keep)

    training_file = io.StringIO()
    json.dump({"match": matches, "distinct": distinct}, training_file)
    training_file.seek(0)

    # Cora is small -> use the full records dict for prepare_training
    deduper.prepare_training(records, training_file=training_file)
    deduper.train(
        recall=float(cfg["recall"]),
        index_predicates=bool(cfg["index_predicates"]),
    )

    def score_pairs(pairs):
        rec = [((lid, records[lid]), (rid, records[rid])) for (lid, rid) in pairs]
        scored = deduper.score(rec)
        sm = {}
        for (l, r), s in scored:
            sm[(int(l), int(r))] = float(s)
            sm[(int(r), int(l))] = float(s)
        return [sm.get((lid, rid), 0.0) for (lid, rid) in pairs]

    valid_scores = score_pairs(valid_pairs)
    test_scores  = score_pairs(test_pairs)

    best_t, best_valid_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        preds = [1 if s >= t else 0 for s in valid_scores]
        f1 = f1_score(valid_labels, preds, zero_division=0)
        if f1 > best_valid_f1:
            best_valid_f1, best_t = f1, float(t)

    preds_test = [1 if s >= best_t else 0 for s in test_scores]
    test_p = precision_score(test_labels, preds_test, zero_division=0)
    test_r = recall_score(test_labels, preds_test, zero_division=0)
    test_f1 = f1_score(test_labels, preds_test, zero_division=0)

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