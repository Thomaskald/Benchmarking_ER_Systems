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
import zope
zope.__path__.append("/home/it2022025/.local/lib/python3.10/site-packages/zope")
import dedupe

D1_PATH    = "/home/it2022025/er_scalability/datasets/D7/tmdb.csv"   # left_id side
D2_PATH    = "/home/it2022025/er_scalability/datasets/D7/tvdb.csv"   # right_id side
TRAIN_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/db7/train_set.csv"
VALID_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/db7/valid_set.csv"
TEST_PATH  = "/home/it2022025/er_scalability/train_validation_test_sets/db7/test_set.csv"

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
    O = "https://www.scads.de/movieBenchmark/ontology/"   # D5 ontology prefix
    data = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")     # D5 uses |
        for row in reader:
            try:
                data[int(str(row["id"]).strip())] = {
                    "title":   clean_text(row.get(O + "title") or row.get(O + "name")),
                    "year":    clean_text(row.get(O + "startYear") or row.get(O + "releaseDate")),
                    "genre":   clean_text(row.get(O + "genre_list")),
                    "runtime": clean_text(row.get(O + "runtimeMinutes")
                                          or row.get("http://dbpedia.org/ontology/runtime")),
                }
            except (KeyError, ValueError, TypeError):
                continue
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

    abt = load_csv(D1_PATH)
    buy = load_csv(D2_PATH)
    train_pairs, train_labels = load_split_pairs(TRAIN_PATH, abt, buy)
    valid_pairs, valid_labels = load_split_pairs(VALID_PATH, abt, buy)
    test_pairs,  test_labels  = load_split_pairs(TEST_PATH,  abt, buy)

    fields = [
        dedupe.variables.String("title"),
        dedupe.variables.String("year", has_missing=True),
        dedupe.variables.String("genre", has_missing=True),
        dedupe.variables.String("runtime", has_missing=True),
    ]
    deduper = dedupe.RecordLink(fields)

    matches, distinct = [], []
    for (lid, rid), label in zip(train_pairs, train_labels):
        if label == 1:
            matches.append((abt[lid], buy[rid]))
        else:
            distinct.append((abt[lid], buy[rid]))

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

    def score_pairs(pairs):
        rec = [((lid, abt[lid]), (rid, buy[rid])) for (lid, rid) in pairs]
        scored = deduper.score(rec)
        sm = {(int(l), int(r)): float(s) for (l, r), s in scored}
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