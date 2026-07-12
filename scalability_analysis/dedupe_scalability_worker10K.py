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
from sklearn.metrics import precision_score, recall_score, f1_score
import zope
zope.__path__.append("/home/it2022025/.local/lib/python3.10/site-packages/zope")
import dedupe

# Synthetic FEBRL 10K (scalability tuning size). Single-source dirty ER.
PROFILES_PATH = "/home/it2022025/er_scalability/converted/10K/profiles.csv"
TRAIN_PATH    = "/home/it2022025/er_scalability/splits/10K/train_set.csv"
VALID_PATH    = "/home/it2022025/er_scalability/splits/10K/valid_set.csv"
TEST_PATH     = "/home/it2022025/er_scalability/splits/10K/test_set.csv"

# End-to-end operating-point grid. This threshold is the dedupe.partition()
# CLUSTERING cut (not a raw pairwise-score cut) -- the same knob the fixed
# scalability run freezes. Start at 0.05 to avoid the degenerate low-threshold
# case where nearly all records merge into one cluster.
THRESHOLD_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 3)

# prepare_training record sample (from the scalability script: 10K -> 2000)
SAMPLE_SIZE = 2000


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def clean_text(text):
    if text is None or str(text).strip() in ("", "nan"):
        return None
    text = re.sub(r"\s+", " ", str(text)).strip().lower()
    return text if text else None


def load_records(path):
    df = pd.read_csv(path, engine="python", na_filter=False).astype(str)
    records = {}
    for _, row in df.iterrows():
        rid = int(row["id"])
        records[rid] = {
            "given_name":    clean_text(row.get("given_name")),
            "surname":       clean_text(row.get("surname")),
            "address_1":     clean_text(row.get("address_1")),
            "suburb":        clean_text(row.get("suburb")),
            "postcode":      clean_text(row.get("postcode")),
            "state":         clean_text(row.get("state")),
            "date_of_birth": clean_text(row.get("date_of_birth")),
            "soc_sec_id":    clean_text(row.get("soc_sec_id")),
            "phone_number":  clean_text(row.get("phone_number")),
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


def cluster_id_map(clusters):
    """record_id -> cluster index. Records absent from `clusters` (singletons
    that dedupe did not link) are simply not in the map."""
    cid = {}
    for k, (ids, _scores) in enumerate(clusters):
        for x in ids:
            cid[int(x)] = k
    return cid


def eval_split(pairs, labels, cid):
    """A pair is a predicted match iff BOTH records are in the SAME cluster.
    Defaults -1/-2 guarantee two unclustered records never count as a match."""
    y_pred = [1 if cid.get(a, -1) == cid.get(b, -2) else 0 for (a, b) in pairs]
    return (precision_score(labels, y_pred, zero_division=0),
            recall_score(labels, y_pred, zero_division=0),
            f1_score(labels, y_pred, zero_division=0))


def main():
    cfg = json.loads(sys.argv[1])
    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed)
    t_start = time.time()

    records = load_records(PROFILES_PATH)
    train_pairs, train_labels = load_split_pairs(TRAIN_PATH, records)
    valid_pairs, valid_labels = load_split_pairs(VALID_PATH, records)
    test_pairs,  test_labels  = load_split_pairs(TEST_PATH,  records)

    # FEBRL field types from the validated scalability script
    fields = [
        dedupe.variables.Text("given_name", has_missing=True),
        dedupe.variables.Text("surname", has_missing=True),
        dedupe.variables.Text("address_1", has_missing=True),
        dedupe.variables.String("suburb", has_missing=True),
        dedupe.variables.String("postcode", has_missing=True),
        dedupe.variables.String("state", has_missing=True),
        dedupe.variables.String("date_of_birth", has_missing=True),
        dedupe.variables.String("soc_sec_id", has_missing=True),
        dedupe.variables.String("phone_number", has_missing=True),
    ]
    deduper = dedupe.Dedupe(fields)   # single-source dirty ER

    matches, distinct = [], []
    for (id1, id2), label in zip(train_pairs, train_labels):
        if label == 1:
            matches.append((records[id1], records[id2]))
        else:
            distinct.append((records[id1], records[id2]))

    rng = random.Random(seed)
    # neg_ratio searched: distinct count = neg_ratio * matches
    neg_ratio = float(cfg["neg_ratio"])
    n_keep = int(round(len(matches) * neg_ratio))
    n_keep = min(n_keep, len(distinct))
    if 0 < n_keep < len(distinct):
        distinct = rng.sample(distinct, n_keep)

    training_file = io.StringIO()
    json.dump({"match": matches, "distinct": distinct}, training_file)
    training_file.seek(0)

    # sampled records for prepare_training (10K -> 2000), as in the scalability script
    sample_keys = rng.sample(list(records.keys()), min(SAMPLE_SIZE, len(records)))
    sample_records = {k: records[k] for k in sample_keys}

    deduper.prepare_training(sample_records, training_file=training_file,
                             sample_size=5000, blocked_proportion=0.9)
    # train() learns BOTH the pairwise classifier AND the blocking predicates.
    # recall / index_predicates are blocking knobs -- they only take effect
    # because we now actually block below (via pairs()).
    deduper.train(
        recall=float(cfg["recall"]),
        index_predicates=bool(cfg["index_predicates"]),
    )

    # ----- END-TO-END EVALUATION (mirrors dedupe.partition) -----
    # Block over ALL records with the learned predicates, score the candidate
    # pairs ONCE, then re-cluster at each threshold. A valid/test pair is
    # recalled only if it survives Dedupe's own blocking AND clustering --
    # the same bar Splink / pyJedAI / Zingg clear.
    candidate_pairs = deduper.pairs(records)
    pair_scores = deduper.score(candidate_pairs)

    best_t, best_valid_f1 = float(THRESHOLD_GRID[0]), -1.0
    curve = []
    test_at = {}
    for t in THRESHOLD_GRID:
        t = float(t)
        clusters = deduper.cluster(pair_scores, threshold=t)
        cid = cluster_id_map(clusters)

        v_f1 = eval_split(valid_pairs, valid_labels, cid)[2]
        if v_f1 > best_valid_f1:
            best_valid_f1, best_t = v_f1, t

        tp_, tr_, tf_ = eval_split(test_pairs, test_labels, cid)
        curve.append({"t": round(t, 3), "precision": round(tp_, 6),
                      "recall": round(tr_, 6), "f1": round(tf_, 6)})
        test_at[round(t, 3)] = (tp_, tr_, tf_)

    test_p, test_r, test_f1 = test_at[round(best_t, 3)]

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
