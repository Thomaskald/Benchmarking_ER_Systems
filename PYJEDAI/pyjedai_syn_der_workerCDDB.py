import sys
import os
import json
import time
import resource
import warnings
import logging
from itertools import combinations

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

import pyjedai
from pyjedai.datamodel import Data
from pyjedai.block_building import StandardBlocking
from pyjedai.block_cleaning import BlockPurging, BlockFiltering
from pyjedai.comparison_cleaning import (
    CardinalityEdgePruning, CardinalityNodePruning,
)
from pyjedai.matching import EntityMatching
from pyjedai.clustering import ConnectedComponentsClustering

# SYNTACTIC workflow on DER dataset CDDB.
#
# This is the method used by pyjedai_ccer_workerD2..D9.py -- StandardBlocking ->
# BlockPurging -> BlockFiltering -> comparison cleaning -> EntityMatching ->
# clustering, with the 101-point decision-threshold sweep chosen on VALIDATION
# (the syntactic approach keeps its threshold protocol; the semantic one samples).
#
# Dataset handling comes from the DER side (pyjedai_der_workerCDDB.py):
#   - a single dirty source plus its ground-truth file,
#   - Data(dataset_1=..., ground_truth=...) with no dataset_2,
#   - pairs are all combinations WITHIN a cluster (no two-source index split),
#   - ids compared as strings, matching the DER worker.
#
# ConnectedComponentsClustering only: UniqueMappingClustering enforces 1-to-1
# matching, which is meaningful for Clean-Clean ER and not for Dirty ER.

SRC_PATH   = "/path/to/your/er_scalability/datasets/CDDB/cddb.csv"
GT_PATH    = "/path/to/your/er_scalability/datasets/CDDB/gt.csv"
VALID_PATH = "/path/to/your/er_scalability/train_validation_test_sets/cddb/valid_set.csv"
TEST_PATH  = "/path/to/your/er_scalability/train_validation_test_sets/cddb/test_set.csv"

ID_COL      = "id"
BLOCK_ATTRS = ["artist", "title", "year"]

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)

COMPARISON_CLEANERS = {
    "CardinalityEdgePruning": CardinalityEdgePruning,
    "CardinalityNodePruning": CardinalityNodePruning,
}


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def predicted_pairs_at_threshold(pairs_graph, data, entity_ids, t):
    """Dirty ER: every pair inside a cluster is a predicted match."""
    ccc = ConnectedComponentsClustering()
    clusters = ccc.process(pairs_graph, data, similarity_threshold=t)
    pred = set()
    for cluster in clusters:
        for i1, i2 in combinations(sorted(cluster), 2):
            pred.add((str(entity_ids[i1]), str(entity_ids[i2])))
    return pred


def score_split(split_df, pred):
    y_true, y_pred = [], []
    for _, row in split_df.iterrows():
        lid, rid = str(row["left_id"]), str(row["right_id"])
        matched = (lid, rid) in pred or (rid, lid) in pred
        y_true.append(int(row["label"]))
        y_pred.append(1 if matched else 0)
    return (precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0))


def main():
    cfg = json.loads(sys.argv[1])
    t_start = time.time()

    df = pd.read_csv(SRC_PATH, engine="python", na_filter=False).astype(str)
    gt_df = pd.read_csv(GT_PATH, engine="python")

    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)

    data = Data(
        dataset_1=df,
        id_column_name_1=ID_COL,
        ground_truth=gt_df,
    )
    data.clean_dataset(
        remove_stopwords=False, remove_punctuation=False,
        remove_numbers=False, remove_unicodes=False,
    )

    bb = StandardBlocking()
    blocks = bb.build_blocks(data, attributes_1=BLOCK_ATTRS)

    bp = BlockPurging()
    cleaned = bp.process(blocks, data, tqdm_disable=True)

    bf = BlockFiltering(ratio=cfg["ratio"])
    filtered = bf.process(cleaned, data, tqdm_disable=True)

    # Comparison cleaning (meta-blocking) -- algorithm is SEARCHED
    cc_class = COMPARISON_CLEANERS[cfg["comparison_cleaner"]]
    mb = cc_class(weighting_scheme=cfg["weighting_scheme"])
    candidates = mb.process(filtered, data, tqdm_disable=True)

    em = EntityMatching(
        metric=cfg["metric"],
        tokenizer=cfg["tokenizer"],
        vectorizer=cfg["vectorizer"],
        qgram=cfg["qgram"],
        similarity_threshold=0.0,
    )
    pairs_graph = em.predict(candidates, data, tqdm_disable=True)

    entity_ids = df[ID_COL].tolist()
    pred_cache = {}
    def preds_at(t):
        if t not in pred_cache:
            pred_cache[t] = predicted_pairs_at_threshold(
                pairs_graph, data, entity_ids, float(t))
        return pred_cache[t]

    # threshold (clustering similarity_threshold) chosen on VALIDATION
    best_t, best_valid_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        _, _, f1 = score_split(valid_df, preds_at(float(t)))
        if f1 > best_valid_f1:
            best_valid_f1, best_t = f1, float(t)

    test_p, test_r, test_f1 = score_split(test_df, preds_at(best_t))

    curve = []
    for t in THRESHOLD_GRID:
        p, r, f1 = score_split(test_df, preds_at(float(t)))
        curve.append({"t": round(float(t), 3), "precision": round(p, 6),
                      "recall": round(r, 6), "f1": round(f1, 6)})

    result = {
        "config_id": cfg.get("config_id"),
        "params": {k: cfg[k] for k in
                   ["ratio", "comparison_cleaner", "weighting_scheme",
                    "vectorizer", "metric", "tokenizer", "qgram"]},
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
