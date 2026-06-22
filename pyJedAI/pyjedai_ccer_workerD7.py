import sys
import os
import json
import time
import resource
import warnings
import logging

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
from pyjedai.clustering import UniqueMappingClustering, ConnectedComponentsClustering

D1_PATH    = "/home/it2022025/er_scalability/datasets/D7/tmdb.csv"
D2_PATH    = "/home/it2022025/er_scalability/datasets/D7/tvdb.csv"
VALID_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/db7/valid_set.csv"
TEST_PATH  = "/home/it2022025/er_scalability/train_validation_test_sets/db7/test_set.csv"
DELIM      = "|"          # D7 uses | as separator
# D5 (IMDB-TMDB) blocks on TWO ontology-URI attributes:
BLOCK_ATTRS = [
    "https://www.scads.de/movieBenchmark/ontology/title",
    "https://www.scads.de/movieBenchmark/ontology/name",
]

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)

# Comparison-cleaning (meta-blocking) algorithms searched for CCER.
# Edge-centric (WEP, CEP) accept the standard weighting schemes;
# node-centric (CNP) accepts only the normalized/EJS family.
COMPARISON_CLEANERS = {
    "CardinalityEdgePruning": CardinalityEdgePruning,
    "CardinalityNodePruning": CardinalityNodePruning,
}
# Clustering algorithms for Clean-Clean ER (both take similarity_threshold).
CLUSTERERS = {
    "UniqueMappingClustering": UniqueMappingClustering,
    "ConnectedComponentsClustering": ConnectedComponentsClustering,
}


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def predicted_pairs_at_threshold(clusterer_name, pairs_graph, data, d1, d2, t):
    cclass = CLUSTERERS[clusterer_name]
    ccc = cclass()
    clusters = ccc.process(pairs_graph, data, similarity_threshold=t)
    pred = set()
    n1 = len(d1)
    for cl in clusters:
        ids = list(cl)
        a_ids = [i for i in ids if i < n1]
        b_ids = [i for i in ids if i >= n1]
        for a in a_ids:
            for b in b_ids:
                pred.add((d1.iloc[a]["id"], d2.iloc[b - n1]["id"]))
    return pred


def main():
    cfg = json.loads(sys.argv[1])
    t_start = time.time()

    d1 = pd.read_csv(D1_PATH, sep=DELIM, engine="python", na_filter=False)
    d2 = pd.read_csv(D2_PATH, sep=DELIM, engine="python", na_filter=False)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)

    data = Data(
        dataset_1=d1, id_column_name_1="id",
        dataset_2=d2, id_column_name_2="id",
    )
    data.clean_dataset(
        remove_stopwords=False, remove_punctuation=False,
        remove_numbers=False, remove_unicodes=False,
    )

    bb = StandardBlocking()
    blocks = bb.build_blocks(data, attributes_1=BLOCK_ATTRS, attributes_2=BLOCK_ATTRS)

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

    y_valid = valid_df["label"].astype(int).tolist()
    valid_keys = list(zip(valid_df["left_id"], valid_df["right_id"]))
    y_test = test_df["label"].astype(int).tolist()
    test_keys = list(zip(test_df["left_id"], test_df["right_id"]))

    clusterer_name = cfg["clusterer"]
    pred_cache = {}
    def preds_at(t):
        if t not in pred_cache:
            pred_cache[t] = predicted_pairs_at_threshold(
                clusterer_name, pairs_graph, data, d1, d2, float(t))
        return pred_cache[t]

    # threshold (clustering similarity_threshold) chosen on VALIDATION
    best_t, best_valid_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        pred = preds_at(float(t))
        y_pred = [1 if k in pred else 0 for k in valid_keys]
        f1 = f1_score(y_valid, y_pred, zero_division=0)
        if f1 > best_valid_f1:
            best_valid_f1, best_t = f1, float(t)

    pred_best = preds_at(best_t)
    y_pred_test = [1 if k in pred_best else 0 for k in test_keys]
    test_p = precision_score(y_test, y_pred_test, zero_division=0)
    test_r = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)

    curve = []
    for t in THRESHOLD_GRID:
        pred = preds_at(float(t))
        y_pred = [1 if k in pred else 0 for k in test_keys]
        curve.append({"t": round(float(t), 3),
                      "precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
                      "recall": round(recall_score(y_test, y_pred, zero_division=0), 6),
                      "f1": round(f1_score(y_test, y_pred, zero_division=0), 6)})

    result = {
        "config_id": cfg.get("config_id"),
        "params": {k: cfg[k] for k in
                   ["ratio", "comparison_cleaner", "weighting_scheme",
                    "vectorizer", "metric", "tokenizer", "qgram", "clusterer"]},
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