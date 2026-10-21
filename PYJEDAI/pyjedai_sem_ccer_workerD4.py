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
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.clustering import UniqueMappingClustering, ConnectedComponentsClustering

# SEMANTIC (embeddings-NN) workflow on CCER dataset D4.
#
# Structure is the CCER structure used by pyjedai_ccer_workerD2..D9.py --
# Data(two sources) -> clean_dataset -> block building -> clustering, with the
# components called directly. The ONLY substitution is the block-building step:
# StandardBlocking + BlockPurging + BlockFiltering + comparison cleaning +
# EntityMatching  ->  EmbeddingsNNBlockBuilding, which produces the similarity
# graph in one step (the embedding NN search IS the matching).
#
# From the semantic pipeline (pyjedai_der_worker{Cora,CDDB}.py):
#   - EmbeddingsNNBlockBuilding with similarity_search="faiss",
#   - every attribute embedded (no blocking-key attribute list; attributes_1/2 in
#     the syntactic workers are blocking keys, a syntactic-only concept),
#   - the decision threshold is a SAMPLED hyperparameter -- no validation sweep,
#     so pr_curve is the single operating point.

D1_PATH    = "/path/to/your/er_scalability/datasets/D4/dblp.csv"
D2_PATH    = "/path/to/your/er_scalability/datasets/D4/acm.csv"
VALID_PATH = "/path/to/your/er_scalability/train_validation_test_sets/db4/valid_set.csv"
TEST_PATH  = "/path/to/your/er_scalability/train_validation_test_sets/db4/test_set.csv"
DELIM      = "%"

ID_COL = "id"

# Clean-Clean ER admits 1-to-1 mapping, so both clusterers are searched
# (the DER semantic pipeline can only use connected components).
CLUSTERERS = {
    "UniqueMappingClustering": UniqueMappingClustering,
    "ConnectedComponentsClustering": ConnectedComponentsClustering,
}


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def predicted_pairs(clusters, d1, d2):
    """Clusters carry positional indices over d1 concatenated with d2.

    Same index split as pyjedai_ccer_workerD4.py: index < n1 belongs to source 1,
    index >= n1 belongs to source 2. Ids are left in their native dtype so they
    compare directly against the split files, exactly as the syntactic worker does.
    """
    n1 = len(d1)
    pred = set()
    for cl in clusters:
        ids = list(cl)
        a_ids = [i for i in ids if i < n1]
        b_ids = [i for i in ids if i >= n1]
        for a in a_ids:
            for b in b_ids:
                pred.add((d1.iloc[a][ID_COL], d2.iloc[b - n1][ID_COL]))
    return pred


def main():
    cfg = json.loads(sys.argv[1])
    t_start = time.time()
    cid = cfg.get("config_id", 0)

    d1 = pd.read_csv(D1_PATH, sep=DELIM, engine="python", na_filter=False)
    d2 = pd.read_csv(D2_PATH, sep=DELIM, engine="python", na_filter=False)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)

    data = Data(
        dataset_1=d1, id_column_name_1=ID_COL,
        dataset_2=d2, id_column_name_2=ID_COL,
    )
    data.clean_dataset(
        remove_stopwords=False, remove_punctuation=False,
        remove_numbers=False, remove_unicodes=False,
    )

    # --- semantic block building (replaces the whole syntactic blocking chain) ---
    emb = EmbeddingsNNBlockBuilding(
        vectorizer=cfg["vectorizer"],
        similarity_search="faiss",
    )
    # with_entity_matching=True makes build_blocks return (blocks, graph) instead of
    # just blocks -- the similarity graph is what the clusterers consume. This is
    # exactly what EmbeddingsNNWorkFlow passes internally (workflow.py: build_blocks
    # is called with with_entity_matching=True, then clustering runs on em_graph).
    blocks, pairs_graph = emb.build_blocks(
        data,
        top_k=int(cfg["top_k"]),
        similarity_distance=cfg["similarity_distance"],
        with_entity_matching=True,
        load_embeddings_if_exist=False,
        save_embeddings=False,
        tqdm_disable=True,
    )

    ccc = CLUSTERERS[cfg["clusterer"]]()
    clusters = ccc.process(pairs_graph, data,
                           similarity_threshold=float(cfg["similarity_threshold"]))
    pred = predicted_pairs(clusters, d1, d2)

    y_valid = valid_df["label"].astype(int).tolist()
    valid_keys = list(zip(valid_df["left_id"], valid_df["right_id"]))
    y_test = test_df["label"].astype(int).tolist()
    test_keys = list(zip(test_df["left_id"], test_df["right_id"]))

    y_pred_valid = [1 if k in pred else 0 for k in valid_keys]
    valid_f1 = f1_score(y_valid, y_pred_valid, zero_division=0)

    y_pred_test = [1 if k in pred else 0 for k in test_keys]
    test_p = precision_score(y_test, y_pred_test, zero_division=0)
    test_r = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)

    # threshold is sampled, so the "curve" is the single operating point
    point = {"t": round(float(cfg["similarity_threshold"]), 4),
             "precision": round(test_p, 6), "recall": round(test_r, 6),
             "f1": round(test_f1, 6)}

    result = {
        "config_id": cid,
        "params": {"vectorizer": cfg["vectorizer"],
                   "similarity_distance": cfg["similarity_distance"],
                   "top_k": int(cfg["top_k"]),
                   "similarity_threshold": round(float(cfg["similarity_threshold"]), 4),
                   "clusterer": cfg["clusterer"]},
        "status": "OK",
        "chosen_threshold": round(float(cfg["similarity_threshold"]), 4),
        "valid_f1_at_threshold": round(valid_f1, 6),
        "test_point": point,
        "pr_curve": [point],
        "time_sec": round(time.time() - t_start, 2),
        "peak_mem_mb": round(peak_mem_mb(), 1),
    }
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
