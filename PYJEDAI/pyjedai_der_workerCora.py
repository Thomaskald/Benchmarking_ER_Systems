import sys
import os
import json
import time
import resource
import warnings
from itertools import combinations

warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from pyjedai.datamodel import Data
from pyjedai.workflow import EmbeddingsNNWorkFlow
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.clustering import ConnectedComponentsClustering

CORA_PATH  = "/home/it2022025/er_scalability/datasets/cora/cora.csv"
GT_PATH    = "/home/it2022025/er_scalability/datasets/cora/cora_gt.csv"
TRAIN_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/cora/train_set.csv"
VALID_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/cora/valid_set.csv"
TEST_PATH  = "/home/it2022025/er_scalability/train_validation_test_sets/cora/test_set.csv"

ID_COL = "Entity Id"

def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def get_predicted_pairs(clusters, entity_ids):
    predicted = set()
    for cluster in clusters:
        for idx1, idx2 in combinations(sorted(cluster), 2):
            predicted.add((entity_ids[idx1], entity_ids[idx2]))
    return predicted

def score_split(split_df, predicted_pairs):
    y_true, y_pred = [], []
    for _, row in split_df.iterrows():
        lid, rid = str(row["left_id"]), str(row["right_id"])
        matched = (lid, rid) in predicted_pairs or (rid, lid) in predicted_pairs
        y_true.append(int(row["label"]))
        y_pred.append(1 if matched else 0)
    return (precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0))

def main():
    cfg = json.loads(sys.argv[1])
    t_start = time.time()
    cid = cfg.get("config_id", 0)

    cora_df = pd.read_csv(CORA_PATH, sep="|", engine="python", na_filter=False).astype(str)
    gt_df = pd.read_csv(GT_PATH, sep="|", header=None,
                        names=["left_id", "right_id"], engine="python")
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)

    data = Data(
        dataset_1=cora_df,
        id_column_name_1=ID_COL,
        ground_truth=gt_df,
    )

    # Build + run the embeddings-NN workflow for this sampled config.
    # vectorizer + similarity_search are constructor params; top_k +
    # similarity_distance are exec_params; threshold is sampled (no sweep).
    w = EmbeddingsNNWorkFlow(
        block_building=dict(
            method=EmbeddingsNNBlockBuilding,
            params=dict(
                vectorizer=cfg["vectorizer"],
                similarity_search="faiss",
            ),
            exec_params=dict(
                top_k=int(cfg["top_k"]),
                similarity_distance=cfg["similarity_distance"],
                load_embeddings_if_exist=False,
                save_embeddings=False,
            ),
        ),
        clustering=dict(
            method=ConnectedComponentsClustering,
            exec_params=dict(
                similarity_threshold=float(cfg["similarity_threshold"]),
            ),
        ),
        name=f"CORA-DER-cfg{cid}",
    )
    w.run(data, verbose=False)

    entity_ids = cora_df[ID_COL].tolist()
    predicted_pairs = get_predicted_pairs(w.clusters, entity_ids)

    valid_p, valid_r, valid_f1 = score_split(valid_df, predicted_pairs)
    test_p, test_r, test_f1 = score_split(test_df, predicted_pairs)

    # threshold is sampled, so the "curve" is the single operating point
    point = {"t": round(float(cfg["similarity_threshold"]), 4),
             "precision": round(test_p, 6), "recall": round(test_r, 6), "f1": round(test_f1, 6)}

    result = {
        "config_id": cid,
        "params": {"vectorizer": cfg["vectorizer"],
                   "similarity_distance": cfg["similarity_distance"],
                   "top_k": int(cfg["top_k"]),
                   "similarity_threshold": round(float(cfg["similarity_threshold"]), 4)},
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