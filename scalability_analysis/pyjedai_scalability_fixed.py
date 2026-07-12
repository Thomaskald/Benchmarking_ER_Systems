"""
pyjedai_scalability_fixed.py
----------------------------
Fixed-config scalability runs for pyJedAI on synthetic FEBRL datasets.
Uses the single BEST config chosen from the 10K B=50 search:
    vectorizer=sminilm, similarity_distance=cosine, top_k=14, threshold=0.7698

No random search, no B -- one fixed config applied to each size.
10K is excluded (it was the tuning size).

Sizes: 50K, 100K, 200K, 300K, 1M, 2M
Metrics per size: runtime, peak memory, and test P/R/F1 (scored over the
split-file labelled pairs, consistent with all other tools).
Failure policy: OOM / TIMEOUT / ERROR are recorded, not crashes.
"""

import time
import traceback
import warnings
import logging
import os
import resource
from itertools import combinations

import pandas as pd
import psutil
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from pyjedai.datamodel import Data
from pyjedai.workflow import EmbeddingsNNWorkFlow
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.clustering import ConnectedComponentsClustering

CONVERTED_DIR = "/home/it2022025/er_scalability/converted"
SPLITS_DIR    = "/home/it2022025/er_scalability/splits"
OUTPUT_CSV    = "/home/it2022025/er_scalability/scalability/pyjedai_scalability_results.csv"

# 10K excluded -- it was the tuning size. Rest of the ladder:
DATASETS = ["50K", "100K", "200K", "300K", "1M", "2M"]

EMBEDDING_ATTRS = ["given_name", "surname", "address_1", "suburb", "postcode"]

# ---- BEST CONFIG from the 10K B=50 search (fixed for all sizes) ----
BEST_VECTORIZER = "sminilm"
BEST_DISTANCE   = "cosine"
BEST_TOP_K      = 2
BEST_THRESHOLD  = 0.7656


def mem_mb():
    return psutil.Process().memory_info().rss / 1024 ** 2

def peak_mem_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return mem_mb()


def get_predicted_pairs(clusters, entity_ids):
    predicted = set()
    for cluster in clusters:
        for i1, i2 in combinations(sorted(cluster), 2):
            predicted.add((entity_ids[i1], entity_ids[i2]))
    return predicted


def score_split(split_df, predicted_pairs):
    # score over the split-file labelled pairs (consistent with all other tools)
    y_true, y_pred = [], []
    for _, row in split_df.iterrows():
        lid, rid = str(row["left_id"]), str(row["right_id"])
        matched = (lid, rid) in predicted_pairs or (rid, lid) in predicted_pairs
        y_true.append(int(row["label"]))
        y_pred.append(1 if matched else 0)
    return (precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0))


def run_pipeline(df):
    total_start = time.time()
    data = Data(dataset_1=df, id_column_name_1="id")
    w = EmbeddingsNNWorkFlow(
        block_building=dict(
            method=EmbeddingsNNBlockBuilding,
            params=dict(vectorizer=BEST_VECTORIZER, similarity_search="faiss"),
            attributes_1=EMBEDDING_ATTRS,
            exec_params=dict(
                top_k=BEST_TOP_K,
                similarity_distance=BEST_DISTANCE,
                load_embeddings_if_exist=False,
                save_embeddings=False,
            ),
        ),
        clustering=dict(
            method=ConnectedComponentsClustering,
            exec_params=dict(similarity_threshold=BEST_THRESHOLD),
        ),
        name="FEBRL-scale",
    )
    w.run(data, verbose=False)
    return w.clusters, time.time() - total_start


print("\n" + "=" * 60)
print("  PYJEDAI SCALABILITY (fixed best config)")
print(f"  {BEST_VECTORIZER} | {BEST_DISTANCE} | top_k={BEST_TOP_K} | thr={BEST_THRESHOLD}")
print("=" * 60)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
results = []

for ds in DATASETS:
    profiles_path = os.path.join(CONVERTED_DIR, ds, "profiles.csv")
    test_path     = os.path.join(SPLITS_DIR, ds, "test_set.csv")

    print(f"\n{'='*60}\n  Dataset: {ds}\n{'='*60}", flush=True)

    if not os.path.exists(profiles_path) or not os.path.exists(test_path):
        print(f"  [SKIP] missing files for {ds}")
        results.append({"dataset": ds, "status": "SKIP: file not found"})
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        continue

    try:
        t_load = time.time()
        df = pd.read_csv(profiles_path, engine="python", na_filter=False).astype(str)
        test_df = pd.read_csv(test_path, engine="python")
        t_load = time.time() - t_load

        print(f"  Records    : {len(df):,}")
        print(f"  Test pairs : {len(test_df):,}  (pos={int(test_df['label'].sum()):,})")
        print(f"  Load time  : {t_load:.2f}s  |  mem: {mem_mb():.0f} MB", flush=True)

        clusters, t_workflow = run_pipeline(df)

        entity_ids = df["id"].tolist()
        predicted_pairs = get_predicted_pairs(clusters, entity_ids)
        test_p, test_r, test_f1 = score_split(test_df, predicted_pairs)
        peak = peak_mem_mb()

        print(f"\n  --- RESULTS ---")
        print(f"  Clusters   : {len(clusters):,}")
        print(f"  Precision  : {test_p:.4f}  Recall: {test_r:.4f}  F1: {test_f1:.4f}")
        print(f"  Workflow   : {t_workflow:.2f}s  ({t_workflow/60:.1f} min)")
        print(f"  Peak mem   : {peak:.0f} MB", flush=True)

        results.append({
            "dataset": ds, "n_records": len(df), "n_test_pairs": len(test_df),
            "n_clusters": len(clusters),
            "precision": round(test_p, 4), "recall": round(test_r, 4), "f1": round(test_f1, 4),
            "time_load": round(t_load, 2), "time_workflow": round(t_workflow, 2),
            "peak_mem_mb": round(peak, 1), "status": "OK",
        })

    except MemoryError:
        print(f"  [OOM] {ds}", flush=True)
        results.append({"dataset": ds, "status": "OOM"})
    except Exception as e:
        print(f"  [ERROR] {ds}: {e}", flush=True)
        traceback.print_exc()
        results.append({"dataset": ds, "status": f"FAILED: {str(e)[:200]}"})

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"  [checkpoint saved]", flush=True)

pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print("\n" + "=" * 60)
print("  SCALABILITY SUMMARY")
print("=" * 60)
dfr = pd.DataFrame(results)
cols = [c for c in ["dataset","n_records","precision","recall","f1","time_workflow","peak_mem_mb","status"] if c in dfr.columns]
print(dfr[cols].to_string(index=False))
print(f"\nSaved to: {OUTPUT_CSV}")