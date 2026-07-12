"""
pyjedai_scalability.py
-----------------------
Scalability benchmarking for PyJedAI on synthetic FEBRL datasets.
Datasets: 10K, 50K, 100K, 200K, 300K, 1M, 2M records.

Pipeline (same as DER benchmarking):
  EmbeddingsNNWorkFlow with sminilm + faiss
  -> ConnectedComponentsClustering

PyJedAI is fully unsupervised — no training needed.
Train/valid splits are not used.
Evaluation is on test set only for fair comparison with
supervised frameworks.

Metrics: runtime, peak memory (VmHWM),
         Precision, Recall, F1 on test set.

Output: pyjedai_scalability_results.csv
"""

import time
import traceback
import warnings
import logging
import os
from itertools import combinations

import pandas as pd
import psutil

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from pyjedai.datamodel import Data
from pyjedai.workflow import EmbeddingsNNWorkFlow
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.clustering import ConnectedComponentsClustering

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

CONVERTED_DIR = "/home/it2022025/er_scalability/converted"
SPLITS_DIR    = "/home/it2022025/er_scalability/splits"
OUTPUT_CSV    = "/home/it2022025/er_scalability/pyjedai/pyjedai_scalability_results.csv"

DATASETS = ["10K", "50K", "100K", "200K", "300K", "1M", "2M"]

# Attributes to use for embeddings — same meaningful fields as DER
EMBEDDING_ATTRS = ["given_name", "surname", "address_1", "suburb", "postcode"]

# -------------------------------------------------------
# UTILITIES
# -------------------------------------------------------

def mem_mb():
    """Current RSS memory in MB."""
    return psutil.Process().memory_info().rss / 1024 ** 2

def peak_mem_mb():
    """Peak RSS memory in MB (VmHWM = true peak physical RAM)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return mem_mb()

def evaluate(clusters, df, test_df):
    """
    Compute Precision, Recall, F1 against test set.
    Only positive pairs (label==1) from test_df are used as ground truth.
    Evaluation is on test set pairs only for fair comparison with
    supervised frameworks.

    clusters : list of sets of integer positional indices into df
    df       : profiles DataFrame with integer 'id' column
    test_df  : test set DataFrame with left_id, right_id, label columns
    """
    # Map positional index -> integer record ID
    id_col = df["id"].astype(int).tolist()

    # Build predicted pairs from clusters
    predicted = set()
    for cluster in clusters:
        for i1, i2 in combinations(sorted(cluster), 2):
            a, b = id_col[i1], id_col[i2]
            predicted.add((min(a, b), max(a, b)))

    # Build ground truth from test set positive pairs only
    ground_truth = set()
    for _, row in test_df[test_df["label"] == 1].iterrows():
        a, b = int(row["left_id"]), int(row["right_id"])
        ground_truth.add((min(a, b), max(a, b)))

    # Evaluate on test set pairs
    test_pairs = set()
    for _, row in test_df.iterrows():
        a, b = int(row["left_id"]), int(row["right_id"])
        test_pairs.add((min(a, b), max(a, b)))

    y_true, y_pred = [], []
    for pair in test_pairs:
        y_true.append(1 if pair in ground_truth else 0)
        y_pred.append(1 if pair in predicted    else 0)

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return precision, recall, f1, len(predicted), len(ground_truth)

# -------------------------------------------------------
# PIPELINE
# -------------------------------------------------------

def run_pipeline(df, ds_label):
    """
    PyJedAI EmbeddingsNN pipeline — same as DER benchmarking.
    EmbeddingsNNWorkFlow with sminilm + faiss
    -> ConnectedComponentsClustering
    """
    mem_start   = mem_mb()
    total_start = time.time()

    # -- Data init --
    t0 = time.time()
    data = Data(dataset_1=df, id_column_name_1="id")
    t_data_init = time.time() - t0
    print(f"    [data_init]  {t_data_init:.2f}s  |  mem: {mem_mb():.0f} MB", flush=True)

    # -- EmbeddingsNN Workflow -- same as DER script
    w = EmbeddingsNNWorkFlow(
        block_building=dict(
            method=EmbeddingsNNBlockBuilding,
            params=dict(
                vectorizer='sminilm',
                similarity_search='faiss'
            ),
            attributes_1=EMBEDDING_ATTRS,
            exec_params=dict(
                top_k=5,
                similarity_distance='cosine',
                load_embeddings_if_exist=False,
                save_embeddings=False
            )
        ),
        clustering=dict(
            method=ConnectedComponentsClustering,
            exec_params=dict(
                similarity_threshold=0.85
            )
        ),
        name=f"FEBRL-DirtyER-EmbeddingsNN-{ds_label}"
    )

    t0 = time.time()
    w.run(data, verbose=False)
    t_workflow = time.time() - t0
    print(f"    [workflow]   {t_workflow:.2f}s  |  mem: {mem_mb():.0f} MB", flush=True)

    clusters   = w.clusters
    total_time = time.time() - total_start
    mem_used   = mem_mb() - mem_start
    peak_mem   = peak_mem_mb()

    return clusters, t_data_init, t_workflow, total_time, mem_used, peak_mem

# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------

print("\n" + "=" * 60)
print("  PYJEDAI SCALABILITY ANALYSIS  (EmbeddingsNN)")
print("=" * 60)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

results = []

for ds in DATASETS:
    profiles_path = os.path.join(CONVERTED_DIR, ds, "profiles.csv")
    test_path     = os.path.join(SPLITS_DIR,    ds, "test_set.csv")

    print(f"\n{'='*60}")
    print(f"  Dataset: {ds}")
    print(f"{'='*60}")

    if not os.path.exists(profiles_path):
        print(f"  [SKIP] profiles not found: {profiles_path}")
        results.append({"dataset": ds, "status": "SKIP: file not found"})
        continue

    if not os.path.exists(test_path):
        print(f"  [SKIP] test set not found: {test_path}")
        results.append({"dataset": ds, "status": "SKIP: test set not found"})
        continue

    try:
        # -- Load --
        t_load  = time.time()
        df      = pd.read_csv(profiles_path, engine="python", na_filter=False).astype(str)
        test_df = pd.read_csv(test_path, engine="python")
        t_load  = time.time() - t_load

        n_test_pos = int(test_df["label"].sum())
        n_test_neg = len(test_df) - n_test_pos

        print(f"  Records    : {len(df):,}")
        print(f"  Test pairs : {len(test_df):,}  (pos={n_test_pos:,}, neg={n_test_neg:,})")
        print(f"  Load time  : {t_load:.2f}s")
        print(f"  Memory     : {mem_mb():.0f} MB")
        print()

        # -- Run pipeline --
        clusters, t_data_init, t_workflow, total_time, mem_used, peak_mem = run_pipeline(
            df, ds
        )

        # -- Evaluate on test set --
        precision, recall, f1, n_pred, n_gt = evaluate(clusters, df, test_df)

        # -- Print --
        print(f"\n  --- RESULTS ---")
        print(f"  Clusters   : {len(clusters):,}")
        print(f"  Pred pairs : {n_pred:,}  |  Test GT pairs: {n_gt:,}")
        print(f"  Precision  : {precision:.4f}")
        print(f"  Recall     : {recall:.4f}")
        print(f"  F1         : {f1:.4f}")
        print(f"  Total time : {total_time:.2f}s  ({total_time/60:.1f} min)")
        print(f"  Mem used   : {mem_used:.0f} MB  |  Peak: {peak_mem:.0f} MB")

        results.append({
            "dataset"        : ds,
            "n_records"      : len(df),
            "n_test_pairs"   : len(test_df),
            "n_test_pos"     : n_test_pos,
            "n_pred_pairs"   : n_pred,
            "n_clusters"     : len(clusters),
            "precision"      : round(precision, 4),
            "recall"         : round(recall,    4),
            "f1"             : round(f1,        4),
            "time_load"      : round(t_load,       2),
            "time_data_init" : round(t_data_init,  2),
            "time_workflow"  : round(t_workflow,   2),
            "time_total"     : round(total_time,   2),
            "mem_used_mb"    : round(mem_used,  0),
            "mem_peak_mb"    : round(peak_mem,  0),
            "status"         : "OK"
        })

    except Exception as e:
        print(f"  [ERROR] {ds}: {e}", flush=True)
        traceback.print_exc()
        results.append({
            "dataset": ds,
            "status" : f"FAILED: {str(e)}"
        })

    # Checkpoint after every dataset
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"  [checkpoint saved]", flush=True)

# -------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------

df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 60)
print("  SCALABILITY SUMMARY")
print("=" * 60)
cols = ["dataset", "n_records", "precision", "recall", "f1",
        "time_total", "mem_peak_mb", "status"]
available_cols = [c for c in cols if c in df_results.columns]
print(df_results[available_cols].to_string(index=False))
print(f"\nFull results saved to: {OUTPUT_CSV}")