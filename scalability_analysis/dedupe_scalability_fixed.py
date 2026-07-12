"""
dedupe_scalability_fixed.py
---------------------------
Fixed-config scalability runs for Dedupe on synthetic FEBRL datasets.
Uses the single BEST config chosen from the 10K search (selected by valid_f1),
applied verbatim to every size -- INCLUDING a frozen decision threshold,
exactly like pyjedai_scalability_fixed.py. No per-dataset threshold sweep,
no valid set at scale.

END-TO-END: each size blocks over ALL records with the learned predicates,
scores the candidate pairs, and clusters at the frozen threshold via
deduper.partition(). A test pair counts as a predicted match iff both of its
records land in the SAME cluster -- so Dedupe eats blocking/candidate-generation
loss and folds that cost into the runtime, exactly like Splink / pyJedAI / Zingg.

10K is excluded (it was the tuning size).
Loops: 50K, 100K, 200K, 300K, 1M, 2M -- one run each, checkpointing after each.
Metrics per size: runtime, peak memory (VmHWM), and test P/R/F1 (scored over the
split-file labelled pairs by cluster co-membership, consistent with all tools).
Failure policy: OOM / TIMEOUT / ERROR are recorded, not crashes.

Output: dedupe_scalability_results.csv
"""

import sys
# zope namespace fix
import zope
zope.__path__.append("/home/it2022025/.local/lib/python3.10/site-packages/zope")

import os
import re
import io
import json
import time
import random
import traceback
import warnings
import logging

import numpy as np
import pandas as pd
import psutil
import dedupe
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

random.seed(42)
np.random.seed(42)

CONVERTED_DIR = "/home/it2022025/er_scalability/converted"
SPLITS_DIR    = "/home/it2022025/er_scalability/splits"
OUTPUT_CSV    = "/home/it2022025/er_scalability/scalability/dedupe_scalability_results.csv"

# 10K excluded (tuning size). Rest of the ladder, looped internally:
DATASETS = ["50K", "100K", "200K", "300K", "1M", "2M"]

# BEST config from the 10K end-to-end search, by argmax(valid_f1) -- same rule
# as the splink/pyjedai/zingg fixed scripts. Winner = config #2 (valid_f1=0.955).
# Search was truncated at 5/50 configs (each now blocks+scores all 10K records);
# completed configs were flat (test_f1 0.9593-0.9639), so this pick is
# representative -- a disclosed caveat, not a re-tune.
# index_predicates=True may OOM/TIMEOUT at 1M-2M; that is recorded as data.
BEST_NEG_RATIO        = 3.204
BEST_RECALL           = 0.914
BEST_INDEX_PREDICATES = True
BEST_THRESHOLD        = 0.1       # frozen partition() cut -- NOT re-tuned

# prepare_training record-sample sizes (training input only; all records are
# still blocked, matched and evaluated).
SAMPLE_SIZES = {
    "50K":  5000,
    "100K": 10000,
    "200K": 15000,
    "300K": 15000,
    "1M":   20000,
    "2M":   20000,
}

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

def clean_text(text):
    if text is None or str(text).strip() in ("", "nan"):
        return None
    text = re.sub(r"\s+", " ", str(text).lower()).strip()
    return text if text else None

def df_to_records(df):
    records = {}
    for _, row in df.iterrows():
        rid = int(row["id"])
        records[rid] = {
            "given_name"   : clean_text(row.get("given_name")),
            "surname"      : clean_text(row.get("surname")),
            "address_1"    : clean_text(row.get("address_1")),
            "suburb"       : clean_text(row.get("suburb")),
            "postcode"     : clean_text(row.get("postcode")),
            "state"        : clean_text(row.get("state")),
            "date_of_birth": clean_text(row.get("date_of_birth")),
            "soc_sec_id"   : clean_text(row.get("soc_sec_id")),
            "phone_number" : clean_text(row.get("phone_number")),
        }
    return records

def load_split_pairs(split_df, records):
    pairs, labels = [], []
    for _, row in split_df.iterrows():
        lid   = int(row["left_id"])
        rid   = int(row["right_id"])
        label = int(row["label"])
        if lid not in records or rid not in records:
            continue
        pairs.append((lid, rid))
        labels.append(label)
    return pairs, labels

def cluster_id_map(clusters):
    """record_id -> cluster index. Records dedupe left unclustered are absent;
    the -1/-2 lookup defaults then guarantee they never count as a match."""
    cid = {}
    for k, (ids, _scores) in enumerate(clusters):
        for x in ids:
            cid[int(x)] = k
    return cid

def run_pipeline(records, train_df, test_df, ds_label):
    """Train on TRAIN with the frozen config, then run Dedupe END-TO-END over
    ALL records (block -> score -> partition at the frozen threshold) and score
    the TEST split by cluster co-membership. No valid set, no per-dataset sweep
    (mirrors pyjedai_fixed)."""
    workflow_start = time.time()
    sample_size    = SAMPLE_SIZES.get(ds_label, 20000)

    fields = [
        dedupe.variables.Text("given_name",      has_missing=True),
        dedupe.variables.Text("surname",         has_missing=True),
        dedupe.variables.Text("address_1",       has_missing=True),
        dedupe.variables.String("suburb",        has_missing=True),
        dedupe.variables.String("postcode",      has_missing=True),
        dedupe.variables.String("state",         has_missing=True),
        dedupe.variables.String("date_of_birth", has_missing=True),
        dedupe.variables.String("soc_sec_id",    has_missing=True),
        dedupe.variables.String("phone_number",  has_missing=True),
    ]

    train_pairs, train_labels = load_split_pairs(train_df, records)

    # fresh deduper per dataset (no settings cache)
    deduper = dedupe.Dedupe(fields)

    matches, distinct = [], []
    for (id1, id2), label in zip(train_pairs, train_labels):
        if label == 1:
            matches.append((records[id1], records[id2]))
        else:
            distinct.append((records[id1], records[id2]))

    n_keep = int(round(len(matches) * BEST_NEG_RATIO))
    n_keep = min(n_keep, len(distinct))
    if 0 < n_keep < len(distinct):
        distinct = random.sample(distinct, n_keep)

    print(f"    Match pairs    : {len(matches):,}", flush=True)
    print(f"    Distinct pairs : {len(distinct):,}", flush=True)

    training_file = io.StringIO()
    json.dump({"match": matches, "distinct": distinct}, training_file)
    training_file.seek(0)

    sample_keys    = random.sample(list(records.keys()), min(sample_size, len(records)))
    sample_records = {k: records[k] for k in sample_keys}

    print(f"    prepare_training on {len(sample_records):,} records...", flush=True)
    deduper.prepare_training(
        sample_records,
        training_file=training_file,
        sample_size=5000,
        blocked_proportion=0.9,
    )
    deduper.train(recall=BEST_RECALL, index_predicates=BEST_INDEX_PREDICATES)

    # end-to-end over ALL records; a test pair matches iff its two records land
    # in the same cluster (see module docstring)
    print(f"    partition() over {len(records):,} records at thr={BEST_THRESHOLD}...", flush=True)
    clusters = deduper.partition(records, threshold=BEST_THRESHOLD)
    cid = cluster_id_map(clusters)

    test_pairs, test_labels = load_split_pairs(test_df, records)
    preds_test = [1 if cid.get(lid, -1) == cid.get(rid, -2) else 0
                  for (lid, rid) in test_pairs]
    test_p  = precision_score(test_labels, preds_test, zero_division=0)
    test_r  = recall_score(test_labels,    preds_test, zero_division=0)
    test_f1 = f1_score(test_labels,        preds_test, zero_division=0)

    t_workflow = time.time() - workflow_start
    return test_p, test_r, test_f1, t_workflow


print("\n" + "=" * 60)
print("  DEDUPE SCALABILITY (fixed best config)")
print(f"  neg_ratio={BEST_NEG_RATIO} | recall={BEST_RECALL} | "
      f"index_predicates={BEST_INDEX_PREDICATES} | thr={BEST_THRESHOLD}")
print("=" * 60)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
results = []

for ds in DATASETS:
    profiles_path = os.path.join(CONVERTED_DIR, ds, "profiles.csv")
    train_path    = os.path.join(SPLITS_DIR,    ds, "train_set.csv")
    test_path     = os.path.join(SPLITS_DIR,    ds, "test_set.csv")

    print(f"\n{'='*60}\n  Dataset: {ds}\n{'='*60}", flush=True)

    if not (os.path.exists(profiles_path) and os.path.exists(train_path)
            and os.path.exists(test_path)):
        print(f"  [SKIP] missing files for {ds}")
        results.append({"dataset": ds, "status": "SKIP: file not found"})
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        continue

    try:
        t_load = time.time()
        df       = pd.read_csv(profiles_path, engine="python", na_filter=False).astype(str)
        train_df = pd.read_csv(train_path,    engine="python")
        test_df  = pd.read_csv(test_path,     engine="python")
        records  = df_to_records(df)
        t_load   = time.time() - t_load

        print(f"  Records    : {len(records):,}")
        print(f"  Test pairs : {len(test_df):,}  (pos={int(test_df['label'].sum()):,})")
        print(f"  Load time  : {t_load:.2f}s  |  mem: {mem_mb():.0f} MB", flush=True)

        test_p, test_r, test_f1, t_workflow = run_pipeline(records, train_df, test_df, ds)
        peak = peak_mem_mb()

        print(f"\n  --- RESULTS ---")
        print(f"  Precision  : {test_p:.4f}  Recall: {test_r:.4f}  F1: {test_f1:.4f}")
        print(f"  Workflow   : {t_workflow:.2f}s  ({t_workflow/60:.1f} min)")
        print(f"  Peak mem   : {peak:.0f} MB", flush=True)

        results.append({
            "dataset": ds, "n_records": len(records), "n_test_pairs": len(test_df),
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
