"""
dedupe_scalability.py
----------------------
Scalability benchmarking for Dedupe on synthetic FEBRL datasets.
Datasets: 10K, 50K, 100K, 200K, 300K, 1M, 2M records.

Mode: Supervised — train pairs used as match/distinct examples,
threshold tuned on valid set, final evaluation on test set.
Same approach as DER benchmarking on CDDB.

Pipeline stages timed:
  - data loading
  - training (prepare + train)
  - scoring (valid + test pairs)
  - threshold sweep on valid set
  - evaluation on test set

Metrics: runtime per stage, total runtime, peak memory (VmHWM),
         Precision, Recall, F1 on test set.

Output: dedupe_scalability_results.csv
"""

import sys
# Fix zope namespace to include our installed zope.index
sys.path.insert(0, '/home/it2022025/er_scalability/dedupe/packages')
import zope
zope.__path__.append('/home/it2022025/er_scalability/dedupe/packages/zope')
# end fix

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

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

CONVERTED_DIR = "/home/it2022025/er_scalability/converted"
SPLITS_DIR    = "/home/it2022025/er_scalability/splits"
OUTPUT_CSV    = "/home/it2022025/er_scalability/dedupe/dedupe_scalability_results.csv"
SETTINGS_DIR  = "/home/it2022025/er_scalability/dedupe/settings"

DATASETS = ["10K", "50K", "100K", "200K", "300K", "1M", "2M"]

# Sample size for prepare_training — same as CDDB DER script
SAMPLE_SIZES = {
    "10K":  2000,
    "50K":  5000,
    "100K": 10000,
    "200K": 15000,
    "300K": 15000,
    "1M":   20000,
    "2M":   20000,
}

# -------------------------------------------------------
# UTILITIES
# -------------------------------------------------------

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
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else None

def df_to_records(df):
    """Convert profiles DataFrame to Dedupe records dict."""
    records = {}
    for _, row in df.iterrows():
        rid = int(row["id"])
        records[rid] = {
            "given_name"   : clean_text(row.get("given_name")),
            "surname"      : clean_text(row.get("surname")),
            "suburb"       : clean_text(row.get("suburb")),
            "postcode"     : clean_text(row.get("postcode")),
            "state"        : clean_text(row.get("state")),
            "address_1"    : clean_text(row.get("address_1")),
            "date_of_birth": clean_text(row.get("date_of_birth")),
        }
    return records

def load_split_pairs(split_df, records):
    """Load pairs and labels from split DataFrame, filtering to known record IDs."""
    pairs  = []
    labels = []
    for _, row in split_df.iterrows():
        lid   = int(row["left_id"])
        rid   = int(row["right_id"])
        label = int(row["label"])
        if lid not in records or rid not in records:
            continue
        pairs.append((lid, rid))
        labels.append(label)
    return pairs, labels

# -------------------------------------------------------
# PIPELINE
# -------------------------------------------------------

def run_pipeline(records, train_df, valid_df, test_df, ds_label):
    """
    Full Dedupe supervised pipeline.
    Same approach as DER CDDB script:
      - Build match/distinct from train pairs
      - prepare_training on sample of records
      - train()
      - score() valid and test pairs
      - threshold sweep on valid set
      - final evaluation on test set
    """
    stage_times  = {}
    mem_start    = mem_mb()
    total_start  = time.time()

    sample_size   = SAMPLE_SIZES.get(ds_label, 20000)
    settings_path = os.path.join(SETTINGS_DIR, f"dedupe_{ds_label}_settings")

    # -- Define fields --
    fields = [
        dedupe.variables.Text("given_name",      has_missing=True),
        dedupe.variables.Text("surname",         has_missing=True),
        dedupe.variables.Text("address_1",       has_missing=True),
        dedupe.variables.String("suburb",        has_missing=True),
        dedupe.variables.String("postcode",      has_missing=True),
        dedupe.variables.String("state",         has_missing=True),
        dedupe.variables.String("date_of_birth", has_missing=True),
    ]

    # -- Load train pairs --
    train_pairs, train_labels = load_split_pairs(train_df, records)

    # -- Training --
    t0 = time.time()

    if os.path.exists(settings_path):
        print(f"    Loading cached settings: {settings_path}", flush=True)
        with open(settings_path, "rb") as f:
            deduper = dedupe.StaticDedupe(f)
    else:
        deduper = dedupe.Dedupe(fields)

        # Build match and distinct lists from train pairs
        matches  = []
        distinct = []
        for (id1, id2), label in zip(train_pairs, train_labels):
            if label == 1:
                matches.append((records[id1], records[id2]))
            else:
                distinct.append((records[id1], records[id2]))

        print(f"    Match pairs    : {len(matches):,}", flush=True)
        print(f"    Distinct pairs : {len(distinct):,}", flush=True)

        # Build training file (same as CDDB DER script)
        training_data = {"match": matches, "distinct": distinct}
        training_file = io.StringIO()
        json.dump(training_data, training_file)
        training_file.seek(0)

        # Sample records for prepare_training (same as CDDB DER script)
        sample_keys    = random.sample(list(records.keys()),
                                       min(sample_size, len(records)))
        sample_records = {k: records[k] for k in sample_keys}

        print(f"    prepare_training on {len(sample_records):,} records...", flush=True)
        deduper.prepare_training(
            sample_records,
            training_file=training_file,
            sample_size=5000,
            blocked_proportion=0.9
        )

        print(f"    training...", flush=True)
        deduper.train()

        os.makedirs(SETTINGS_DIR, exist_ok=True)
        with open(settings_path, "wb") as f:
            deduper.write_settings(f)

    stage_times["training"] = time.time() - t0
    print(f"    [training]       {stage_times['training']:.2f}s  |  mem: {mem_mb():.0f} MB", flush=True)

    # -- Load valid and test pairs --
    valid_pairs, valid_labels = load_split_pairs(valid_df, records)
    test_pairs,  test_labels  = load_split_pairs(test_df,  records)

    # -- Scoring (same as CDDB DER script) --
    t0 = time.time()

    valid_record_pairs = [
        ((lid, records[lid]), (rid, records[rid]))
        for (lid, rid) in valid_pairs
    ]
    valid_scored   = deduper.score(valid_record_pairs)
    valid_score_map = {
        (int(l), int(r)): float(score)
        for (l, r), score in valid_scored
    }
    valid_scores = [
        valid_score_map.get((lid, rid), 0.0)
        for (lid, rid) in valid_pairs
    ]

    test_record_pairs = [
        ((lid, records[lid]), (rid, records[rid]))
        for (lid, rid) in test_pairs
    ]
    test_scored   = deduper.score(test_record_pairs)
    test_score_map = {
        (int(l), int(r)): float(score)
        for (l, r), score in test_scored
    }
    test_scores = [
        test_score_map.get((lid, rid), 0.0)
        for (lid, rid) in test_pairs
    ]

    stage_times["scoring"] = time.time() - t0
    print(f"    [scoring]        {stage_times['scoring']:.2f}s  |  mem: {mem_mb():.0f} MB", flush=True)
    print(f"    Valid scored : {len(valid_scored):,}", flush=True)
    print(f"    Test  scored : {len(test_scored):,}", flush=True)

    # -- Threshold sweep on valid set --
    t0 = time.time()
    thresholds     = [round(x * 0.05, 2) for x in range(1, 20)]
    best_f1_valid  = 0.0
    best_threshold = 0.5

    for t in thresholds:
        preds = [1 if s >= t else 0 for s in valid_scores]
        f1    = f1_score(valid_labels, preds, zero_division=0)
        if f1 > best_f1_valid:
            best_f1_valid  = f1
            best_threshold = t

    stage_times["threshold_sweep"] = time.time() - t0
    print(f"    Best threshold : {best_threshold}  |  Valid F1: {best_f1_valid:.4f}", flush=True)

    # -- Final evaluation on test set --
    preds_test = [1 if s >= best_threshold else 0 for s in test_scores]
    test_p     = precision_score(test_labels, preds_test, zero_division=0)
    test_r     = recall_score(test_labels,    preds_test, zero_division=0)
    test_f1    = f1_score(test_labels,        preds_test, zero_division=0)

    total_time = time.time() - total_start
    mem_used   = mem_mb() - mem_start
    peak_mem   = peak_mem_mb()

    return (stage_times, total_time, mem_used, peak_mem,
            test_p, test_r, test_f1, best_threshold,
            len(valid_scored), len(test_scored))

# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------

print("\n" + "=" * 60)
print("  DEDUPE SCALABILITY ANALYSIS")
print("=" * 60)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(SETTINGS_DIR, exist_ok=True)

results = []

for ds in DATASETS:
    profiles_path = os.path.join(CONVERTED_DIR, ds, "profiles.csv")
    train_path    = os.path.join(SPLITS_DIR,    ds, "train_set.csv")
    valid_path    = os.path.join(SPLITS_DIR,    ds, "valid_set.csv")
    test_path     = os.path.join(SPLITS_DIR,    ds, "test_set.csv")

    print(f"\n{'='*60}")
    print(f"  Dataset: {ds}")
    print(f"{'='*60}")

    skip = False
    for p, name in [(profiles_path, "profiles"),
                    (train_path,    "train set"),
                    (valid_path,    "valid set"),
                    (test_path,     "test set")]:
        if not os.path.exists(p):
            print(f"  [SKIP] {name} not found: {p}")
            results.append({"dataset": ds, "status": f"SKIP: {name} not found"})
            skip = True
            break
    if skip:
        continue

    try:
        # -- Load --
        t_load    = time.time()
        df        = pd.read_csv(profiles_path, engine="python", na_filter=False).astype(str)
        train_df  = pd.read_csv(train_path,    engine="python")
        valid_df  = pd.read_csv(valid_path,    engine="python")
        test_df   = pd.read_csv(test_path,     engine="python")
        t_load    = time.time() - t_load

        records = df_to_records(df)

        print(f"  Records    : {len(records):,}")
        print(f"  Train pairs: {len(train_df):,}  "
              f"(pos={int(train_df['label'].sum()):,}, "
              f"neg={int((train_df['label']==0).sum()):,})")
        print(f"  Valid pairs: {len(valid_df):,}  "
              f"(pos={int(valid_df['label'].sum()):,}, "
              f"neg={int((valid_df['label']==0).sum()):,})")
        print(f"  Test  pairs: {len(test_df):,}  "
              f"(pos={int(test_df['label'].sum()):,}, "
              f"neg={int((test_df['label']==0).sum()):,})")
        print(f"  Load time  : {t_load:.2f}s")
        print(f"  Memory     : {mem_mb():.0f} MB")
        print()

        # -- Run --
        (stage_times, total_time, mem_used, peak_mem,
         test_p, test_r, test_f1, best_threshold,
         n_valid_scored, n_test_scored) = run_pipeline(
            records, train_df, valid_df, test_df, ds
        )

        # -- Print --
        print(f"\n  --- RESULTS ---")
        print(f"  Threshold  : {best_threshold}")
        print(f"  Precision  : {test_p:.4f}")
        print(f"  Recall     : {test_r:.4f}")
        print(f"  F1         : {test_f1:.4f}")
        print(f"  Total time : {total_time:.2f}s  ({total_time/60:.1f} min)")
        print(f"  Mem used   : {mem_used:.0f} MB  |  Peak: {peak_mem:.0f} MB")

        results.append({
            "dataset"              : ds,
            "n_records"            : len(records),
            "n_train_pairs"        : len(train_df),
            "n_valid_pairs"        : len(valid_df),
            "n_test_pairs"         : len(test_df),
            "n_test_pos"           : int(test_df["label"].sum()),
            "n_valid_scored"       : n_valid_scored,
            "n_test_scored"        : n_test_scored,
            "best_threshold"       : best_threshold,
            "precision"            : round(test_p,   4),
            "recall"               : round(test_r,   4),
            "f1"                   : round(test_f1,  4),
            "time_load"            : round(t_load,                           2),
            "time_training"        : round(stage_times["training"],          2),
            "time_scoring"         : round(stage_times["scoring"],           2),
            "time_threshold_sweep" : round(stage_times["threshold_sweep"],   2),
            "time_total"           : round(total_time,                       2),
            "mem_used_mb"          : round(mem_used,  0),
            "mem_peak_mb"          : round(peak_mem,  0),
            "status"               : "OK"
        })

    except Exception as e:
        print(f"  [ERROR] {ds}: {e}", flush=True)
        traceback.print_exc()
        results.append({
            "dataset": ds,
            "status" : f"FAILED: {str(e)}"
        })

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