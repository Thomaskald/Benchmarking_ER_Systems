"""
splink_scalability.py
----------------------
Scalability benchmarking for Splink on synthetic FEBRL datasets.
Datasets: 10K, 50K, 100K, 200K, 300K, 1M, 2M records.

Mode: Supervised EM using train+valid labelled pairs
(same approach as DER benchmarking on CDDB).

Pipeline stages timed:
  - data loading
  - linker setup
  - u estimation
  - supervised EM training (estimate_m_from_pairwise_labels)
  - prediction (scoring)
  - threshold sweep on valid set
  - evaluation on test set

Metrics: runtime per stage, total runtime, peak memory (VmHWM),
         Precision, Recall, F1 on test set.

Output: splink_scalability_results.csv
"""

import json
import re
import time
import traceback
import warnings
import logging
import os

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

CONVERTED_DIR = "/home/it2022025/er_scalability/converted"
SPLITS_DIR    = "/home/it2022025/er_scalability/splits"
OUTPUT_CSV    = "/home/it2022025/er_scalability/splink/splink_scalability_results.csv"
MODEL_DIR     = "/home/it2022025/er_scalability/splink/models"

DATASETS = ["10K", "50K", "100K", "200K", "300K", "1M", "2M"]

RANDOM_SEED = 42

# Random pairs for u estimation
U_PAIRS_SMALL = 1e6
U_PAIRS_LARGE = 5e6

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
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None

# -------------------------------------------------------
# PIPELINE
# -------------------------------------------------------

def run_pipeline(df, train_df, valid_df, test_df, ds_label):
    """
    Full Splink supervised EM pipeline.
    Returns: stage_times, total_time, mem_used, peak_mem,
             test_p, test_r, test_f1, best_thresh, n_candidates
    """
    stage_times = {}
    mem_start   = mem_mb()
    total_start = time.time()

    n_records = len(df)
    u_pairs   = U_PAIRS_LARGE if n_records >= 300_000 else U_PAIRS_SMALL

    # -- Prepare dataframe --
    df = df.copy()
    df = df.rename(columns={"id": "unique_id"})
    df["unique_id"] = df["unique_id"].astype(str)

    # Clean text fields
    for col in ["given_name", "surname", "address_1", "suburb", "state",
                "postcode", "date_of_birth"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Derived blocking features
    df["surname_2"]  = df["surname"].str[:2]
    df["surname_4"]  = df["surname"].str[:4]
    df["postcode_3"] = df["postcode"].str[:3]

    # -- Linker setup --
    t0 = time.time()
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            cl.JaroWinklerAtThresholds("given_name", [0.95, 0.88, 0.80]),
            cl.JaroWinklerAtThresholds("surname",    [0.95, 0.88, 0.80]),
            cl.ExactMatch("postcode"),
            cl.ExactMatch("suburb"),
            cl.ExactMatch("state"),
            cl.JaroWinklerAtThresholds("address_1",  [0.95, 0.88]),
        ],
        blocking_rules_to_generate_predictions=[
            block_on("surname_2"),
            block_on("surname_4"),
            block_on("postcode"),
            block_on("postcode_3"),
            block_on("suburb"),
            block_on("given_name"),
        ],
        retain_intermediate_calculation_columns=False,
        retain_matching_columns=False,
    )
    db_api = DuckDBAPI()
    linker  = Linker(df, settings, db_api)
    stage_times["linker_setup"] = time.time() - t0
    print(f"    [linker_setup]   {stage_times['linker_setup']:.2f}s  |  mem: {mem_mb():.0f} MB", flush=True)

    # -- Estimate u --
    t0 = time.time()
    linker.training.estimate_u_using_random_sampling(max_pairs=u_pairs)
    stage_times["u_estimation"] = time.time() - t0
    print(f"    [u_estimation]   {stage_times['u_estimation']:.2f}s  |  mem: {mem_mb():.0f} MB", flush=True)

    # -- Supervised EM training (same as DER script) --
    t0 = time.time()

    # Combine train + valid for training (same as CDDB DER script)
    all_labeled = pd.concat([train_df, valid_df], ignore_index=True)

    # Use all available train+valid labelled pairs (no capping)
    matches  = all_labeled[all_labeled["label"] == 1]
    distinct = all_labeled[all_labeled["label"] == 0]
    all_labeled_sampled = pd.concat([matches, distinct], ignore_index=True)

    print(f"    Match pairs used    : {len(matches):,}", flush=True)
    print(f"    Distinct pairs used : {len(distinct):,}", flush=True)

    # Build Splink labelled pairs DataFrame
    labelled_pairs_df = pd.DataFrame({
        "unique_id_l":          all_labeled_sampled["left_id"].apply(lambda x: str(int(x))),
        "unique_id_r":          all_labeled_sampled["right_id"].apply(lambda x: str(int(x))),
        "clerical_match_score": all_labeled_sampled["label"].astype(float),
    })

    labels_sdf = linker.table_management.register_labels_table(
        labelled_pairs_df, overwrite=True
    )
    linker.training.estimate_m_from_pairwise_labels(labels_sdf)

    # Patch prior to ground-truth match rate
    total_possible  = n_records * (n_records - 1) / 2
    n_true_matches  = int(all_labeled["label"].sum())
    true_match_rate = n_true_matches / total_possible

    model_path         = os.path.join(MODEL_DIR, f"splink_{ds_label}.json")
    model_patched_path = os.path.join(MODEL_DIR, f"splink_{ds_label}_patched.json")
    os.makedirs(MODEL_DIR, exist_ok=True)

    linker.misc.save_model_to_json(model_path, overwrite=True)
    with open(model_path) as f:
        model_json = json.load(f)
    model_json["probability_two_random_records_match"] = true_match_rate
    with open(model_patched_path, "w") as f:
        json.dump(model_json, f)

    linker = Linker(df, model_patched_path, db_api)

    print(f"    Prior patched : {true_match_rate:.8f}", flush=True)

    stage_times["em_training"] = time.time() - t0
    print(f"    [em_training]    {stage_times['em_training']:.2f}s  |  mem: {mem_mb():.0f} MB", flush=True)

    # -- Predict --
    t0 = time.time()
    results_sdf = linker.inference.predict(threshold_match_probability=0.0)
    results_df  = results_sdf.as_pandas_dataframe()
    stage_times["prediction"] = time.time() - t0
    print(f"    [prediction]     {stage_times['prediction']:.2f}s  |  mem: {mem_mb():.0f} MB", flush=True)
    print(f"    Candidate pairs  : {len(results_df):,}", flush=True)

    # -- Build score map --
    score_map = {}
    for _, row in results_df.iterrows():
        l = str(row["unique_id_l"])
        r = str(row["unique_id_r"])
        p = float(row["match_probability"])
        score_map[(l, r)] = p
        score_map[(r, l)] = p

    # -- Score valid and test pairs --
    def get_scores(split_df):
        scores, covered = [], 0
        for _, row in split_df.iterrows():
            key = (str(int(row["left_id"])), str(int(row["right_id"])))
            s   = score_map.get(key, 0.0)
            scores.append(s)
            if key in score_map:
                covered += 1
        return np.array(scores), covered

    y_valid = valid_df["label"].values
    y_test  = test_df["label"].values

    valid_scores, valid_covered = get_scores(valid_df)
    test_scores,  test_covered  = get_scores(test_df)

    print(f"    Valid covered : {valid_covered:,} / {len(valid_df):,} "
          f"({100*valid_covered/len(valid_df):.1f}%)", flush=True)
    print(f"    Test covered  : {test_covered:,} / {len(test_df):,} "
          f"({100*test_covered/len(test_df):.1f}%)", flush=True)

    # -- Threshold sweep on valid set --
    t0 = time.time()
    thresholds      = np.arange(0.01, 1.00, 0.01)
    best_thresh     = 0.50
    best_f1_valid   = 0.0
    for t in thresholds:
        preds = (valid_scores >= t).astype(int)
        f1    = f1_score(y_valid, preds, zero_division=0)
        if f1 > best_f1_valid:
            best_f1_valid = f1
            best_thresh   = t
    stage_times["threshold_sweep"] = time.time() - t0
    print(f"    Best threshold : {best_thresh:.2f}  |  Valid F1: {best_f1_valid:.4f}", flush=True)

    # -- Final evaluation on test set --
    test_preds = (test_scores >= best_thresh).astype(int)
    test_p     = precision_score(y_test, test_preds, zero_division=0)
    test_r     = recall_score(y_test,    test_preds, zero_division=0)
    test_f1    = f1_score(y_test,        test_preds, zero_division=0)

    total_time = time.time() - total_start
    mem_used   = mem_mb() - mem_start
    peak_mem   = peak_mem_mb()

    return (stage_times, total_time, mem_used, peak_mem,
            test_p, test_r, test_f1, best_thresh,
            len(results_df), valid_covered, test_covered)

# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------

print("\n" + "=" * 60)
print("  SPLINK SCALABILITY ANALYSIS")
print("=" * 60)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

results = []

for ds in DATASETS:
    profiles_path = os.path.join(CONVERTED_DIR, ds, "profiles.csv")
    train_path    = os.path.join(SPLITS_DIR,    ds, "train_set.csv")
    valid_path    = os.path.join(SPLITS_DIR,    ds, "valid_set.csv")
    test_path     = os.path.join(SPLITS_DIR,    ds, "test_set.csv")

    print(f"\n{'='*60}")
    print(f"  Dataset: {ds}")
    print(f"{'='*60}")

    for p, name in [(profiles_path, "profiles"),
                    (train_path,    "train set"),
                    (valid_path,    "valid set"),
                    (test_path,     "test set")]:
        if not os.path.exists(p):
            print(f"  [SKIP] {name} not found: {p}")
            results.append({"dataset": ds, "status": f"SKIP: {name} not found"})
            continue

    try:
        # -- Load --
        t_load    = time.time()
        df        = pd.read_csv(profiles_path, engine="python", na_filter=False).astype(str)
        train_df  = pd.read_csv(train_path,    engine="python")
        valid_df  = pd.read_csv(valid_path,    engine="python")
        test_df   = pd.read_csv(test_path,     engine="python")
        t_load    = time.time() - t_load

        print(f"  Records    : {len(df):,}")
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
         test_p, test_r, test_f1, best_thresh,
         n_candidates, valid_covered, test_covered) = run_pipeline(
            df, train_df, valid_df, test_df, ds
        )

        # -- Print --
        print(f"\n  --- RESULTS ---")
        print(f"  Candidates : {n_candidates:,}")
        print(f"  Threshold  : {best_thresh:.2f}")
        print(f"  Precision  : {test_p:.4f}")
        print(f"  Recall     : {test_r:.4f}")
        print(f"  F1         : {test_f1:.4f}")
        print(f"  Total time : {total_time:.2f}s  ({total_time/60:.1f} min)")
        print(f"  Mem used   : {mem_used:.0f} MB  |  Peak: {peak_mem:.0f} MB")

        results.append({
            "dataset"              : ds,
            "n_records"            : len(df),
            "n_train_pairs"        : len(train_df),
            "n_valid_pairs"        : len(valid_df),
            "n_test_pairs"         : len(test_df),
            "n_test_pos"           : int(test_df["label"].sum()),
            "n_candidates"         : n_candidates,
            "best_threshold"       : round(best_thresh,  2),
            "precision"            : round(test_p,       4),
            "recall"               : round(test_r,       4),
            "f1"                   : round(test_f1,      4),
            "time_load"            : round(t_load,                           2),
            "time_linker_setup"    : round(stage_times["linker_setup"],      2),
            "time_u_estimation"    : round(stage_times["u_estimation"],      2),
            "time_em_training"     : round(stage_times["em_training"],       2),
            "time_prediction"      : round(stage_times["prediction"],        2),
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