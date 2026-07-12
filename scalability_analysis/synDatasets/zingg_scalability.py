"""
zingg_scalability.py
---------------------
Scalability benchmarking for Zingg on synthetic FEBRL datasets.
Datasets: 10K, 50K, 100K, 200K, 300K, 1M, 2M records.

Mode: Supervised — train+valid pairs used as labelled training data,
threshold tuned on valid set, final evaluation on test set.
Same approach as DER benchmarking on CDDB.

Pipeline stages timed:
  - data loading
  - building Zingg training data
  - Zingg train phase
  - Zingg match phase
  - scoring (valid + test pairs)
  - threshold sweep on valid set
  - evaluation on test set

Metrics: runtime per stage, total runtime, peak memory (VmHWM),
         Precision, Recall, F1 on test set.

Output: zingg_scalability_results.csv
"""

import os
import re
import time
import glob
import random
import traceback
import warnings
import logging

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

random.seed(42)
np.random.seed(42)

# -------------------------------------------------------
# ZINGG CONFIG
# -------------------------------------------------------

# Find the zingg jar
import zingg as _zingg_module
ZINGG_JAR = os.path.join(
    os.path.dirname(_zingg_module.__file__),
    "jars",
    "zingg-0.6.0.jar"
)

# Fallback if jar name differs
if not os.path.exists(ZINGG_JAR):
    jar_candidates = glob.glob(
        os.path.join(os.path.dirname(_zingg_module.__file__), "jars", "*.jar")
    )
    if jar_candidates:
        ZINGG_JAR = jar_candidates[0]
    else:
        raise FileNotFoundError("Zingg JAR not found!")

print(f"Using Zingg JAR: {ZINGG_JAR}", flush=True)

os.environ["PYSPARK_SUBMIT_ARGS"] = (
    f"--jars {ZINGG_JAR} "
    f"--driver-class-path {ZINGG_JAR} "
    "pyspark-shell"
)

from pyspark.sql import SparkSession
from zingg.client import (
    Arguments,
    ClientOptions,
    ZinggWithSpark,
    FieldDefinition,
    MatchType
)
from zingg.pipes import CsvPipe, Pipe

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

CONVERTED_DIR = "/home/it2022025/er_scalability/converted"
SPLITS_DIR    = "/home/it2022025/er_scalability/splits"
OUTPUT_CSV    = "/home/it2022025/er_scalability/zingg/zingg_scalability_results.csv"
ZINGG_BASE    = "/home/it2022025/er_scalability/zingg/zingg_work"

DATASETS = ["10K", "50K", "100K", "200K", "300K", "1M", "2M"]

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

def clear_output_dir(output_dir):
    for path in glob.glob(os.path.join(output_dir, "*.csv")):
        try:
            os.remove(path)
        except OSError:
            pass
    success_path = os.path.join(output_dir, "_SUCCESS")
    if os.path.exists(success_path):
        try:
            os.remove(success_path)
        except OSError:
            pass

def build_zingg_training(profiles_df, train_df, valid_df):
    """
    Build Zingg training DataFrame from labelled pairs.
    Same format as DER script: z_cluster, z_isMatch, z_zsource.
    Uses train + valid combined (same as CDDB DER script).
    """
    profiles_idx      = profiles_df.set_index("id")
    training_labeled  = pd.concat([train_df, valid_df], ignore_index=True)

    rows       = []
    cluster_id = 0

    for _, row in training_labeled.iterrows():
        left_id  = str(int(row["left_id"]))
        right_id = str(int(row["right_id"]))
        label    = int(row["label"])

        try:
            left  = profiles_idx.loc[left_id]
            right = profiles_idx.loc[right_id]
        except KeyError:
            continue

        for entity_id, record, side in [
            (left_id,  left,  "left"),
            (right_id, right, "right"),
        ]:
            rows.append({
                "id"          : str(entity_id),
                "given_name"  : str(record.get("given_name",   "")),
                "surname"     : str(record.get("surname",      "")),
                "suburb"      : str(record.get("suburb",       "")),
                "postcode"    : str(record.get("postcode",     "")),
                "state"       : str(record.get("state",        "")),
                "address_1"   : str(record.get("address_1",    "")),
                "date_of_birth": str(record.get("date_of_birth", "")),
                "z_cluster"   : cluster_id,
                "z_isMatch"   : label,
                "z_zsource"   : side,
            })

        cluster_id += 1

    return pd.DataFrame(rows), cluster_id

def get_pair_scores(split_df, entity_cluster, entity_scores):
    """
    Score pairs based on Zingg output clusters.
    Same logic as CDDB DER script.
    """
    scores, covered = [], 0
    for _, row in split_df.iterrows():
        left_id  = str(int(row["left_id"]))
        right_id = str(int(row["right_id"]))
        lc = entity_cluster.get(left_id)
        rc = entity_cluster.get(right_id)
        if lc is not None and lc == rc:
            ls    = float(entity_scores.get(left_id,  0.0))
            rs    = float(entity_scores.get(right_id, 0.0))
            score = min(ls, rs)
            if score > 0:
                covered += 1
        else:
            score = 0.0
        scores.append(score)
    return np.array(scores), covered

# -------------------------------------------------------
# PIPELINE
# -------------------------------------------------------

def run_pipeline(spark, profiles_df, train_df, valid_df, test_df, ds_label):
    """
    Full Zingg supervised pipeline.
    Same approach as DER CDDB script.
    """
    stage_times = {}
    mem_start   = mem_mb()
    total_start = time.time()

    zingg_dir        = os.path.join(ZINGG_BASE, ds_label)
    training_parquet = os.path.join(zingg_dir, "training_data")
    output_dir       = os.path.join(zingg_dir, "output")
    profiles_path    = os.path.join(
        CONVERTED_DIR, ds_label, "profiles.csv"
    )

    os.makedirs(zingg_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # -- Build training data --
    t0 = time.time()
    training_pd, n_clusters = build_zingg_training(
        profiles_df, train_df, valid_df
    )
    training_pd.to_parquet(training_parquet, index=False, engine="pyarrow")
    stage_times["build_training"] = time.time() - t0
    print(f"    [build_training] {stage_times['build_training']:.2f}s  |  "
          f"mem: {mem_mb():.0f} MB  |  clusters: {n_clusters:,}", flush=True)

    # -- Zingg Arguments --
    args = Arguments()
    args.setFieldDefinition([
        FieldDefinition("id",            "string", MatchType.DONT_USE),
        FieldDefinition("given_name",    "string", MatchType.FUZZY, MatchType.TEXT),
        FieldDefinition("surname",       "string", MatchType.FUZZY, MatchType.TEXT),
        FieldDefinition("address_1",     "string", MatchType.FUZZY),
        FieldDefinition("suburb",        "string", MatchType.EXACT),
        FieldDefinition("postcode",      "string", MatchType.EXACT),
        FieldDefinition("state",         "string", MatchType.EXACT),
        FieldDefinition("date_of_birth", "string", MatchType.EXACT),
    ])
    args.setModelId(ds_label)
    args.setZinggDir(zingg_dir)
    args.setNumPartitions(16)
    args.setLabelDataSampleSize(1.0)

    # Input data pipe
    data_pipe = CsvPipe(ds_label, profiles_path)
    data_pipe.addProperty("header",    "true")
    data_pipe.addProperty("delimiter", ",")
    args.setData(data_pipe)

    # Output pipe
    output_pipe = CsvPipe("output", output_dir)
    output_pipe.addProperty("header",    "true")
    output_pipe.addProperty("delimiter", "|")
    args.setOutput(output_pipe)

    # Training data pipe
    training_pipe = Pipe("training", "parquet")
    training_pipe.addProperty("location", training_parquet)
    args.setTrainingSamples(training_pipe)

    # -- Train phase --
    t0 = time.time()
    clear_output_dir(output_dir)
    ZinggWithSpark(
        args,
        ClientOptions([ClientOptions.PHASE, "train"])
    ).initAndExecute()
    stage_times["train"] = time.time() - t0
    print(f"    [train]          {stage_times['train']:.2f}s  |  mem: {mem_mb():.0f} MB", flush=True)

    # -- Match phase --
    t0 = time.time()
    ZinggWithSpark(
        args,
        ClientOptions([ClientOptions.PHASE, "match"])
    ).initAndExecute()
    stage_times["match"] = time.time() - t0
    print(f"    [match]          {stage_times['match']:.2f}s  |  mem: {mem_mb():.0f} MB", flush=True)

    # -- Read output --
    t0 = time.time()
    results_sdf = (
        spark.read
        .option("header", "true")
        .option("sep", "|")
        .csv(output_dir + "/*.csv")
    )
    match_output_pd = results_sdf.toPandas()

    # Find id column
    id_col = None
    for c in ["id", "Entity Id", "Entity id"]:
        if c in match_output_pd.columns:
            id_col = c
            break
    if id_col is None:
        raise KeyError(f"No entity id column. Columns: {list(match_output_pd.columns)}")

    score_col = "z_maxScore" if "z_maxScore" in match_output_pd.columns else "z_score"
    if score_col not in match_output_pd.columns:
        raise KeyError(f"No score column. Columns: {list(match_output_pd.columns)}")

    match_output_pd[id_col]      = match_output_pd[id_col].astype(str)
    match_output_pd["z_cluster"] = match_output_pd["z_cluster"].astype(str)
    match_output_pd[score_col]   = pd.to_numeric(
        match_output_pd[score_col], errors="coerce"
    ).fillna(0.0)

    entity_scores  = match_output_pd.set_index(id_col)[score_col].to_dict()
    entity_cluster = match_output_pd.set_index(id_col)["z_cluster"].to_dict()

    stage_times["read_output"] = time.time() - t0
    print(f"    [read_output]    {stage_times['read_output']:.2f}s  |  "
          f"mem: {mem_mb():.0f} MB  |  "
          f"records: {len(match_output_pd):,}", flush=True)

    # -- Score valid and test pairs --
    t0 = time.time()
    y_valid       = valid_df["label"].values
    y_test        = test_df["label"].values

    valid_scores, valid_covered = get_pair_scores(valid_df, entity_cluster, entity_scores)
    test_scores,  test_covered  = get_pair_scores(test_df,  entity_cluster, entity_scores)

    stage_times["scoring"] = time.time() - t0
    print(f"    [scoring]        {stage_times['scoring']:.2f}s  |  mem: {mem_mb():.0f} MB", flush=True)
    print(f"    Valid covered : {valid_covered:,} / {len(valid_df):,} "
          f"({100*valid_covered/len(valid_df):.1f}%)", flush=True)
    print(f"    Test covered  : {test_covered:,} / {len(test_df):,} "
          f"({100*test_covered/len(test_df):.1f}%)", flush=True)

    # -- Threshold sweep on valid set --
    t0 = time.time()
    thresholds    = np.arange(0.01, 1.00, 0.01)
    best_f1_valid = 0.0
    best_thresh   = 0.50

    for thresh in thresholds:
        preds = (valid_scores >= thresh).astype(int)
        f1    = f1_score(y_valid, preds, zero_division=0)
        if f1 > best_f1_valid:
            best_f1_valid = f1
            best_thresh   = thresh

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
            valid_covered, test_covered,
            len(match_output_pd))

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

print("\n" + "=" * 60)
print("  ZINGG SCALABILITY ANALYSIS")
print("=" * 60)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(ZINGG_BASE, exist_ok=True)

# -- Start Spark once for all datasets --
spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("ZinggScalability")
    .config("spark.driver.memory",           "32g")
    .config("spark.executor.memory",         "16g")
    .config("spark.driver.extraClassPath",   ZINGG_JAR)
    .config("spark.executor.extraClassPath", ZINGG_JAR)
    .config("spark.sql.shuffle.partitions",  "64")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setCheckpointDir(ZINGG_BASE)

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
        t_load       = time.time()
        profiles_df  = pd.read_csv(profiles_path, engine="python",
                                   na_filter=False).astype(str)
        train_df     = pd.read_csv(train_path,    engine="python")
        valid_df     = pd.read_csv(valid_path,    engine="python")
        test_df      = pd.read_csv(test_path,     engine="python")
        t_load       = time.time() - t_load

        print(f"  Records    : {len(profiles_df):,}")
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
         valid_covered, test_covered,
         n_output_records) = run_pipeline(
            spark, profiles_df, train_df, valid_df, test_df, ds
        )

        # -- Print --
        print(f"\n  --- RESULTS ---")
        print(f"  Output records : {n_output_records:,}")
        print(f"  Threshold      : {best_thresh:.2f}")
        print(f"  Precision      : {test_p:.4f}")
        print(f"  Recall         : {test_r:.4f}")
        print(f"  F1             : {test_f1:.4f}")
        print(f"  Total time     : {total_time:.2f}s  ({total_time/60:.1f} min)")
        print(f"  Mem used       : {mem_used:.0f} MB  |  Peak: {peak_mem:.0f} MB")

        results.append({
            "dataset"              : ds,
            "n_records"            : len(profiles_df),
            "n_train_pairs"        : len(train_df),
            "n_valid_pairs"        : len(valid_df),
            "n_test_pairs"         : len(test_df),
            "n_test_pos"           : int(test_df["label"].sum()),
            "n_output_records"     : n_output_records,
            "valid_covered"        : valid_covered,
            "test_covered"         : test_covered,
            "best_threshold"       : round(best_thresh,  2),
            "precision"            : round(test_p,       4),
            "recall"               : round(test_r,       4),
            "f1"                   : round(test_f1,      4),
            "time_load"            : round(t_load,                           2),
            "time_build_training"  : round(stage_times["build_training"],    2),
            "time_train"           : round(stage_times["train"],             2),
            "time_match"           : round(stage_times["match"],             2),
            "time_read_output"     : round(stage_times["read_output"],       2),
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

spark.stop()

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