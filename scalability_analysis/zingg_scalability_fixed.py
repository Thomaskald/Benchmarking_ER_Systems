"""
zingg_scalability_fixed.py
--------------------------
Fixed-config scalability runs for Zingg on synthetic FEBRL datasets.
Uses the single BEST config chosen from the 10K B=50 search (by valid_f1),
applied verbatim to every size -- INCLUDING a frozen decision threshold,
exactly like pyjedai_scalability_fixed.py. No per-dataset threshold sweep,
no valid set at scale.

    BEST config (10K search, config #6):
        numPartitions        = 43
        labelDataSampleSize  = 0.279
        neg_ratio            = 6.849
        threshold            = 0.01     (valid_f1 = 0.902819)

Note: Zingg emits hard cluster decisions, so the threshold is effectively inert
(0.01 = "use Zingg's clusters as-is"); it is kept frozen for consistency.

MEMORY CAVEAT: Zingg does its work in a child JVM (Spark). The peak_mem_mb below
is the PYTHON process only and badly understates real usage. For the memory axis
use SLURM `sacct -j <jobid> --format=MaxRSS` (captures the JVM too).

10K is excluded (it was the tuning size).
Loops: 50K, 100K, 200K, 300K, 1M, 2M -- one serial job, checkpoint after each.
Failure policy: OOM / ERROR recorded per dataset, not crashes.

Output: zingg_scalability_results.csv
"""

import sys
import os
# Use the Zingg 0.5.0 python package (matches the 0.5.0 jar; avoids the 0.6.0
# license/registration step). Prepend so it shadows the 0.6.0 install in ~/.local.
ZINGG_050_PY = "/home/it2022025/software/zingg/zingg-0.5.0/python/build/lib"
if ZINGG_050_PY not in sys.path:
    sys.path.insert(0, ZINGG_050_PY)

import json
import time
import shutil
import random
import resource
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

ZINGG_JAR = os.environ.get(
    "ZINGG_JAR",
    "/home/it2022025/software/zingg/zingg-0.5.0/zingg-0.5.0.jar",
)

_NCORES = os.environ.get("SLURM_CPUS_PER_TASK", "4")
# Zingg builds the SparkSession on import, so config must be set via
# PYSPARK_SUBMIT_ARGS BEFORE importing zingg (prevents the treeAggregate stall).
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    f"--jars {ZINGG_JAR} --driver-class-path {ZINGG_JAR} "
    f"--master local[{_NCORES}] "
    f"--driver-memory 48g "
    f"--conf spark.sql.shuffle.partitions={_NCORES} "
    f"--conf spark.default.parallelism={_NCORES} "
    f"--conf spark.driver.host=127.0.0.1 "
    f"--conf spark.driver.bindAddress=127.0.0.1 "
    f"pyspark-shell"
)

from pyspark.sql import SparkSession
from zingg.client import Arguments, ClientOptions, ZinggWithSpark, FieldDefinition, MatchType
from zingg.pipes import CsvPipe, Pipe

CONVERTED_DIR = "/home/it2022025/er_scalability/converted"
SPLITS_DIR    = "/home/it2022025/er_scalability/splits"
OUTPUT_CSV    = "/home/it2022025/er_scalability/scalability/zingg_scalability_results.csv"

ID_COL = "id"
DATASETS = ["50K", "100K", "200K", "300K", "1M", "2M"]

FEBRL_COLS = ["given_name", "surname", "address_1", "suburb", "postcode",
              "state", "date_of_birth", "soc_sec_id", "phone_number"]

# ---- BEST CONFIG from the 10K B=50 search (fixed for all sizes) ----
# Selected by argmax(valid_f1)=0.902819 -> config #6 (unique max).
BEST_NUM_PARTITIONS   = 43
BEST_LABEL_SAMPLE_SIZE = 0.279
BEST_NEG_RATIO        = 6.849
BEST_THRESHOLD        = 0.01      # frozen from 10K -- NOT re-tuned per dataset

SEED = 42
random.seed(SEED); np.random.seed(SEED)


def peak_mem_mb():
    # NOTE: Python process only -- see MEMORY CAVEAT in the module docstring.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_pipeline(spark, ds_label):
    """Train Zingg on this dataset's TRAIN (frozen config) and match its
    profiles, then score TEST at the frozen threshold. No valid set."""
    workflow_start = time.time()

    profiles_path = os.path.join(CONVERTED_DIR, ds_label, "profiles.csv")
    train_path    = os.path.join(SPLITS_DIR,    ds_label, "train_set.csv")
    test_path     = os.path.join(SPLITS_DIR,    ds_label, "test_set.csv")

    zingg_dir = f"/tmp/zingg_scale_fixed_{ds_label}"
    shutil.rmtree(zingg_dir, ignore_errors=True)
    os.makedirs(zingg_dir, exist_ok=True)
    training_parquet = f"{zingg_dir}/training_data"
    output_dir = f"{zingg_dir}/output"
    model_id = f"scale_fixed_{ds_label}"
    spark.sparkContext.setCheckpointDir(zingg_dir)

    profiles_df = pd.read_csv(profiles_path, engine="python", na_filter=False)  # comma-delim
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    prof_idx = profiles_df.set_index(ID_COL)
    y_test = test_df["label"].values

    # ---- build Zingg training data (labelled pairs) from TRAIN only ----
    matches = train_df[train_df["label"] == 1]
    nonmatches = train_df[train_df["label"] == 0]
    n_neg = int(min(len(nonmatches), round(len(matches) * BEST_NEG_RATIO)))
    nonmatches = (nonmatches.sample(n=n_neg, random_state=SEED)
                  if n_neg < len(nonmatches) else nonmatches)
    train_sample = pd.concat([matches, nonmatches], ignore_index=True)

    rows = []
    cluster_id = 0
    for _, row in train_sample.iterrows():
        lid, rid, label = int(row["left_id"]), int(row["right_id"]), int(row["label"])
        try:
            left  = prof_idx.loc[str(lid)] if str(lid) in prof_idx.index else prof_idx.loc[lid]
            right = prof_idx.loc[str(rid)] if str(rid) in prof_idx.index else prof_idx.loc[rid]
        except KeyError:
            continue
        for eid, rec, side in [(lid, left, "left"), (rid, right, "right")]:
            entry = {ID_COL: str(eid)}
            for c in FEBRL_COLS:
                entry[c] = str(rec.get(c, ""))
            entry.update({"z_cluster": cluster_id, "z_isMatch": label, "z_zsource": side})
            rows.append(entry)
        cluster_id += 1
    pd.DataFrame(rows).to_parquet(training_parquet, index=False, engine="pyarrow")

    # ---- Zingg arguments: SINGLE-SOURCE dedup (frozen FEBRL field MatchTypes) ----
    args = Arguments()
    args.setFieldDefinition([
        FieldDefinition(ID_COL, "string", MatchType.DONT_USE),
        FieldDefinition("given_name", "string", MatchType.FUZZY, MatchType.TEXT),
        FieldDefinition("surname", "string", MatchType.FUZZY, MatchType.TEXT),
        FieldDefinition("address_1", "string", MatchType.FUZZY),
        FieldDefinition("suburb", "string", MatchType.EXACT),
        FieldDefinition("postcode", "string", MatchType.EXACT),
        FieldDefinition("state", "string", MatchType.EXACT),
        FieldDefinition("date_of_birth", "string", MatchType.EXACT),
        FieldDefinition("soc_sec_id", "string", MatchType.EXACT),
        FieldDefinition("phone_number", "string", MatchType.FUZZY),
    ])
    args.setModelId(model_id)
    args.setZinggDir(zingg_dir)
    args.setNumPartitions(int(BEST_NUM_PARTITIONS))
    args.setLabelDataSampleSize(float(BEST_LABEL_SAMPLE_SIZE))

    data_pipe = CsvPipe("febrl", profiles_path)
    data_pipe.addProperty("header", "true"); data_pipe.addProperty("delimiter", ",")
    args.setData(data_pipe)   # SINGLE source
    out_pipe = CsvPipe("output", output_dir)
    out_pipe.addProperty("header", "true"); out_pipe.addProperty("delimiter", "|")
    args.setOutput(out_pipe)
    tr_pipe = Pipe("training", "parquet"); tr_pipe.addProperty("location", training_parquet)
    args.setTrainingSamples(tr_pipe)

    ZinggWithSpark(args, ClientOptions([ClientOptions.PHASE, "train"])).initAndExecute()
    ZinggWithSpark(args, ClientOptions([ClientOptions.PHASE, "match"])).initAndExecute()

    output_df = (spark.read.option("header", "true").option("sep", "|")
                 .csv(output_dir + "/*.csv").toPandas())

    # ---- DER scoring: same z_cluster => predicted duplicates, score=min(entity scores) ----
    id_col = None
    for c in [ID_COL, "Entity Id", "Entity id"]:
        if c in output_df.columns:
            id_col = c
            break
    if id_col is None:
        raise KeyError(f"No entity id column. Columns: {list(output_df.columns)}")
    score_col = "z_maxScore" if "z_maxScore" in output_df.columns else "z_score"
    if score_col not in output_df.columns:
        raise KeyError(f"No score column. Columns: {list(output_df.columns)}")

    output_df[id_col] = output_df[id_col].astype(str)
    output_df["z_cluster"] = output_df["z_cluster"].astype(str)
    output_df[score_col] = pd.to_numeric(output_df[score_col], errors="coerce").fillna(0.0)
    entity_scores = output_df.set_index(id_col)[score_col].to_dict()
    entity_cluster = output_df.set_index(id_col)["z_cluster"].to_dict()

    def scores_for(split_df):
        out = []
        for _, row in split_df.iterrows():
            lid, rid = str(int(row["left_id"])), str(int(row["right_id"]))
            lc, rc = entity_cluster.get(lid), entity_cluster.get(rid)
            if lc is not None and lc == rc:
                out.append(min(float(entity_scores.get(lid, 0.0)),
                               float(entity_scores.get(rid, 0.0))))
            else:
                out.append(0.0)
        return np.array(out)

    test_scores = scores_for(test_df)
    preds_test = (test_scores >= BEST_THRESHOLD).astype(int)
    test_p = precision_score(y_test, preds_test, zero_division=0)
    test_r = recall_score(y_test, preds_test, zero_division=0)
    test_f1 = f1_score(y_test, preds_test, zero_division=0)

    shutil.rmtree(zingg_dir, ignore_errors=True)
    return len(profiles_df), len(test_df), int(test_df["label"].sum()), \
        test_p, test_r, test_f1, time.time() - workflow_start


# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------

print("\n" + "=" * 60)
print("  ZINGG SCALABILITY (fixed best config)")
print(f"  numPartitions={BEST_NUM_PARTITIONS} | labelSampleSize={BEST_LABEL_SAMPLE_SIZE} | "
      f"neg_ratio={BEST_NEG_RATIO} | thr={BEST_THRESHOLD}")
print("=" * 60)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

spark = SparkSession.builder.appName("ZinggScaleFixed").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

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
        n_rec, n_test, n_pos, test_p, test_r, test_f1, t_workflow = run_pipeline(spark, ds)

        print(f"\n  --- RESULTS ---")
        print(f"  Records    : {n_rec:,}  |  Test pairs: {n_test:,} (pos={n_pos:,})")
        print(f"  Precision  : {test_p:.4f}  Recall: {test_r:.4f}  F1: {test_f1:.4f}")
        print(f"  Workflow   : {t_workflow:.2f}s  ({t_workflow/60:.1f} min)")
        print(f"  Peak mem   : {peak_mem_mb():.0f} MB (python only -- use sacct MaxRSS)", flush=True)

        results.append({
            "dataset": ds, "n_records": n_rec, "n_test_pairs": n_test,
            "precision": round(test_p, 4), "recall": round(test_r, 4), "f1": round(test_f1, 4),
            "time_workflow": round(t_workflow, 2),
            "peak_mem_mb": round(peak_mem_mb(), 1), "status": "OK",
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

spark.stop()

pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print("\n" + "=" * 60)
print("  SCALABILITY SUMMARY")
print("=" * 60)
dfr = pd.DataFrame(results)
cols = [c for c in ["dataset","n_records","precision","recall","f1","time_workflow","peak_mem_mb","status"] if c in dfr.columns]
print(dfr[cols].to_string(index=False))
print(f"\nSaved to: {OUTPUT_CSV}")
