import sys
import os
# Use the Zingg 0.5.0 python package (matches the 0.5.0 jar; avoids the 0.6.0
# license/registration step). Prepend so it shadows the 0.6.0 install in ~/.local.
ZINGG_050_PY = "/home/it2022025/software/zingg/zingg-0.5.0/python/build/lib"
if ZINGG_050_PY not in sys.path:
    sys.path.insert(0, ZINGG_050_PY)

import json
import time
import glob
import shutil
import random
import resource
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Zingg 0.5.0 jar (matches the 0.5.0 python package above). Overridable via env.
ZINGG_JAR = os.environ.get(
    "ZINGG_JAR",
    "/home/it2022025/software/zingg/zingg-0.5.0/zingg-0.5.0.jar",
)

_NCORES = os.environ.get("SLURM_CPUS_PER_TASK", "4")
# Set Spark resources at submit time. Zingg builds the SparkSession on import,
# so SparkSession.builder.config(...) later is ignored. These --conf flags are
# read when the JVM launches, which controls the training stage and prevents
# the treeAggregate stall.
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    f"--jars {ZINGG_JAR} --driver-class-path {ZINGG_JAR} "
    f"--master local[{_NCORES}] "
    f"--driver-memory 32g "
    f"--conf spark.sql.shuffle.partitions={_NCORES} "
    f"--conf spark.default.parallelism={_NCORES} "
    f"--conf spark.driver.host=127.0.0.1 "
    f"--conf spark.driver.bindAddress=127.0.0.1 "
    f"pyspark-shell"
)

from pyspark.sql import SparkSession
from zingg.client import Arguments, ClientOptions, ZinggWithSpark, FieldDefinition, MatchType
from zingg.pipes import CsvPipe, Pipe

CORA_PATH  = "/home/it2022025/er_scalability/datasets/cora/cora.csv"
TRAIN_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/cora/train_set.csv"
VALID_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/cora/valid_set.csv"
TEST_PATH  = "/home/it2022025/er_scalability/train_validation_test_sets/cora/test_set.csv"

ID_COL = "Entity Id"

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    cfg = json.loads(sys.argv[1])
    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed)
    t_start = time.time()

    cid = cfg.get("config_id", 0)
    zingg_dir = f"/tmp/zingg_cora_cfg{cid}"
    if os.path.exists(zingg_dir):
        shutil.rmtree(zingg_dir, ignore_errors=True)
    os.makedirs(zingg_dir, exist_ok=True)
    training_parquet = f"{zingg_dir}/training_data"
    output_dir = f"{zingg_dir}/output"
    model_id = f"cora_cfg{cid}"

    cora_df = pd.read_csv(CORA_PATH, sep="|", engine="python", na_filter=False)
    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)
    cora_idx = cora_df.set_index(ID_COL)
    y_valid = valid_df["label"].values
    y_test = test_df["label"].values

    # ---- build Zingg training data (labelled pairs) from TRAIN only ----
    matches = train_df[train_df["label"] == 1]
    nonmatches = train_df[train_df["label"] == 0]
    neg_ratio = float(cfg["neg_ratio"])
    n_neg = int(min(len(nonmatches), round(len(matches) * neg_ratio)))
    nonmatches = (nonmatches.sample(n=n_neg, random_state=seed)
                  if n_neg < len(nonmatches) else nonmatches)
    train_sample = pd.concat([matches, nonmatches], ignore_index=True)

    rows = []
    cluster_id = 0
    for _, row in train_sample.iterrows():
        lid, rid, label = int(row["left_id"]), int(row["right_id"]), int(row["label"])
        try:
            left, right = cora_idx.loc[lid], cora_idx.loc[rid]
        except KeyError:
            continue
        rows.append({ID_COL: str(lid),
                     "title": str(left.get("title", "")),
                     "author": str(left.get("author", "")),
                     "venue": str(left.get("venue", "")),
                     "publisher": str(left.get("publisher", "")),
                     "year": str(left.get("year", "")),
                     "z_cluster": cluster_id, "z_isMatch": label, "z_zsource": "left"})
        rows.append({ID_COL: str(rid),
                     "title": str(right.get("title", "")),
                     "author": str(right.get("author", "")),
                     "venue": str(right.get("venue", "")),
                     "publisher": str(right.get("publisher", "")),
                     "year": str(right.get("year", "")),
                     "z_cluster": cluster_id, "z_isMatch": label, "z_zsource": "right"})
        cluster_id += 1
    pd.DataFrame(rows).to_parquet(training_parquet, index=False, engine="pyarrow")

    # ---- Spark (session config already set via PYSPARK_SUBMIT_ARGS) ----
    spark = SparkSession.builder.appName(f"ZinggCORAcfg{cid}").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir(zingg_dir)

    # ---- Zingg arguments: SINGLE-SOURCE dedup (Dirty ER) ----
    args = Arguments()
    args.setFieldDefinition([
        FieldDefinition(ID_COL, "string", MatchType.DONT_USE),
        FieldDefinition("title", "string", MatchType.FUZZY, MatchType.TEXT),
        FieldDefinition("author", "string", MatchType.FUZZY, MatchType.TEXT),
        FieldDefinition("venue", "string", MatchType.FUZZY, MatchType.TEXT),
        FieldDefinition("publisher", "string", MatchType.DONT_USE),
        FieldDefinition("year", "string", MatchType.DONT_USE),
    ])
    args.setModelId(model_id)
    args.setZinggDir(zingg_dir)
    args.setNumPartitions(int(cfg["numPartitions"]))
    args.setLabelDataSampleSize(float(cfg["labelDataSampleSize"]))

    data_pipe = CsvPipe("cora", CORA_PATH)
    data_pipe.addProperty("header", "true"); data_pipe.addProperty("delimiter", "|")
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
    spark.stop()

    # ---- DER scoring: entities in the SAME z_cluster are predicted duplicates,
    #      pair score = min(entity scores) (matches the laptop get_pair_scores) ----
    id_col = None
    for c in [ID_COL, "Entity id", "id"]:
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

    valid_scores = scores_for(valid_df)
    test_scores = scores_for(test_df)

    # ---- threshold chosen on VALIDATION, reported on TEST (3-way split) ----
    best_t, best_valid_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        preds = (valid_scores >= t).astype(int)
        f1 = f1_score(y_valid, preds, zero_division=0)
        if f1 > best_valid_f1:
            best_valid_f1, best_t = f1, float(t)

    preds_test = (test_scores >= best_t).astype(int)
    test_p = precision_score(y_test, preds_test, zero_division=0)
    test_r = recall_score(y_test, preds_test, zero_division=0)
    test_f1 = f1_score(y_test, preds_test, zero_division=0)

    curve = []
    for t in THRESHOLD_GRID:
        preds = (test_scores >= t).astype(int)
        curve.append({"t": round(float(t), 3),
                      "precision": round(precision_score(y_test, preds, zero_division=0), 6),
                      "recall": round(recall_score(y_test, preds, zero_division=0), 6),
                      "f1": round(f1_score(y_test, preds, zero_division=0), 6)})

    shutil.rmtree(zingg_dir, ignore_errors=True)

    result = {
        "config_id": cid,
        "params": {k: cfg[k] for k in ["numPartitions", "labelDataSampleSize", "neg_ratio"]},
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