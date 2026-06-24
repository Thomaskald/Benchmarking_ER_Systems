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
# so SparkSession.builder.config(...) later is ignored ("Using an existing
# Spark session"). These --conf flags are read when the JVM launches, which is
# what actually controls the training stage and prevents the treeAggregate stall.
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

# --- D8 (Amazon <-> Walmart electronics) paths, following the same layout
# convention as D2/D7. NOTE: inferred by analogy — the reference D8 script
# you sent used /home/thomas/train_test_valid_datasets/db8/{tableA,tableB}.csv
# (i.e. table A == Amazon, table B == Walmart, both alongside the splits).
# Double check these against your actual it2022025 layout and swap the two
# file names if your tableA/tableB assignment is the other way round.
AMAZON_PATH  = "/home/it2022025/er_scalability/datasets/D8/amazon.csv"
WALMART_PATH = "/home/it2022025/er_scalability/datasets/D8/walmart.csv"
TRAIN_PATH   = "/home/it2022025/er_scalability/train_validation_test_sets/db8/train_set.csv"
VALID_PATH   = "/home/it2022025/er_scalability/train_validation_test_sets/db8/valid_set.csv"
TEST_PATH    = "/home/it2022025/er_scalability/train_validation_test_sets/db8/test_set.csv"

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    cfg = json.loads(sys.argv[1])
    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed)
    t_start = time.time()

    cid = cfg.get("config_id", 0)
    zingg_dir = f"/tmp/zingg_d8_cfg{cid}"
    if os.path.exists(zingg_dir):
        shutil.rmtree(zingg_dir, ignore_errors=True)
    os.makedirs(zingg_dir, exist_ok=True)
    training_parquet = f"{zingg_dir}/training_data"
    output_dir = f"{zingg_dir}/output"
    model_id = f"d8_cfg{cid}"

    # D8 tables are comma-delimited (no sep="|" like D2's abt/buy).
    amazon_df = pd.read_csv(AMAZON_PATH, engine="python", na_filter=False)
    walmart_df = pd.read_csv(WALMART_PATH, engine="python", na_filter=False)
    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)
    amazon_idx = amazon_df.set_index("id")
    walmart_idx = walmart_df.set_index("id")
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
            left, right = amazon_idx.loc[lid], walmart_idx.loc[rid]
        except KeyError:
            continue
        rows.append({"id": str(lid), "title": str(left["title"]),
                     "modelno": str(left["modelno"]), "brand": str(left["brand"]),
                     "z_cluster": cluster_id, "z_isMatch": label, "z_zsource": "amazon"})
        rows.append({"id": str(rid), "title": str(right["title"]),
                     "modelno": str(right["modelno"]), "brand": str(right["brand"]),
                     "z_cluster": cluster_id, "z_isMatch": label, "z_zsource": "walmart"})
        cluster_id += 1
    pd.DataFrame(rows).to_parquet(training_parquet, index=False, engine="pyarrow")

    # ---- Spark (session config already set via PYSPARK_SUBMIT_ARGS) ----
    spark = SparkSession.builder.appName(f"ZinggD8cfg{cid}").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # ---- Zingg arguments: TWO-SOURCE LINK (Clean-Clean ER, amazon <-> walmart) ----
    args = Arguments()
    args.setFieldDefinition([
        FieldDefinition("id", "string", MatchType.DONT_USE),
        FieldDefinition("title", "string", MatchType.FUZZY, MatchType.TEXT),
        FieldDefinition("modelno", "string", MatchType.FUZZY),
        FieldDefinition("brand", "string", MatchType.FUZZY),
    ])
    args.setModelId(model_id)
    args.setZinggDir(zingg_dir)
    args.setNumPartitions(int(cfg["numPartitions"]))
    args.setLabelDataSampleSize(float(cfg["labelDataSampleSize"]))

    amazon_pipe = CsvPipe("amazon", AMAZON_PATH); amazon_pipe.addProperty("header", "true"); amazon_pipe.addProperty("delimiter", ",")
    walmart_pipe = CsvPipe("walmart", WALMART_PATH); walmart_pipe.addProperty("header", "true"); walmart_pipe.addProperty("delimiter", ",")
    args.setData(amazon_pipe, walmart_pipe)
    out_pipe = CsvPipe("output", output_dir); out_pipe.addProperty("header", "true"); out_pipe.addProperty("delimiter", "|")
    args.setOutput(out_pipe)
    tr_pipe = Pipe("training", "parquet"); tr_pipe.addProperty("location", training_parquet)
    args.setTrainingSamples(tr_pipe)

    # train, then LINK (cross-source), proven phases now that Spark has memory
    ZinggWithSpark(args, ClientOptions([ClientOptions.PHASE, "train"])).initAndExecute()
    ZinggWithSpark(args, ClientOptions([ClientOptions.PHASE, "link"])).initAndExecute()

    output_df = (spark.read.option("header", "true").option("delimiter", "|")
                 .csv(output_dir + "/*.csv").toPandas())

    # ---- score pairs: within each output z_cluster, link amazon ids to walmart ids ----
    score_map = {}
    for _, group in output_df.groupby("z_cluster"):
        sources = group["z_zsource"].values
        ids = group["id"].values
        scores = group["z_score"].astype(float).values
        amazon_ids = [ids[i] for i, s in enumerate(sources) if s == "amazon"]
        walmart_ids = [ids[i] for i, s in enumerate(sources) if s == "walmart"]
        score = float(scores.max())
        for a in amazon_ids:
            for b in walmart_ids:
                score_map[(str(a), str(b))] = score
                score_map[(str(b), str(a))] = score

    def scores_for(split_df):
        out = []
        for _, row in split_df.iterrows():
            key = (str(int(row["left_id"])), str(int(row["right_id"])))
            out.append(score_map.get(key, 0.0))
        return np.array(out)

    valid_scores = scores_for(valid_df)
    test_scores = scores_for(test_df)
    spark.stop()

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