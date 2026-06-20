import sys
import os
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

ZINGG_JAR = "/home/thomas/miniconda3/envs/zingg/lib/python3.10/site-packages/zingg/jars/zingg-0.5.0.jar"
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    f"--jars {ZINGG_JAR} --driver-class-path {ZINGG_JAR} pyspark-shell"
)

from pyspark.sql import SparkSession
from zingg.client import Arguments, ClientOptions, ZinggWithSpark, FieldDefinition, MatchType
from zingg.pipes import CsvPipe, Pipe

ABT_PATH   = "/home/thomas/pyJedAI/data/ccer/D2/abt.csv"
BUY_PATH   = "/home/thomas/pyJedAI/data/ccer/D2/buy.csv"
TRAIN_PATH = "/home/thomas/train_test_valid_datasets/db2/train_set.csv"
VALID_PATH = "/home/thomas/train_test_valid_datasets/db2/valid_set.csv"
TEST_PATH  = "/home/thomas/train_test_valid_datasets/db2/test_set.csv"

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    cfg = json.loads(sys.argv[1])
    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed)
    t_start = time.time()

    cid = cfg.get("config_id", 0)
    zingg_dir = f"/tmp/zingg_d2_cfg{cid}"
    if os.path.exists(zingg_dir):
        shutil.rmtree(zingg_dir, ignore_errors=True)
    os.makedirs(zingg_dir, exist_ok=True)
    training_parquet = f"{zingg_dir}/training_data"
    output_dir = f"{zingg_dir}/output"
    model_id = f"d2_cfg{cid}"

    abt_df = pd.read_csv(ABT_PATH, sep="|", engine="python", na_filter=False)
    buy_df = pd.read_csv(BUY_PATH, sep="|", engine="python", na_filter=False)
    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)
    abt_idx = abt_df.set_index("id")
    buy_idx = buy_df.set_index("id")
    y_valid = valid_df["label"].values
    y_test = test_df["label"].values

    matches = train_df[train_df["label"] == 1]
    nonmatches = train_df[train_df["label"] == 0]
    neg_ratio = float(cfg["neg_ratio"])
    n_neg = int(min(len(nonmatches), round(len(matches) * neg_ratio)))
    nonmatches = nonmatches.sample(n=n_neg, random_state=seed) if n_neg < len(nonmatches) else nonmatches
    train_sample = pd.concat([matches, nonmatches], ignore_index=True)

    rows = []
    cluster_id = 0
    for _, row in train_sample.iterrows():
        lid, rid, label = int(row["left_id"]), int(row["right_id"]), int(row["label"])
        try:
            left, right = abt_idx.loc[lid], buy_idx.loc[rid]
        except KeyError:
            continue
        rows.append({"id": str(lid), "name": str(left["name"]),
                     "description": str(left["description"]), "price": str(left["price"]),
                     "z_cluster": cluster_id, "z_isMatch": label, "z_zsource": "abt"})
        rows.append({"id": str(rid), "name": str(right["name"]),
                     "description": str(right["description"]), "price": str(right["price"]),
                     "z_cluster": cluster_id, "z_isMatch": label, "z_zsource": "buy"})
        cluster_id += 1
    pd.DataFrame(rows).to_parquet(training_parquet, index=False, engine="pyarrow")

    spark = (SparkSession.builder.master("local[*]").appName(f"ZinggD2cfg{cid}")
             .config("spark.driver.memory", "6g")
             .config("spark.driver.extraClassPath", ZINGG_JAR)
             .config("spark.executor.extraClassPath", ZINGG_JAR)
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    args = Arguments()
    args.setFieldDefinition([
        FieldDefinition("id", "string", MatchType.DONT_USE),
        FieldDefinition("name", "string", MatchType.FUZZY, MatchType.TEXT),
        FieldDefinition("description", "string", MatchType.FUZZY),
        FieldDefinition("price", "string", MatchType.DONT_USE),
    ])
    args.setModelId(model_id)
    args.setZinggDir(zingg_dir)
    args.setNumPartitions(int(cfg["numPartitions"]))
    args.setLabelDataSampleSize(float(cfg["labelDataSampleSize"]))

    abt_pipe = CsvPipe("abt", ABT_PATH); abt_pipe.addProperty("header", "true"); abt_pipe.addProperty("delimiter", "|")
    buy_pipe = CsvPipe("buy", BUY_PATH); buy_pipe.addProperty("header", "true"); buy_pipe.addProperty("delimiter", "|")
    args.setData(abt_pipe, buy_pipe)
    out_pipe = CsvPipe("output", output_dir); out_pipe.addProperty("header", "true"); out_pipe.addProperty("delimiter", "|")
    args.setOutput(out_pipe)
    tr_pipe = Pipe("training", "parquet"); tr_pipe.addProperty("location", training_parquet)
    args.setTrainingSamples(tr_pipe)

    ZinggWithSpark(args, ClientOptions([ClientOptions.PHASE, "train"])).initAndExecute()
    ZinggWithSpark(args, ClientOptions([ClientOptions.PHASE, "link"])).initAndExecute()

    output_df = (spark.read.option("header", "true").option("delimiter", "|")
                 .csv(output_dir + "/*.csv").toPandas())

    score_map = {}
    for _, group in output_df.groupby("z_cluster"):
        sources = group["z_zsource"].values
        ids = group["id"].values
        scores = group["z_score"].astype(float).values
        abt_ids = [ids[i] for i, s in enumerate(sources) if s == "abt"]
        buy_ids = [ids[i] for i, s in enumerate(sources) if s == "buy"]
        score = float(scores.max())
        for a in abt_ids:
            for b in buy_ids:
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