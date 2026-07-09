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

# NOTE: confirmed from an actual cluster run - the D5 entity tables live
# under datasets/D5/ as imdb.csv / tmdb.csv, '|'-delimited, with RDF-style
# URI column headers (e.g. "https://www.scads.de/movieBenchmark/ontology/title")
# instead of plain "title"/"name"/"genre_list", plus several extra columns we
# don't use. Splits (train/valid/test) remain under
# train_validation_test_sets/db5/ with the usual left_id/right_id/label schema.
IMDB_PATH  = "/home/it2022025/er_scalability/datasets/D5/imdb.csv"
TMDB_PATH  = "/home/it2022025/er_scalability/datasets/D5/tmdb.csv"
TRAIN_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/db5/train_set.csv"
VALID_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/db5/valid_set.csv"
TEST_PATH  = "/home/it2022025/er_scalability/train_validation_test_sets/db5/test_set.csv"

# Real column headers in imdb.csv / tmdb.csv (shared by both tables) for the
# three fields we actually match on.
TITLE_COL = "https://www.scads.de/movieBenchmark/ontology/title"
NAME_COL  = "https://www.scads.de/movieBenchmark/ontology/name"
GENRE_COL = "https://www.scads.de/movieBenchmark/ontology/genre_list"

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_clean(path):
    """Read the raw '|'-delimited, URI-headered entity table and reduce it to
    the plain id/title/name/genre_list columns we use for matching."""
    raw = pd.read_csv(path, sep="|", engine="python", na_filter=False)
    df = raw[["id", TITLE_COL, NAME_COL, GENRE_COL]].rename(columns={
        TITLE_COL: "title", NAME_COL: "name", GENRE_COL: "genre_list",
    })
    return df


def main():
    cfg = json.loads(sys.argv[1])
    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed)
    t_start = time.time()

    cid = cfg.get("config_id", 0)
    zingg_dir = f"/tmp/zingg_d5_cfg{cid}"
    if os.path.exists(zingg_dir):
        shutil.rmtree(zingg_dir, ignore_errors=True)
    os.makedirs(zingg_dir, exist_ok=True)
    training_parquet = f"{zingg_dir}/training_data"
    output_dir = f"{zingg_dir}/output"
    model_id = f"d5_cfg{cid}"

    imdb_df = load_clean(IMDB_PATH)
    tmdb_df = load_clean(TMDB_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)
    imdb_idx = imdb_df.set_index("id")
    tmdb_idx = tmdb_df.set_index("id")
    y_valid = valid_df["label"].values
    y_test = test_df["label"].values

    # ---- write cleaned, comma-delimited copies for Zingg's own CsvPipe to
    # read (it reads the raw entity files directly, independent of pandas
    # above, so it needs the same simplified id/title/name/genre_list schema
    # rather than the original '|'-delimited URI-headered files) ----
    imdb_clean_path = f"{zingg_dir}/imdb_clean.csv"
    tmdb_clean_path = f"{zingg_dir}/tmdb_clean.csv"
    imdb_df.to_csv(imdb_clean_path, index=False)
    tmdb_df.to_csv(tmdb_clean_path, index=False)

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
            left, right = imdb_idx.loc[lid], tmdb_idx.loc[rid]
        except KeyError:
            continue
        rows.append({"id": str(lid), "title": str(left["title"]),
                     "name": str(left["name"]), "genre_list": str(left["genre_list"]),
                     "z_cluster": cluster_id, "z_isMatch": label, "z_zsource": "imdb"})
        rows.append({"id": str(rid), "title": str(right["title"]),
                     "name": str(right["name"]), "genre_list": str(right["genre_list"]),
                     "z_cluster": cluster_id, "z_isMatch": label, "z_zsource": "tmdb"})
        cluster_id += 1
    pd.DataFrame(rows).to_parquet(training_parquet, index=False, engine="pyarrow")

    # ---- Spark (session config already set via PYSPARK_SUBMIT_ARGS) ----
    spark = SparkSession.builder.appName(f"ZinggD5cfg{cid}").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # ---- Zingg arguments: TWO-SOURCE LINK (Clean-Clean ER, imdb <-> tmdb) ----
    args = Arguments()
    args.setFieldDefinition([
        FieldDefinition("id", "string", MatchType.DONT_USE),
        FieldDefinition("title", "string", MatchType.FUZZY, MatchType.TEXT),
        FieldDefinition("name", "string", MatchType.FUZZY),
        FieldDefinition("genre_list", "string", MatchType.FUZZY),
    ])
    args.setModelId(model_id)
    args.setZinggDir(zingg_dir)
    args.setNumPartitions(int(cfg["numPartitions"]))
    args.setLabelDataSampleSize(float(cfg["labelDataSampleSize"]))

    imdb_pipe = CsvPipe("imdb", imdb_clean_path); imdb_pipe.addProperty("header", "true"); imdb_pipe.addProperty("delimiter", ",")
    tmdb_pipe = CsvPipe("tmdb", tmdb_clean_path); tmdb_pipe.addProperty("header", "true"); tmdb_pipe.addProperty("delimiter", ",")
    args.setData(imdb_pipe, tmdb_pipe)
    out_pipe = CsvPipe("output", output_dir); out_pipe.addProperty("header", "true"); out_pipe.addProperty("delimiter", "|")
    args.setOutput(out_pipe)
    tr_pipe = Pipe("training", "parquet"); tr_pipe.addProperty("location", training_parquet)
    args.setTrainingSamples(tr_pipe)

    # train, then LINK (cross-source), proven phases now that Spark has memory
    ZinggWithSpark(args, ClientOptions([ClientOptions.PHASE, "train"])).initAndExecute()
    ZinggWithSpark(args, ClientOptions([ClientOptions.PHASE, "link"])).initAndExecute()

    output_df = (spark.read.option("header", "true").option("delimiter", "|")
                 .csv(output_dir + "/*.csv").toPandas())

    # ---- score pairs: within each output z_cluster, link imdb ids to tmdb ids ----
    score_map = {}
    for _, group in output_df.groupby("z_cluster"):
        sources = group["z_zsource"].values
        ids = group["id"].values
        scores = group["z_score"].astype(float).values
        imdb_ids = [ids[i] for i, s in enumerate(sources) if s == "imdb"]
        tmdb_ids = [ids[i] for i, s in enumerate(sources) if s == "tmdb"]
        score = float(scores.max())
        for a in imdb_ids:
            for b in tmdb_ids:
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