import sys
import os
# Use the Zingg 0.5.0 python package (matches the 0.5.0 jar; avoids the 0.6.0
# license/registration step). Prepend so it shadows the 0.6.0 install in ~/.local.
ZINGG_050_PY = "/home/it2022025/software/zingg/zingg-0.5.0/python/build/lib"
if ZINGG_050_PY not in sys.path:
    sys.path.insert(0, ZINGG_050_PY)

import csv
import json
import time
import shutil
import random
import warnings
import logging
import subprocess
from collections import Counter
from itertools import combinations

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy as np

# Zingg 0.5.0 jar (matches the 0.5.0 python package above).
ZINGG_JAR = os.environ.get(
    "ZINGG_JAR",
    "/home/it2022025/software/zingg/zingg-0.5.0/zingg-0.5.0.jar",
)

_NCORES = os.environ.get("SLURM_CPUS_PER_TASK", "4")
# Set Spark resources at submit time -- Zingg builds the SparkSession on import,
# so this must be in place before `import zingg` in the child process. Same flags
# the workers use (prevents the LogisticRegression treeAggregate stall).
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

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PAIRS_DIR = os.path.join(RESULTS_DIR, "pairs")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "zingg_bestconfig_eval.csv")

DATA_ROOT = "/home/it2022025/er_scalability/datasets"
SPLIT_ROOT = "/home/it2022025/er_scalability/train_validation_test_sets"

# SCADS movie-benchmark column headers (D5 raw imdb/tmdb tables are '|'-delimited
# with these URI headers; the D5 worker reduces them to id/title/name/genre_list).
SCADS_TITLE = "https://www.scads.de/movieBenchmark/ontology/title"
SCADS_NAME = "https://www.scads.de/movieBenchmark/ontology/name"
SCADS_GENRE = "https://www.scads.de/movieBenchmark/ontology/genre_list"

# ---------------------------------------------------------------------------
# Per-dataset configuration. Each block mirrors the matching zingg_ccer_worker /
# zingg_der_worker EXACTLY (paths, delimiters, field definitions, source order,
# training-row schema) so re-running the best config reproduces the worker.
#
#   family "ccer" -> two-source LINK   (train + link)
#   family "der"  -> single-source dedup (train + match)
#
# fields:  ordered [(name, [MatchType names])] passed to setFieldDefinition.
# src1/src2 (ccer): (source_name, path, read_sep, pipe_delim). The worker maps
#   left_id -> src1 and right_id -> src2. For B-cubed each entity is prefixed
#   with its source name; GT column 0 -> src1, column 1 -> src2 (flip with
#   gt_swap=True if the guard reports the GT ids don't line up).
#
# >>> The gt.* paths/seps below are copied from the pyjedai example. The loud
#     check_gt_connects() guard fails the dataset if they don't line up with the
#     entity universe, so a wrong path errors clearly instead of reporting junk.
# ---------------------------------------------------------------------------
CCER_FUZZY = ["FUZZY"]
FT = ["FUZZY", "TEXT"]
DU = ["DONT_USE"]

DATASETS = {
    "D2": dict(
        family="ccer",
        src1=("abt", f"{DATA_ROOT}/D2/abt.csv", "|", "|"),
        src2=("buy", f"{DATA_ROOT}/D2/buy.csv", "|", "|"),
        id_col="id",
        fields=[("id", DU), ("name", FT), ("description", CCER_FUZZY), ("price", DU)],
        test=f"{SPLIT_ROOT}/db2/test_set.csv",
        train=f"{SPLIT_ROOT}/db2/train_set.csv",
        gt=f"{DATA_ROOT}/D2/gt.csv", gt_sep="|", gt_header=0, gt_swap=False,
    ),
    "D3": dict(
        family="ccer",
        src1=("amazon", f"{DATA_ROOT}/D3/amazon.csv", "#", "#"),
        src2=("gp", f"{DATA_ROOT}/D3/gp.csv", "#", "#"),
        id_col="id",
        fields=[("id", DU), ("title", CCER_FUZZY), ("manufacturer", CCER_FUZZY),
                ("description", CCER_FUZZY), ("price", DU)],
        test=f"{SPLIT_ROOT}/db3/test_set.csv",
        train=f"{SPLIT_ROOT}/db3/train_set.csv",
        gt=f"{DATA_ROOT}/D3/gt.csv", gt_sep="#", gt_header=0, gt_swap=False,
    ),
    "D4": dict(
        family="ccer",
        src1=("dblp", f"{DATA_ROOT}/D4/dblp.csv", "%", "%"),
        src2=("acm", f"{DATA_ROOT}/D4/acm.csv", "%", "%"),
        id_col="id",
        fields=[("id", DU), ("title", FT), ("authors", CCER_FUZZY),
                ("venue", CCER_FUZZY), ("year", DU)],
        test=f"{SPLIT_ROOT}/db4/test_set.csv",
        train=f"{SPLIT_ROOT}/db4/train_set.csv",
        gt=f"{DATA_ROOT}/D4/gt.csv", gt_sep="%", gt_header=0, gt_swap=False,
    ),
    "D5": dict(
        family="ccer",
        src1=("imdb", f"{DATA_ROOT}/D5/imdb.csv", "|", ","),
        src2=("tmdb", f"{DATA_ROOT}/D5/tmdb.csv", "|", ","),
        id_col="id",
        fields=[("id", DU), ("title", FT), ("name", CCER_FUZZY), ("genre_list", CCER_FUZZY)],
        clean="scads",  # reduce '|'-URI-headered raw tables to id/title/name/genre_list
        test=f"{SPLIT_ROOT}/db5/test_set.csv",
        train=f"{SPLIT_ROOT}/db5/train_set.csv",
        gt=f"{DATA_ROOT}/D5/gt.csv", gt_sep="|", gt_header=0, gt_swap=False,
    ),
    "D6": dict(
        family="ccer",
        src1=("imdb", f"{SPLIT_ROOT}/db6/tableA.csv", ",", ","),
        src2=("tvdb", f"{SPLIT_ROOT}/db6/tableB.csv", ",", ","),
        id_col="id",
        fields=[("id", DU), ("title", FT), ("name", CCER_FUZZY)],
        test=f"{SPLIT_ROOT}/db6/test_set.csv",
        train=f"{SPLIT_ROOT}/db6/train_set.csv",
        gt=f"{DATA_ROOT}/D6/gt.csv", gt_sep="|", gt_header=0, gt_swap=False,
    ),
    "D7": dict(
        family="ccer",
        src1=("tmdb", f"{SPLIT_ROOT}/db7/tableA.csv", ",", ","),
        src2=("tvdb", f"{SPLIT_ROOT}/db7/tableB.csv", ",", ","),
        id_col="id",
        fields=[("id", DU), ("title", FT), ("name", CCER_FUZZY), ("abstract", CCER_FUZZY)],
        test=f"{SPLIT_ROOT}/db7/test_set.csv",
        train=f"{SPLIT_ROOT}/db7/train_set.csv",
        gt=f"{DATA_ROOT}/D7/gt.csv", gt_sep="|", gt_header=0, gt_swap=False,
    ),
    "D8": dict(
        family="ccer",
        # worker: setData(amazon=tableB, walmart=tableA); left_id -> amazon.
        src1=("amazon", f"{SPLIT_ROOT}/db8/tableB.csv", ",", ","),
        src2=("walmart", f"{SPLIT_ROOT}/db8/tableA.csv", ",", ","),
        id_col="id",
        fields=[("id", DU), ("title", FT), ("modelno", CCER_FUZZY), ("brand", CCER_FUZZY)],
        test=f"{SPLIT_ROOT}/db8/test_set.csv",
        train=f"{SPLIT_ROOT}/db8/train_set.csv",
        # pyjedai treats gt col0=walmart(d1), col1=amazon(d2); Zingg's source order
        # is reversed (src1=amazon), so flip the gt columns to match pyjedai.
        gt=f"{DATA_ROOT}/D8/gt.csv", gt_sep="|", gt_header=0, gt_swap=True,
    ),
    "D9": dict(
        family="ccer",
        src1=("dblp", f"{SPLIT_ROOT}/db9/tableA.csv", ",", ","),
        src2=("scholar", f"{SPLIT_ROOT}/db9/tableB.csv", ",", ","),
        id_col="id",
        fields=[("id", DU), ("title", FT), ("authors", CCER_FUZZY),
                ("venue", CCER_FUZZY), ("year", DU)],
        test=f"{SPLIT_ROOT}/db9/test_set.csv",
        train=f"{SPLIT_ROOT}/db9/train_set.csv",
        gt=f"{DATA_ROOT}/D9/gt.csv", gt_sep=">", gt_header=0, gt_swap=False,
    ),
    "CORA": dict(
        family="der",
        data=(f"{DATA_ROOT}/cora/cora.csv", "|", "|"),  # (path, read_sep, pipe_delim)
        id_col="Entity Id",
        fields=[("Entity Id", DU), ("title", FT), ("author", FT), ("venue", FT),
                ("publisher", DU), ("year", DU)],
        test=f"{SPLIT_ROOT}/cora/test_set.csv",
        train=f"{SPLIT_ROOT}/cora/train_set.csv",
        gt=f"{DATA_ROOT}/cora/cora_gt.csv", gt_sep="|", gt_header=None,
    ),
    "CDDB": dict(
        family="der",
        data=(f"{DATA_ROOT}/CDDB/cddb.csv", ",", ","),
        id_col="id",
        fields=[("id", DU), ("artist", FT), ("title", FT), ("genre", ["EXACT"]),
                ("category", ["EXACT"]), ("year", ["EXACT"])],
        test=f"{SPLIT_ROOT}/cddb/test_set.csv",
        train=f"{SPLIT_ROOT}/cddb/train_set.csv",
        gt=f"{DATA_ROOT}/CDDB/gt.csv", gt_sep=",", gt_header=0,
    ),
}

ALL_DATASETS = list(DATASETS.keys())


# ===========================================================================
# Generic metric helpers (identical semantics to the pyjedai example)
# ===========================================================================
def connected_components(pairs, universe):
    """Union-find over `universe`; `pairs` are (a, b) edges. Returns entity->root."""
    parent = {e: e for e in universe}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    return {e: find(e) for e in universe}


def bcubed(entity_to_pred, entity_to_gt):
    """Standard B-cubed over the shared entity set (both maps cover all entities)."""
    entities = list(entity_to_gt.keys())
    n = len(entities)
    if n == 0:
        return 0.0, 0.0, 0.0
    pred_size = Counter(entity_to_pred.values())
    gt_size = Counter(entity_to_gt.values())
    joint = Counter((entity_to_pred[e], entity_to_gt[e]) for e in entities)
    p_sum = r_sum = 0.0
    for e in entities:
        p, g = entity_to_pred[e], entity_to_gt[e]
        correct = joint[(p, g)]
        p_sum += correct / pred_size[p]
        r_sum += correct / gt_size[g]
    P, R = p_sum / n, r_sum / n
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return P, R, F


def pairwise_prf(y_true, y_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score
    return (precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0))


def check_gt_connects(ds, n_gt_total, n_gt_connected):
    """Guard: if ground-truth pairs don't reference known entities, B-cubed is
    meaningless (recall collapses to 1.0). Fail loudly instead of reporting junk."""
    if n_gt_total == 0:
        raise RuntimeError(f"{ds}: ground-truth file loaded 0 pairs -- check gt path/format.")
    frac = n_gt_connected / n_gt_total
    if frac < 0.5:
        raise RuntimeError(
            f"{ds}: only {n_gt_connected}/{n_gt_total} GT pairs match dataset ids "
            f"({frac:.1%}). GT ids don't line up with the entity universe -- check "
            f"gt_sep/gt_header/column order (try gt_swap=True). Refusing to report "
            f"bogus B-cubed.")
    sys.stderr.write(f"[{ds}] GT: {n_gt_connected}/{n_gt_total} pairs connect "
                     f"({frac:.1%}).\n")


def load_gt_pairs(path, sep, header):
    """Load ground-truth match pairs as a list of (left, right) string tuples.

    Separator and header are given EXPLICITLY per dataset (auto-detection is
    unreliable on '|'-delimited files). The first two columns are the id pair.
    """
    import pandas as pd
    df = pd.read_csv(path, sep=sep, header=header, engine="python", dtype=str)
    df = df.fillna("")
    if df.shape[1] < 2:
        raise RuntimeError(
            f"GT file {path} parsed into {df.shape[1]} column with sep={sep!r} "
            f"(first row: {df.iloc[0,0]!r}). Wrong delimiter -- fix gt_sep for this dataset.")
    left_col, right_col = df.columns[0], df.columns[1]
    return [(str(a), str(b)) for a, b in zip(df[left_col], df[right_col])]


# ===========================================================================
# Shared Zingg helpers
# ===========================================================================
def field_defs(fields):
    from zingg.client import FieldDefinition, MatchType
    out = []
    for name, mts in fields:
        out.append(FieldDefinition(name, "string", *[getattr(MatchType, m) for m in mts]))
    return out


def sample_train(train_df, neg_ratio, seed):
    """Reproduce the workers' TRAIN-only negative down-sampling."""
    import pandas as pd
    matches = train_df[train_df["label"] == 1]
    nonmatches = train_df[train_df["label"] == 0]
    n_neg = int(min(len(nonmatches), round(len(matches) * float(neg_ratio))))
    nonmatches = (nonmatches.sample(n=n_neg, random_state=seed)
                  if n_neg < len(nonmatches) else nonmatches)
    return pd.concat([matches, nonmatches], ignore_index=True)


def read_source(path, read_sep, id_col, clean=None):
    import pandas as pd
    df = pd.read_csv(path, sep=read_sep, engine="python", na_filter=False)
    if clean == "scads":
        df = df[["id", SCADS_TITLE, SCADS_NAME, SCADS_GENRE]].rename(columns={
            SCADS_TITLE: "title", SCADS_NAME: "name", SCADS_GENRE: "genre_list"})
    return df


# ===========================================================================
# CCER (clean-clean, two-source LINK) evaluation of one dataset
# ===========================================================================
def eval_ccer(ds, cfg, params, seed, zingg_dir):
    import pandas as pd
    from pyspark.sql import SparkSession
    from zingg.client import Arguments, ClientOptions, ZinggWithSpark
    from zingg.pipes import CsvPipe, Pipe

    (s1name, s1path, s1sep, s1delim) = cfg["src1"]
    (s2name, s2path, s2sep, s2delim) = cfg["src2"]
    id_col = cfg["id_col"]
    attrs = [name for name, _ in cfg["fields"] if name != id_col]

    training_parquet = f"{zingg_dir}/training_data"
    output_dir = f"{zingg_dir}/output"
    model_id = f"{ds.lower()}_best"

    df1 = read_source(s1path, s1sep, id_col, cfg.get("clean"))
    df2 = read_source(s2path, s2sep, id_col, cfg.get("clean"))
    idx1 = df1.set_index(id_col)
    idx2 = df2.set_index(id_col)
    train_df = pd.read_csv(cfg["train"])
    test_df = pd.read_csv(cfg["test"])

    # ---- D5 needs cleaned, comma-delimited copies for Zingg's own CsvPipe ----
    if cfg.get("clean") == "scads":
        s1path = f"{zingg_dir}/{s1name}_clean.csv"
        s2path = f"{zingg_dir}/{s2name}_clean.csv"
        df1.to_csv(s1path, index=False)
        df2.to_csv(s2path, index=False)

    # ---- build Zingg training data (labelled pairs) from TRAIN only ----
    train_sample = sample_train(train_df, params["neg_ratio"], seed)
    rows = []
    cluster_id = 0
    for _, row in train_sample.iterrows():
        lid, rid, label = int(row["left_id"]), int(row["right_id"]), int(row["label"])
        try:
            left, right = idx1.loc[lid], idx2.loc[rid]
        except KeyError:
            continue
        r1 = {id_col: str(lid), "z_cluster": cluster_id, "z_isMatch": label, "z_zsource": s1name}
        r2 = {id_col: str(rid), "z_cluster": cluster_id, "z_isMatch": label, "z_zsource": s2name}
        for a in attrs:
            r1[a] = str(left[a])
            r2[a] = str(right[a])
        rows.append(r1)
        rows.append(r2)
        cluster_id += 1
    pd.DataFrame(rows).to_parquet(training_parquet, index=False, engine="pyarrow")

    spark = SparkSession.builder.appName(f"Zingg{ds}best").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    args = Arguments()
    args.setFieldDefinition(field_defs(cfg["fields"]))
    args.setModelId(model_id)
    args.setZinggDir(zingg_dir)
    args.setNumPartitions(int(params["numPartitions"]))
    args.setLabelDataSampleSize(float(params["labelDataSampleSize"]))

    p1 = CsvPipe(s1name, s1path); p1.addProperty("header", "true"); p1.addProperty("delimiter", s1delim)
    p2 = CsvPipe(s2name, s2path); p2.addProperty("header", "true"); p2.addProperty("delimiter", s2delim)
    args.setData(p1, p2)
    out_pipe = CsvPipe("output", output_dir); out_pipe.addProperty("header", "true"); out_pipe.addProperty("delimiter", "|")
    args.setOutput(out_pipe)
    tr_pipe = Pipe("training", "parquet"); tr_pipe.addProperty("location", training_parquet)
    args.setTrainingSamples(tr_pipe)

    ZinggWithSpark(args, ClientOptions([ClientOptions.PHASE, "train"])).initAndExecute()
    ZinggWithSpark(args, ClientOptions([ClientOptions.PHASE, "link"])).initAndExecute()

    output_df = (spark.read.option("header", "true").option("delimiter", "|")
                 .csv(output_dir + "/*.csv").toPandas())
    spark.stop()

    t = float(params["chosen_threshold"])

    # ---- within each output z_cluster link src1 ids to src2 ids (worker logic).
    #      score_map -> pairwise cross-check; tagged edges (score>=t) -> B-cubed ----
    score_map = {}          # (native id, native id) -> cluster max score, both directions
    pred_pairs_native = []  # (src1 id, src2 id) kept at threshold t, for the dump
    pred_edges = []         # ("src1name:id", "src2name:id") kept at threshold t, for B-cubed
    for _, group in output_df.groupby("z_cluster"):
        sources = group["z_zsource"].values
        ids = group["id"].values
        scores = group["z_score"].astype(float).values
        s1ids = [ids[i] for i, s in enumerate(sources) if s == s1name]
        s2ids = [ids[i] for i, s in enumerate(sources) if s == s2name]
        score = float(scores.max()) if len(scores) else 0.0
        keep = score >= t
        for a in s1ids:
            for b in s2ids:
                score_map[(str(a), str(b))] = score
                score_map[(str(b), str(a))] = score
                if keep:
                    pred_pairs_native.append((str(a), str(b)))
                    pred_edges.append((f"{s1name}:{a}", f"{s2name}:{b}"))

    # ---- PAIRWISE cross-check on the test split (reproduces the worker) ----
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        key = (str(int(row["left_id"])), str(int(row["right_id"])))
        y_true.append(int(row["label"]))
        y_pred.append(1 if score_map.get(key, 0.0) >= t else 0)
    pw_p, pw_r, pw_f = pairwise_prf(y_true, y_pred)

    # ---- CLUSTER-LEVEL B-cubed over the full entity universe ----
    universe = ([f"{s1name}:{x}" for x in df1[id_col].tolist()]
                + [f"{s2name}:{x}" for x in df2[id_col].tolist()])
    universe_set = set(universe)

    gt_raw = load_gt_pairs(cfg["gt"], cfg["gt_sep"], cfg["gt_header"])
    if cfg.get("gt_swap"):
        gt_tagged = [(f"{s2name}:{a}", f"{s1name}:{b}") for a, b in gt_raw]
    else:
        gt_tagged = [(f"{s1name}:{a}", f"{s2name}:{b}") for a, b in gt_raw]
    n_gt_conn = sum(1 for a, b in gt_tagged if a in universe_set and b in universe_set)
    check_gt_connects(ds, len(gt_tagged), n_gt_conn)

    entity_to_pred = connected_components(pred_edges, universe)
    entity_to_gt = connected_components(gt_tagged, universe)
    b_p, b_r, b_f = bcubed(entity_to_pred, entity_to_gt)

    dump_pairs = sorted(set(pred_pairs_native))
    return dict(pairwise=(pw_p, pw_r, pw_f), bcubed=(b_p, b_r, b_f),
                n_entities=len(universe), n_pred_pairs=len(dump_pairs),
                n_gt_pairs=len(gt_tagged),
                dump_pairs=dump_pairs, entities=universe)


# ===========================================================================
# DER (dirty ER, single-source MATCH) evaluation of one dataset
# ===========================================================================
def eval_der(ds, cfg, params, seed, zingg_dir):
    import pandas as pd
    from pyspark.sql import SparkSession
    from zingg.client import Arguments, ClientOptions, ZinggWithSpark
    from zingg.pipes import CsvPipe, Pipe

    (path, read_sep, pipe_delim) = cfg["data"]
    id_col = cfg["id_col"]
    attrs = [name for name, _ in cfg["fields"] if name != id_col]

    training_parquet = f"{zingg_dir}/training_data"
    output_dir = f"{zingg_dir}/output"
    model_id = f"{ds.lower()}_best"

    df = pd.read_csv(path, sep=read_sep, engine="python", na_filter=False)
    data_idx = df.set_index(id_col)
    train_df = pd.read_csv(cfg["train"])
    test_df = pd.read_csv(cfg["test"])

    # ---- build Zingg training data (labelled pairs) from TRAIN only ----
    train_sample = sample_train(train_df, params["neg_ratio"], seed)
    rows = []
    cluster_id = 0
    for _, row in train_sample.iterrows():
        lid, rid, label = int(row["left_id"]), int(row["right_id"]), int(row["label"])
        try:
            left, right = data_idx.loc[lid], data_idx.loc[rid]
        except KeyError:
            continue
        r1 = {id_col: str(lid), "z_cluster": cluster_id, "z_isMatch": label, "z_zsource": "left"}
        r2 = {id_col: str(rid), "z_cluster": cluster_id, "z_isMatch": label, "z_zsource": "right"}
        for a in attrs:
            r1[a] = str(left.get(a, ""))
            r2[a] = str(right.get(a, ""))
        rows.append(r1)
        rows.append(r2)
        cluster_id += 1
    pd.DataFrame(rows).to_parquet(training_parquet, index=False, engine="pyarrow")

    spark = SparkSession.builder.appName(f"Zingg{ds}best").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir(zingg_dir)

    args = Arguments()
    args.setFieldDefinition(field_defs(cfg["fields"]))
    args.setModelId(model_id)
    args.setZinggDir(zingg_dir)
    args.setNumPartitions(int(params["numPartitions"]))
    args.setLabelDataSampleSize(float(params["labelDataSampleSize"]))

    data_pipe = CsvPipe(ds.lower(), path)
    data_pipe.addProperty("header", "true"); data_pipe.addProperty("delimiter", pipe_delim)
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

    out_id_col = None
    for c in [id_col, "Entity id", "id"]:
        if c in output_df.columns:
            out_id_col = c
            break
    if out_id_col is None:
        raise KeyError(f"No entity id column. Columns: {list(output_df.columns)}")
    score_col = "z_maxScore" if "z_maxScore" in output_df.columns else "z_score"
    if score_col not in output_df.columns:
        raise KeyError(f"No score column. Columns: {list(output_df.columns)}")

    output_df[out_id_col] = output_df[out_id_col].astype(str)
    output_df["z_cluster"] = output_df["z_cluster"].astype(str)
    output_df[score_col] = pd.to_numeric(output_df[score_col], errors="coerce").fillna(0.0)
    entity_scores = output_df.set_index(out_id_col)[score_col].to_dict()
    entity_cluster = output_df.set_index(out_id_col)["z_cluster"].to_dict()

    t = float(params["chosen_threshold"])

    # ---- PAIRWISE cross-check on the test split (reproduces the worker:
    #      same cluster AND min(entity scores) >= t) ----
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        lid, rid = str(int(row["left_id"])), str(int(row["right_id"]))
        lc, rc = entity_cluster.get(lid), entity_cluster.get(rid)
        if lc is not None and lc == rc:
            score = min(float(entity_scores.get(lid, 0.0)), float(entity_scores.get(rid, 0.0)))
        else:
            score = 0.0
        y_true.append(int(row["label"]))
        y_pred.append(1 if score >= t else 0)
    pw_p, pw_r, pw_f = pairwise_prf(y_true, y_pred)

    # ---- CLUSTER-LEVEL B-cubed over the full entity universe.
    #      Within a z_cluster, entities scoring >= t are mutually predicted
    #      duplicates (every intra pair then has min score >= t) ----
    universe = [str(x) for x in df[id_col].tolist()]
    universe_set = set(universe)

    cluster_members = {}
    for eid, cl in entity_cluster.items():
        cluster_members.setdefault(cl, []).append(eid)
    pred_edges = []
    for cl, members in cluster_members.items():
        keep = [m for m in members if float(entity_scores.get(m, 0.0)) >= t]
        for i, j in combinations(sorted(keep), 2):
            pred_edges.append((i, j))

    gt_pairs = load_gt_pairs(cfg["gt"], cfg["gt_sep"], cfg["gt_header"])
    n_gt_conn = sum(1 for a, b in gt_pairs if a in universe_set and b in universe_set)
    check_gt_connects(ds, len(gt_pairs), n_gt_conn)

    entity_to_pred = connected_components(pred_edges, universe)
    entity_to_gt = connected_components(gt_pairs, universe)
    b_p, b_r, b_f = bcubed(entity_to_pred, entity_to_gt)

    dump_pairs = sorted(set(pred_edges))
    return dict(pairwise=(pw_p, pw_r, pw_f), bcubed=(b_p, b_r, b_f),
                n_entities=len(universe), n_pred_pairs=len(dump_pairs),
                n_gt_pairs=len(gt_pairs),
                dump_pairs=dump_pairs, entities=universe)


# ===========================================================================
# Best-config lookup
# ===========================================================================
def read_best_config(ds):
    """Return (params_dict, config_id, csv_test_f1) for the max-test_f1 OK row."""
    path = os.path.join(RESULTS_DIR, f"zingg_{ds}_configs.csv")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("status") == "OK" and r.get("test_f1") not in (None, "")]
    if not rows:
        raise RuntimeError(f"{ds}: no OK rows in {path}")
    best = max(rows, key=lambda r: float(r["test_f1"]))
    params = dict(
        numPartitions=int(float(best["numPartitions"])),
        labelDataSampleSize=float(best["labelDataSampleSize"]),
        neg_ratio=float(best["neg_ratio"]),
        chosen_threshold=float(best["chosen_threshold"]),
    )
    return params, best["config_id"], float(best["test_f1"])


# ===========================================================================
# Single-dataset driver (subprocess entry point)
# ===========================================================================
def run_single(ds):
    cfg = DATASETS[ds]
    params, config_id, csv_test_f1 = read_best_config(ds)
    seed = 42
    random.seed(seed); np.random.seed(seed)

    zingg_dir = f"/tmp/zingg_eval_{ds.lower()}"
    if os.path.exists(zingg_dir):
        shutil.rmtree(zingg_dir, ignore_errors=True)
    os.makedirs(zingg_dir, exist_ok=True)

    t0 = time.time()
    try:
        if cfg["family"] == "ccer":
            out = eval_ccer(ds, cfg, params, seed, zingg_dir)
        else:
            out = eval_der(ds, cfg, params, seed, zingg_dir)
    finally:
        shutil.rmtree(zingg_dir, ignore_errors=True)

    os.makedirs(PAIRS_DIR, exist_ok=True)
    with open(os.path.join(PAIRS_DIR, f"zingg_{ds}_pred_pairs.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["left_id", "right_id"])
        wtr.writerows(out["dump_pairs"])
    with open(os.path.join(PAIRS_DIR, f"zingg_{ds}_entities.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["entity_id"])
        wtr.writerows([[e] for e in out["entities"]])

    pw, bc = out["pairwise"], out["bcubed"]
    result = {
        "dataset": ds, "config_id": config_id, "family": cfg["family"],
        "csv_test_f1": round(csv_test_f1, 6),
        "pairwise_precision": round(pw[0], 6), "pairwise_recall": round(pw[1], 6),
        "pairwise_f1": round(pw[2], 6),
        "bcubed_precision": round(bc[0], 6), "bcubed_recall": round(bc[1], 6),
        "bcubed_f1": round(bc[2], 6),
        "n_entities": out["n_entities"], "n_pred_pairs": out["n_pred_pairs"],
        "n_gt_pairs": out["n_gt_pairs"], "time_sec": round(time.time() - t0, 2),
    }
    print("RESULT_JSON:" + json.dumps(result))
    return result


# ===========================================================================
# All-datasets driver (one subprocess per dataset -> fresh Spark/JVM each time)
# ===========================================================================
SUMMARY_COLS = ["dataset", "config_id", "family", "csv_test_f1",
                "pairwise_precision", "pairwise_recall", "pairwise_f1",
                "bcubed_precision", "bcubed_recall", "bcubed_f1",
                "n_entities", "n_pred_pairs", "n_gt_pairs", "time_sec", "status"]


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for ds in ALL_DATASETS:
        print(f"\n=== {ds} ===", flush=True)
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), ds],
                              capture_output=True, text=True)
        result = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                result = json.loads(line[len("RESULT_JSON:"):])
        if result is None:
            print(f"  -> ERROR\n{proc.stderr[-1200:]}", flush=True)
            rows.append({"dataset": ds, "status": "ERROR"})
        else:
            result["status"] = "OK"
            print(f"  -> pairwise F1={result['pairwise_f1']:.4f} "
                  f"(csv {result['csv_test_f1']:.4f})  "
                  f"B3 F1={result['bcubed_f1']:.4f} "
                  f"(P={result['bcubed_precision']:.3f} R={result['bcubed_recall']:.3f})  "
                  f"cfg#{result['config_id']}  {result['time_sec']}s", flush=True)
            rows.append(result)
        with open(SUMMARY_CSV, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
            wtr.writeheader()
            for r in rows:
                wtr.writerow({c: r.get(c) for c in SUMMARY_COLS})
    print(f"\nDone. Wrote {SUMMARY_CSV}")


def main():
    if len(sys.argv) > 1:
        ds = sys.argv[1]
        if ds not in DATASETS:
            sys.exit(f"Unknown dataset '{ds}'. Choose from: {', '.join(ALL_DATASETS)}")
        run_single(ds)
    else:
        run_all()


if __name__ == "__main__":
    main()
