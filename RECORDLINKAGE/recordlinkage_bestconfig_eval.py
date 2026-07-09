#!/usr/bin/env python3
"""
Best-config evaluation at TWO levels: pairwise + cluster-level (B-cubed),
for the RecordLinkage CCER experiments -- the same idea as your pyjedai
`pyjedai_bestconfig_eval.py`, done the RecordLinkage way.

For each dataset it:
  1. reads the BEST config (max test_f1 among status==OK rows) straight from the
     existing results/recordlinkage_<DS>_configs.csv,
  2. rebuilds EXACTLY that dataset's worker feature pipeline (same clean/compare/
     TF-IDF/numeric features) and retrains the best classifier on the train split,
  3. reports PAIRWISE P/R/F1 on the test split the same way the workers do
     (classifier probs on the test pairs, thresholded at the stored
     chosen_threshold), as a sanity cross-check against your existing numbers,
  4. reports CLUSTER-LEVEL B-cubed P/R/F1, comparing the predicted clustering
     against the full ground-truth clustering over EVERY entity,
  5. dumps the predicted match pairs + the entity id list under results/pairs/.

Predicted clustering (the RecordLinkage way)
-------------------------------------------
The workers have no separate blocking stage: the train+valid+test splits together
ARE the blocked candidate set (blocking produced these pairs, then they were
partitioned). So the predicted clustering is union-find over every candidate pair
the best classifier scores >= chosen_threshold; the ground-truth clustering comes
from gt.csv (like pyjedai). Both run over the FULL A/B entity universe (every id
in both sources), so unmatched entities are singletons -- exactly as B-cubed
expects. Caveat: the candidate set includes the train pairs the classifier was fit
on, so cluster precision reads a touch in-sample-optimistic; the pairwise
test_point stays the clean held-out number.

Usage
-----
  python3 recordlinkage_bestconfig_eval.py D2   # one dataset -> RESULT_JSON + dumps
  python3 recordlinkage_bestconfig_eval.py       # all datasets -> summary CSV

Output
------
  results/recordlinkage_bestconfig_eval.csv          one row per dataset, both levels
  results/pairs/recordlinkage_<DS>_pred_pairs.csv    predicted matches (left_id,right_id)
  results/pairs/recordlinkage_<DS>_entities.csv       full entity id universe (one col)

>>> VERIFY the "gt"/"gt_sep" paths below. <<<  The CCER workers never loaded a
ground-truth file (they only scored the labeled test split), so B-cubed needs it.
The paths follow the datasets/D<i>/gt.csv layout used by the pyjedai eval; the
connectivity guard fails loudly if the ids don't line up.
"""
import os
import sys
import csv
import json
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import recordlinkage
from recordlinkage.preprocessing import clean
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PAIRS_DIR = os.path.join(RESULTS_DIR, "pairs")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "recordlinkage_bestconfig_eval.csv")

DATA_ROOT = "/home/it2022025/er_scalability/datasets"
SPLIT_ROOT = "/home/it2022025/er_scalability/train_validation_test_sets"

SCADS_RENAME = {
    "https://www.scads.de/movieBenchmark/ontology/title": "title",
    "https://www.scads.de/movieBenchmark/ontology/name": "name",
    "https://www.scads.de/movieBenchmark/ontology/genre_list": "genre_list",
    "http://dbpedia.org/ontology/abstract": "abstract",
    "http://dbpedia.org/ontology/episodeNumber": "episodeNumber",
    "http://dbpedia.org/ontology/seasonNumber": "seasonNumber",
    "http://dbpedia.org/ontology/releaseDate": "releaseDate",
}


# ===========================================================================
# Shared feature helpers (verbatim from the workers)
# ===========================================================================
def rowwise_cosine(left_matrix, right_matrix):
    dot = np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).ravel()
    ln = np.sqrt(np.asarray(left_matrix.multiply(left_matrix).sum(axis=1)).ravel())
    rn = np.sqrt(np.asarray(right_matrix.multiply(right_matrix).sum(axis=1)).ravel())
    denom = ln * rn
    denom[denom == 0] = 1.0
    return dot / denom


def get_texts(df, ids, cols):
    return (df.loc[ids, cols].fillna("").astype(str).apply(" ".join, axis=1).to_numpy())


def build_tfidf_sim(pairs, left_df, right_df, vec, cols, label):
    lv = vec.transform(get_texts(left_df, pairs.get_level_values(0), cols))
    rv = vec.transform(get_texts(right_df, pairs.get_level_values(1), cols))
    return pd.Series(rowwise_cosine(lv, rv), index=pairs, name=label)


def add_numeric_pair_features(pair_index, left_df, right_df, feat_df, cols):
    """Generalised abs/rel-diff + both-known features for the given numeric cols.
    cols is a list of (source_col, feature_prefix). Reproduces add_price_features
    (D2/D3, price) and add_numeric_features (D8, price+shipweight) exactly."""
    left_ids = pair_index.get_level_values(0)
    right_ids = pair_index.get_level_values(1)
    for col, prefix in cols:
        lv = left_df.loc[left_ids, col].to_numpy(dtype=float)
        rv = right_df.loc[right_ids, col].to_numpy(dtype=float)
        both = ~(np.isnan(lv) | np.isnan(rv))
        lvf = np.where(np.isnan(lv), 0.0, lv)
        rvf = np.where(np.isnan(rv), 0.0, rv)
        ad = np.abs(lvf - rvf)
        mx = np.maximum(np.abs(lvf), np.abs(rvf)); mx[mx == 0] = 1.0
        feat_df[f"{prefix}_abs_diff"] = pd.array(ad, dtype="Float64")
        feat_df[f"{prefix}_rel_diff"] = pd.array(ad / mx, dtype="Float64")
        feat_df[f"{prefix}_both_known"] = pd.array(both.astype(float), dtype="Float64")


def _derive_release_year(df):
    df["releaseYear"] = pd.to_datetime(df["releaseDate"], errors="coerce").dt.year
    return df


# ===========================================================================
# Per-dataset pipeline spec (mirrors each worker faithfully)
#   clean   : list of (col, fillna?)      -> clean(df[col]) / clean(df[col].fillna(""))
#   numeric : list of cols                -> pd.to_numeric(errors="coerce")
#   derive  : callable(df)->df or None    -> extra derived cols (D7 releaseYear)
#   compare : list of ("string",col,method,label) / ("exact",col,label)
#   word_cols/char_cols : TF-IDF corpora + similarity columns
#   numeric_feat : list of (col, prefix) for abs/rel/both-known, or None
# ===========================================================================
DATASETS = {
    "D2": dict(
        left=f"{DATA_ROOT}/D2/abt.csv", right=f"{DATA_ROOT}/D2/buy.csv", delim="|",
        split="db2", rename=None,
        clean=[("name", False), ("description", True)], numeric=["price"], derive=None,
        compare=[("string", "name", "cosine", "name_cosine"),
                 ("string", "name", "jarowinkler", "name_jw"),
                 ("string", "description", "cosine", "description_cosine")],
        word_cols=["name", "description"], char_cols=["name"],
        numeric_feat=[("price", "price")],
        gt=f"{DATA_ROOT}/D2/gt.csv", gt_sep="|", gt_header=0),
    "D3": dict(
        left=f"{DATA_ROOT}/D3/amazon.csv", right=f"{DATA_ROOT}/D3/gp.csv", delim="#",
        split="db3", rename=None,
        clean=[("title", False), ("description", True), ("manufacturer", True)],
        numeric=["price"], derive=None,
        compare=[("string", "title", "cosine", "title_cosine"),
                 ("string", "title", "jarowinkler", "title_jw"),
                 ("string", "description", "cosine", "description_cosine"),
                 ("string", "manufacturer", "cosine", "manufacturer_cosine")],
        word_cols=["title", "description"], char_cols=["title"],
        numeric_feat=[("price", "price")],
        gt=f"{DATA_ROOT}/D3/gt.csv", gt_sep="#", gt_header=0),
    "D4": dict(
        left=f"{DATA_ROOT}/D4/dblp.csv", right=f"{DATA_ROOT}/D4/acm.csv", delim="%",
        split="db4", rename=None,
        clean=[("title", False), ("authors", True), ("venue", True)],
        numeric=["year"], derive=None,
        compare=[("string", "title", "cosine", "title_cosine"),
                 ("string", "title", "jarowinkler", "title_jw"),
                 ("string", "authors", "cosine", "authors_cosine"),
                 ("string", "venue", "cosine", "venue_cosine"),
                 ("exact", "year", "year_exact")],
        word_cols=["title", "authors"], char_cols=["title"],
        numeric_feat=None,
        gt=f"{DATA_ROOT}/D4/gt.csv", gt_sep="%", gt_header=0),
    "D5": dict(
        left=f"{DATA_ROOT}/D5/imdb.csv", right=f"{DATA_ROOT}/D5/tmdb.csv", delim="|",
        split="db5", rename=SCADS_RENAME,
        clean=[("title", True), ("name", True), ("genre_list", True)],
        numeric=["episodeNumber", "seasonNumber"], derive=None,
        compare=[("string", "title", "cosine", "title_cosine"),
                 ("string", "title", "jarowinkler", "title_jw"),
                 ("string", "name", "cosine", "name_cosine"),
                 ("string", "name", "jarowinkler", "name_jw"),
                 ("string", "genre_list", "cosine", "genre_cosine"),
                 ("exact", "episodeNumber", "episode_exact"),
                 ("exact", "seasonNumber", "season_exact")],
        word_cols=["title", "name"], char_cols=["title", "name"],
        numeric_feat=None,
        gt=f"{DATA_ROOT}/D5/gt.csv", gt_sep="|", gt_header=0),
    "D6": dict(
        left=f"{DATA_ROOT}/D6/imdb.csv", right=f"{DATA_ROOT}/D6/tvdb.csv", delim="|",
        split="db6", rename=SCADS_RENAME,
        clean=[("title", True), ("name", True)],
        numeric=["episodeNumber", "seasonNumber"], derive=None,
        compare=[("string", "title", "cosine", "title_cosine"),
                 ("string", "title", "jarowinkler", "title_jw"),
                 ("string", "name", "cosine", "name_cosine"),
                 ("string", "name", "jarowinkler", "name_jw"),
                 ("exact", "episodeNumber", "episode_exact"),
                 ("exact", "seasonNumber", "season_exact")],
        word_cols=["title", "name"], char_cols=["title", "name"],
        numeric_feat=None,
        gt=f"{DATA_ROOT}/D6/gt.csv", gt_sep="|", gt_header=0),
    "D7": dict(
        left=f"{DATA_ROOT}/D7/tmdb.csv", right=f"{DATA_ROOT}/D7/tvdb.csv", delim="|",
        split="db7", rename=SCADS_RENAME,
        clean=[("title", True), ("name", True), ("abstract", True)],
        numeric=["episodeNumber", "seasonNumber"], derive=_derive_release_year,
        compare=[("string", "title", "cosine", "title_cosine"),
                 ("string", "title", "jarowinkler", "title_jw"),
                 ("string", "name", "cosine", "name_cosine"),
                 ("string", "name", "jarowinkler", "name_jw"),
                 ("string", "abstract", "cosine", "abstract_cosine"),
                 ("exact", "episodeNumber", "episode_exact"),
                 ("exact", "seasonNumber", "season_exact"),
                 ("exact", "releaseYear", "year_exact")],
        word_cols=["title", "name", "abstract"], char_cols=["title", "name"],
        numeric_feat=None,
        gt=f"{DATA_ROOT}/D7/gt.csv", gt_sep="|", gt_header=0),
    "D8": dict(
        left=f"{DATA_ROOT}/D8/walmart.csv", right=f"{DATA_ROOT}/D8/amazon.csv", delim="|",
        split="db8", rename=None,
        clean=[("title", True), ("brand", True), ("modelno", True), ("dimensions", True)],
        numeric=["price", "shipweight"], derive=None,
        compare=[("string", "title", "cosine", "title_cosine"),
                 ("string", "title", "jarowinkler", "title_jw"),
                 ("string", "brand", "cosine", "brand_cosine"),
                 ("string", "brand", "jarowinkler", "brand_jw"),
                 ("string", "modelno", "jarowinkler", "modelno_jw")],
        word_cols=["title", "brand", "dimensions"], char_cols=["title", "modelno"],
        numeric_feat=[("price", "price"), ("shipweight", "weight")],
        gt=f"{DATA_ROOT}/D8/gt.csv", gt_sep="|", gt_header=0),
    "D9": dict(
        left=f"{DATA_ROOT}/D9/dblp.csv", right=f"{DATA_ROOT}/D9/scholar.csv", delim=">",
        split="db9", rename=None,
        clean=[("title", True), ("authors", True), ("venue", True)],
        numeric=["year"], derive=None,
        compare=[("string", "title", "cosine", "title_cosine"),
                 ("string", "title", "jarowinkler", "title_jw"),
                 ("string", "authors", "cosine", "authors_cosine"),
                 ("string", "venue", "cosine", "venue_cosine"),
                 ("exact", "year", "year_exact")],
        word_cols=["title", "authors"], char_cols=["title"],
        numeric_feat=None,
        gt=f"{DATA_ROOT}/D9/gt.csv", gt_sep=">", gt_header=0),
}

ALL_DATASETS = list(DATASETS.keys())


# ===========================================================================
# Metric helpers (verbatim from the pyjedai eval)
# ===========================================================================
def connected_components(pairs, universe):
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


def check_gt_connects(ds, n_gt_total, n_gt_connected):
    if n_gt_total == 0:
        raise RuntimeError(f"{ds}: ground-truth file loaded 0 pairs -- check gt path/format.")
    frac = n_gt_connected / n_gt_total
    if frac < 0.5:
        raise RuntimeError(
            f"{ds}: only {n_gt_connected}/{n_gt_total} GT pairs match dataset ids "
            f"({frac:.1%}). GT ids don't line up with the entity universe -- check "
            f"gt_sep/gt_header/column order. Refusing to report bogus B-cubed.")
    sys.stderr.write(f"[{ds}] GT: {n_gt_connected}/{n_gt_total} pairs connect "
                     f"({frac:.1%}).\n")


def load_gt_pairs(path, sep, header):
    df = pd.read_csv(path, sep=sep, header=header, engine="python", dtype=str)
    df = df.fillna("")
    if df.shape[1] < 2:
        raise RuntimeError(
            f"GT file {path} parsed into {df.shape[1]} column with sep={sep!r} "
            f"(first row: {df.iloc[0,0]!r}). Wrong delimiter -- fix gt_sep for this dataset.")
    left_col, right_col = df.columns[0], df.columns[1]
    return [(str(a), str(b)) for a, b in zip(df[left_col], df[right_col])]


# ===========================================================================
# Classifier reconstruction (verbatim from the workers)
# ===========================================================================
def make_classifier(cfg, neg, pos):
    m = cfg["matcher"]
    cw = None if cfg.get("class_weight") == "none" else "balanced"
    if m == "LogisticRegression":
        return LogisticRegression(C=float(cfg["C"]), max_iter=1000, class_weight=cw)
    if m == "RandomForest":
        return RandomForestClassifier(n_estimators=int(cfg["n_estimators"]),
                                      max_depth=int(cfg["max_depth"]),
                                      class_weight=cw, random_state=42, n_jobs=-1)
    if m == "GradientBoosting":
        return GradientBoostingClassifier(n_estimators=int(cfg["n_estimators"]),
                                          max_depth=int(cfg["max_depth"]),
                                          learning_rate=float(cfg["learning_rate"]),
                                          random_state=42)
    raise ValueError(f"unknown matcher {m}")


def read_best_config(ds):
    """Return (params, config_id, csv_test_f1) for the max-test_f1 OK row."""
    path = os.path.join(RESULTS_DIR, f"recordlinkage_{ds}_configs.csv")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("status") == "OK" and r.get("test_f1") not in (None, "")]
    if not rows:
        raise RuntimeError(f"{ds}: no OK rows in {path}")
    best = max(rows, key=lambda r: float(r["test_f1"]))
    params = {"matcher": best["matcher"], "class_weight": best.get("class_weight") or "none"}
    for k in ("n_estimators", "max_depth", "learning_rate", "C"):
        v = best.get(k)
        if v not in (None, ""):
            params[k] = v
    return params, best["config_id"], float(best["test_f1"]), float(best["chosen_threshold"])


# ===========================================================================
# Feature building (one spec-driven builder for every dataset)
# ===========================================================================
def build_features(cfg):
    left = pd.read_csv(cfg["left"], delimiter=cfg["delim"])
    right = pd.read_csv(cfg["right"], delimiter=cfg["delim"])
    if cfg["rename"]:
        left = left.rename(columns=cfg["rename"])
        right = right.rename(columns=cfg["rename"])
    for df in (left, right):
        for col, do_fill in cfg["clean"]:
            df[col] = clean(df[col].fillna("")) if do_fill else clean(df[col])
        for col in cfg["numeric"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if cfg["derive"] is not None:
            cfg["derive"](df)
    left.set_index("id", inplace=True)
    right.set_index("id", inplace=True)

    split = cfg["split"]
    train_df = pd.read_csv(f"{SPLIT_ROOT}/{split}/train_set.csv")
    valid_df = pd.read_csv(f"{SPLIT_ROOT}/{split}/valid_set.csv")
    test_df = pd.read_csv(f"{SPLIT_ROOT}/{split}/test_set.csv")
    y = {k: d["label"].values for k, d in
         (("train", train_df), ("valid", valid_df), ("test", test_df))}
    idx = {k: pd.MultiIndex.from_arrays([d["left_id"], d["right_id"]]) for k, d in
           (("train", train_df), ("valid", valid_df), ("test", test_df))}

    comp = recordlinkage.Compare()
    for spec in cfg["compare"]:
        if spec[0] == "string":
            _, col, method, label = spec
            comp.string(col, col, method=method, label=label)
        else:
            _, col, label = spec
            comp.exact(col, col, label=label)
    feats = {k: comp.compute(idx[k], left, right) for k in idx}

    tl = pd.Index(train_df["left_id"].unique())
    tr = pd.Index(train_df["right_id"].unique())
    word_corpus = np.concatenate([get_texts(left, tl, cfg["word_cols"]),
                                  get_texts(right, tr, cfg["word_cols"])])
    char_corpus = np.concatenate([get_texts(left, tl, cfg["char_cols"]),
                                  get_texts(right, tr, cfg["char_cols"])])
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2).fit(word_corpus)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit(char_corpus)

    for k in idx:
        feats[k]["tfidf_word"] = build_tfidf_sim(idx[k], left, right, word_vec,
                                                 cfg["word_cols"], "tfidf_word")
        feats[k]["tfidf_char"] = build_tfidf_sim(idx[k], left, right, char_vec,
                                                 cfg["char_cols"], "tfidf_char")
        if cfg["numeric_feat"]:
            add_numeric_pair_features(idx[k], left, right, feats[k], cfg["numeric_feat"])
        feats[k] = feats[k].fillna(0)

    return left, right, feats, y, idx


# ===========================================================================
# Evaluate one dataset
# ===========================================================================
def run_single(ds):
    cfg = DATASETS[ds]
    params, config_id, csv_test_f1, chosen_t = read_best_config(ds)
    t0 = time.time()

    left, right, feats, y, idx = build_features(cfg)
    neg, pos = (y["train"] == 0).sum(), (y["train"] == 1).sum()

    clf = make_classifier(params, neg, pos)
    if params["matcher"] == "GradientBoosting":
        sw = np.where(y["train"] == 1, neg / pos, 1.0)
        clf.fit(feats["train"], y["train"], sample_weight=sw)
    else:
        clf.fit(feats["train"], y["train"])

    probs = {k: clf.predict_proba(feats[k])[:, 1] for k in feats}

    # ---- pairwise on the held-out test split (same as the workers) ----
    preds_test = (probs["test"] >= chosen_t).astype(int)
    pw_p = precision_score(y["test"], preds_test, zero_division=0)
    pw_r = recall_score(y["test"], preds_test, zero_division=0)
    pw_f = f1_score(y["test"], preds_test, zero_division=0)

    # ---- candidate universe = train+valid+test union (the blocked set) ----
    cand_index = idx["train"].append(idx["valid"]).append(idx["test"])
    cand_probs = np.concatenate([probs["train"], probs["valid"], probs["test"]])
    pred_mask = cand_probs >= chosen_t
    predicted_pairs = [(l, r) for (l, r), m in zip(cand_index, pred_mask) if m]

    # ---- cluster-level B-cubed over EVERY entity, GT from gt.csv ----
    universe = [f"A:{x}" for x in left.index] + [f"B:{x}" for x in right.index]
    universe_set = set(universe)
    pred_edges = [(f"A:{l}", f"B:{r}") for (l, r) in predicted_pairs]

    gt_pairs_raw = load_gt_pairs(cfg["gt"], sep=cfg["gt_sep"], header=cfg["gt_header"])
    gt_pairs_tagged = [(f"A:{a}", f"B:{b}") for a, b in gt_pairs_raw]
    n_gt_connected = sum(1 for a, b in gt_pairs_tagged
                         if a in universe_set and b in universe_set)
    check_gt_connects(ds, len(gt_pairs_tagged), n_gt_connected)

    entity_to_pred = connected_components(pred_edges, universe)
    entity_to_gt = connected_components(gt_pairs_tagged, universe)
    b_p, b_r, b_f = bcubed(entity_to_pred, entity_to_gt)

    os.makedirs(PAIRS_DIR, exist_ok=True)
    with open(os.path.join(PAIRS_DIR, f"recordlinkage_{ds}_pred_pairs.csv"), "w", newline="") as f:
        wtr = csv.writer(f); wtr.writerow(["left_id", "right_id"]); wtr.writerows(predicted_pairs)
    with open(os.path.join(PAIRS_DIR, f"recordlinkage_{ds}_entities.csv"), "w", newline="") as f:
        wtr = csv.writer(f); wtr.writerow(["entity_id"]); wtr.writerows([[e] for e in universe])

    result = {
        "dataset": ds, "config_id": config_id, "matcher": params["matcher"],
        "chosen_threshold": round(chosen_t, 6), "csv_test_f1": round(csv_test_f1, 6),
        "pairwise_precision": round(pw_p, 6), "pairwise_recall": round(pw_r, 6),
        "pairwise_f1": round(pw_f, 6),
        "bcubed_precision": round(b_p, 6), "bcubed_recall": round(b_r, 6),
        "bcubed_f1": round(b_f, 6),
        "n_entities": len(universe), "n_pred_pairs": len(predicted_pairs),
        "n_gt_pairs": len(gt_pairs_tagged), "time_sec": round(time.time() - t0, 2),
    }
    print("RESULT_JSON:" + json.dumps(result))
    return result


# ===========================================================================
# All-datasets driver (each dataset isolated in its own subprocess)
# ===========================================================================
SUMMARY_COLS = ["dataset", "config_id", "matcher", "chosen_threshold", "csv_test_f1",
                "pairwise_precision", "pairwise_recall", "pairwise_f1",
                "bcubed_precision", "bcubed_recall", "bcubed_f1",
                "n_entities", "n_pred_pairs", "n_gt_pairs", "time_sec", "status"]


def run_all():
    import subprocess
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
            print(f"  -> ERROR\n{proc.stderr[-800:]}", flush=True)
            rows.append({"dataset": ds, "status": "ERROR"})
        else:
            result["status"] = "OK"
            print(f"  -> pairwise F1={result['pairwise_f1']:.4f}  "
                  f"B3 F1={result['bcubed_f1']:.4f}  "
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
