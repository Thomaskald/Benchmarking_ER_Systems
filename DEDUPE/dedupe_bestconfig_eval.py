#!/usr/bin/env python3
"""
Dedupe best-config evaluation at TWO levels: pairwise + cluster-level (B-cubed),
BOTH computed on the fixed test set.

Mirrors pyjedai_bestconfig_eval.py, but for the `dedupe` library workers.

Why test-set B-cubed (not full-dataset clustering): clustering the whole dataset
requires scoring dedupe's entire blocked candidate set, which does not scale --
the two largest CCER datasets (D8, D9) time out. It is also inconsistent to score
pairwise on the test set but B-cubed on the full dataset. So B-cubed here is
computed over the SAME test pairs used for pairwise: it always finishes (D8/D9
included) and shares pairwise's universe. This matches how Magellan computes it,
so the metric is uniform across frameworks and gap-free for significance testing.

For each dataset it:
  1. reads the BEST config (max test_f1 among status==OK rows) straight from the
     existing results/dedupe_<DS>_configs.csv,
  2. retrains dedupe with EXACTLY that config (same neg_ratio / recall /
     index_predicates, same seed, same sampling caps as the workers),
  3. scores the fixed test pairs once (used for BOTH metrics),
  4. reports PAIRWISE P/R/F1 on test_set.csv (matches results/dedupe_<DS>_configs.csv),
  5. reports TEST-SET B-cubed P/R/F1: predicted clusters = connected components of
     test pairs scoring >= threshold; true clusters = connected components of test
     pairs with label == 1; B-cubed over the test-set entities,
  6. dumps predicted match pairs + the test-set entity universe under results/pairs/.

Usage
-----
  python3 dedupe_bestconfig_eval.py D2      # one dataset (prints RESULT_JSON)
  python3 dedupe_bestconfig_eval.py         # all datasets, writes summary CSV

Output
------
  results/dedupe_bestconfig_eval.csv          one row per dataset, both levels
  results/pairs/dedupe_<DS>_pred_pairs.csv     predicted matches (left_id,right_id)
  results/pairs/dedupe_<DS>_entities.csv       test-set entity id universe (one col)
"""
import os
import io
import re
import csv
import sys
import json
import time
import random
import subprocess
import warnings
import logging
from collections import Counter

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PAIRS_DIR = os.path.join(RESULTS_DIR, "pairs")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "dedupe_bestconfig_eval.csv")

DATA_ROOT = "/home/it2022025/er_scalability/datasets"
SPLIT_ROOT = "/home/it2022025/er_scalability/train_validation_test_sets"

SEED = 42
# Same training-sample ceilings the DER workers used (OOM guards).
MAX_MATCHES = 2500
MAX_DISTINCT = 7500
CDDB_SAMPLE_SIZE = 2000


# ===========================================================================
# Text / value cleaning -- identical to the workers.
# ===========================================================================
def clean_text(text):
    if text is None:
        return None
    from unidecode import unidecode
    text = unidecode(str(text))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text if text else None


def preprocess_price(price):
    if not price:
        return None
    try:
        return float(str(price).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None


def norm_id(v):
    """Canonical string id. '123', 123, '123.0' -> '123'. Keeps non-numeric as-is."""
    s = str(v).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


# ===========================================================================
# Per-dataset record loaders (mirror each worker's load_csv exactly).
# Each returns {int_id: {field: value, ...}}.
# ===========================================================================
def _load_D2(path):
    data = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|"):
            data[int(row["id"])] = {
                "name": clean_text(row.get("name")),
                "description": clean_text(row.get("description")),
                "price": preprocess_price(row.get("price")),
            }
    return data


def _load_D3(path):
    data = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="#"):
            data[int(row["id"])] = {
                "name": clean_text(row.get("title")),          # D3: title -> name
                "description": clean_text(row.get("description")),
                "manufacturer": clean_text(row.get("manufacturer")),
                "price": preprocess_price(row.get("price")),
            }
    return data


def _load_D4(path):
    data = {}
    with open(path, encoding="utf-8-sig") as f:                # utf-8-sig, '%'
        for row in csv.DictReader(f, delimiter="%"):
            try:
                data[int(str(row["id"]).strip())] = {
                    "title": clean_text(row.get("title")),
                    "authors": clean_text(row.get("authors")),
                    "venue": clean_text(row.get("venue")),
                    "year": clean_text(row.get("year")),
                }
            except (KeyError, ValueError, TypeError):
                continue
    return data


def _make_movie_loader(year_alt):
    """D5/D6/D7 SCADS movie loaders. They differ only in the year fallback key."""
    O = "https://www.scads.de/movieBenchmark/ontology/"

    def _load(path):
        data = {}
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="|"):
                try:
                    data[int(str(row["id"]).strip())] = {
                        "title": clean_text(row.get(O + "title") or row.get(O + "name")),
                        "year": clean_text(row.get(O + "startYear") or row.get(O + year_alt)),
                        "genre": clean_text(row.get(O + "genre_list")),
                        "runtime": clean_text(row.get(O + "runtimeMinutes")
                                              or row.get("http://dbpedia.org/ontology/runtime")),
                    }
                except (KeyError, ValueError, TypeError):
                    continue
        return data
    return _load


def _load_D8(path):
    data = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|"):
            try:
                data[int(str(row["id"]).strip())] = {
                    "title": clean_text(row.get("title")),
                    "modelno": clean_text(row.get("modelno")),
                    "price": preprocess_price(row.get("price")),
                    "brand": clean_text(row.get("brand")),
                    "dimensions": clean_text(row.get("dimensions")),
                }
            except (KeyError, ValueError, TypeError):
                continue
    return data


def _load_D9(path):
    data = {}
    with open(path, encoding="utf-8-sig") as f:                # utf-8-sig, '>'
        for row in csv.DictReader(f, delimiter=">"):
            try:
                data[int(str(row["id"]).strip())] = {
                    "title": clean_text(row.get("title")),
                    "authors": clean_text(row.get("authors")),
                    "venue": clean_text(row.get("venue")),
                    "year": clean_text(row.get("year")),
                }
            except (KeyError, ValueError, TypeError):
                continue
    return data


def _load_cora(path):
    records = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|"):
            try:
                rid = int(row["Entity Id"])
            except (ValueError, TypeError):
                continue
            records[rid] = {
                "title": clean_text(row.get("title")),
                "author": clean_text(row.get("author")),
                "venue": clean_text(row.get("venue")),
                "publisher": clean_text(row.get("publisher")),
                "year": clean_text(row.get("year")),
            }
    return records


def _load_cddb(path):
    records = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):                          # comma-delimited
            try:
                rid = int(row["id"])
            except (ValueError, TypeError):
                continue
            records[rid] = {
                "artist": clean_text(row.get("artist")),
                "title": clean_text(row.get("title")),
                "genre": clean_text(row.get("genre")),
                "category": clean_text(row.get("category")),
                "year": clean_text(row.get("year")),
                # tracks excluded -- very long string, OOM
            }
    return records


# ===========================================================================
# Per-dataset field schemas (mirror each worker's `fields = [...]`).
# Built lazily so `import dedupe` only happens in the child process.
# ===========================================================================
def _fields_D2(d):
    return [d.variables.String("name"),
            d.variables.Text("description"),
            d.variables.Price("price", has_missing=True)]


def _fields_D3(d):
    return [d.variables.String("name"),
            d.variables.Text("description"),
            d.variables.String("manufacturer", has_missing=True),
            d.variables.Price("price", has_missing=True)]


def _fields_D4(d):
    return [d.variables.String("title"),
            d.variables.String("authors"),
            d.variables.String("venue", has_missing=True),
            d.variables.String("year", has_missing=True)]


def _fields_movie(d):
    return [d.variables.String("title"),
            d.variables.String("year", has_missing=True),
            d.variables.String("genre", has_missing=True),
            d.variables.String("runtime", has_missing=True)]


def _fields_D8(d):
    return [d.variables.String("title"),
            d.variables.String("modelno", has_missing=True),
            d.variables.String("brand", has_missing=True),
            d.variables.Price("price", has_missing=True),
            d.variables.String("dimensions", has_missing=True)]


def _fields_D9(d):
    return [d.variables.String("title"),
            d.variables.String("authors"),
            d.variables.String("venue", has_missing=True),
            d.variables.String("year", has_missing=True)]


def _fields_cora(d):
    return [d.variables.Text("title"),
            d.variables.Text("author"),
            d.variables.String("venue", has_missing=True),
            d.variables.String("publisher", has_missing=True),
            d.variables.String("year", has_missing=True)]


def _fields_cddb(d):
    return [d.variables.Text("artist"),
            d.variables.Text("title"),
            d.variables.String("genre", has_missing=True),
            d.variables.String("category", has_missing=True),
            d.variables.String("year", has_missing=True)]


# ===========================================================================
# Dataset registry.
#   family "ccer" -> RecordLink over (d1, d2); left_id side is d1.
#   family "der"  -> Dedupe over a single records dict.
# gt paths/separators are the SAME datasets your pyjedai eval used
# (DATA_ROOT is identical), so they are reused verbatim.
# ===========================================================================
DATASETS = {
    "D2": dict(family="ccer", loader=_load_D2, fields=_fields_D2,
               d1=f"{DATA_ROOT}/D2/abt.csv", d2=f"{DATA_ROOT}/D2/buy.csv",
               test=f"{SPLIT_ROOT}/db2/test_set.csv",
               ),
    "D3": dict(family="ccer", loader=_load_D3, fields=_fields_D3,
               d1=f"{DATA_ROOT}/D3/amazon.csv", d2=f"{DATA_ROOT}/D3/gp.csv",
               test=f"{SPLIT_ROOT}/db3/test_set.csv",
               ),
    "D4": dict(family="ccer", loader=_load_D4, fields=_fields_D4,
               d1=f"{DATA_ROOT}/D4/acm.csv", d2=f"{DATA_ROOT}/D4/dblp.csv",
               test=f"{SPLIT_ROOT}/db4/test_set.csv",
               ),
    "D5": dict(family="ccer", loader=_make_movie_loader("release_year"), fields=_fields_movie,
               d1=f"{DATA_ROOT}/D5/imdb.csv", d2=f"{DATA_ROOT}/D5/tmdb.csv",
               test=f"{SPLIT_ROOT}/db5/test_set.csv",
               ),
    "D6": dict(family="ccer", loader=_make_movie_loader("releaseDate"), fields=_fields_movie,
               d1=f"{DATA_ROOT}/D6/imdb.csv", d2=f"{DATA_ROOT}/D6/tvdb.csv",
               test=f"{SPLIT_ROOT}/db6/test_set.csv",
               ),
    "D7": dict(family="ccer", loader=_make_movie_loader("releaseDate"), fields=_fields_movie,
               d1=f"{DATA_ROOT}/D7/tmdb.csv", d2=f"{DATA_ROOT}/D7/tvdb.csv",
               test=f"{SPLIT_ROOT}/db7/test_set.csv",
               ),
    "D8": dict(family="ccer", loader=_load_D8, fields=_fields_D8,
               d1=f"{DATA_ROOT}/D8/amazon.csv", d2=f"{DATA_ROOT}/D8/walmart.csv",
               test=f"{SPLIT_ROOT}/db8/test_set.csv",
               ),
    "D9": dict(family="ccer", loader=_load_D9, fields=_fields_D9,
               d1=f"{DATA_ROOT}/D9/dblp.csv", d2=f"{DATA_ROOT}/D9/scholar.csv",
               test=f"{SPLIT_ROOT}/db9/test_set.csv",
               ),
    "CORA": dict(family="der", loader=_load_cora, fields=_fields_cora, der_sample=None,
                 data=f"{DATA_ROOT}/cora/cora.csv",
                 test=f"{SPLIT_ROOT}/cora/test_set.csv",
                 ),
    "CDDB": dict(family="der", loader=_load_cddb, fields=_fields_cddb, der_sample=CDDB_SAMPLE_SIZE,
                 data=f"{DATA_ROOT}/CDDB/cddb.csv",
                 test=f"{SPLIT_ROOT}/cddb/test_set.csv",
                 ),
}
ALL_DATASETS = list(DATASETS.keys())


# ===========================================================================
# Generic metric helpers (framework-agnostic; same as pyjedai eval).
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


def prf_at_threshold(scores, labels, t):
    """P/R/F1 for scores thresholded at t vs labels -- exactly the worker's logic.

    The PAIRWISE metric must reproduce the worker (direct scoring of the fixed
    test pairs at the chosen threshold), NOT be derived from the clustering.
    Deriving it from the clustering conflates a pair-classification metric with
    a clustering step and can collapse to 0 at degenerate thresholds (e.g. D4's
    chosen_threshold=0.0). Only B-cubed uses the clustering output.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    preds = [1 if s >= t else 0 for s in scores]
    return (precision_score(labels, preds, zero_division=0),
            recall_score(labels, preds, zero_division=0),
            f1_score(labels, preds, zero_division=0))


def testset_bcubed(pair_keys, scores, labels, threshold, tag_left, tag_right):
    """B-cubed over the TEST-SET-induced clustering (no full-dataset clustering).

    Reuses only the fixed test pairs and the scores we already compute for
    pairwise -- so it ALWAYS finishes (including D8/D9) and is consistent with
    the pairwise metric (same universe):
      - predicted clusters = connected components of test pairs with score >= t
      - true clusters      = connected components of test pairs with label == 1
      - B-cubed over the entities that appear in the test set.
    tag_left/tag_right disambiguate id spaces so left/right ids can't collide
    (CCER: 'A:' / 'B:'; single-source DER: '' / '').
    """
    universe, pred_edges, true_edges, pred_pairs_out = set(), [], [], []
    for (a, b), s, y in zip(pair_keys, scores, labels):
        la, rb = f"{tag_left}{norm_id(a)}", f"{tag_right}{norm_id(b)}"
        universe.add(la)
        universe.add(rb)
        if s >= threshold:
            pred_edges.append((la, rb))
            pred_pairs_out.append((norm_id(a), norm_id(b)))
        if int(y) == 1:
            true_edges.append((la, rb))
    universe = list(universe)
    entity_to_pred = connected_components(pred_edges, universe)
    entity_to_gt = connected_components(true_edges, universe)
    P, R, F = bcubed(entity_to_pred, entity_to_gt)
    return dict(bcubed=(P, R, F), n_entities=len(universe),
                n_pred_pairs=len(pred_pairs_out), n_gt_pairs=len(true_edges),
                dump_pairs=sorted(set(pred_pairs_out)), entities=universe)


# ===========================================================================
# Shared: load split pairs (records must exist on both sides).
# ===========================================================================
def load_split_pairs(path, left_records, right_records):
    import pandas as pd
    pairs, labels = [], []
    for _, row in pd.read_csv(path).iterrows():
        lid, rid = int(row["left_id"]), int(row["right_id"])
        if lid in left_records and rid in right_records:
            pairs.append((lid, rid))
            labels.append(int(row["label"]))
    return pairs, labels


# ===========================================================================
# CCER (clean-clean) evaluation of one dataset.
# ===========================================================================
def eval_ccer(ds, cfg, params):
    import zope
    zope.__path__.append("/home/it2022025/.local/lib/python3.10/site-packages/zope")
    import dedupe

    rng = random.Random(SEED)

    d1 = cfg["loader"](cfg["d1"])
    d2 = cfg["loader"](cfg["d2"])

    train_pairs, train_labels = load_split_pairs(
        cfg["test"].replace("test_set.csv", "train_set.csv"), d1, d2)

    deduper = dedupe.RecordLink(cfg["fields"](dedupe))

    matches, distinct = [], []
    for (lid, rid), label in zip(train_pairs, train_labels):
        (matches if label == 1 else distinct).append((d1[lid], d2[rid]))

    neg_ratio = float(params["neg_ratio"])
    n_keep = int(min(len(distinct), round(len(matches) * neg_ratio)))
    if 0 < n_keep < len(distinct):
        distinct = rng.sample(distinct, n_keep)

    training_file = io.StringIO()
    json.dump({"match": matches, "distinct": distinct}, training_file)
    training_file.seek(0)

    deduper.prepare_training(d1, d2, training_file=training_file)
    deduper.train(recall=float(params["recall"]),
                  index_predicates=bool(params["index_predicates"]))

    t = float(params["chosen_threshold"])
    # Score the fixed test pairs ONCE -- used for BOTH pairwise and test-set
    # B-cubed. No full-dataset clustering: that never scales (D8/D9 timeout) and
    # mixed metrics anyway. Test-set B-cubed reuses these scores, always finishes,
    # and stays consistent with pairwise (same universe).
    test_pairs, test_labels = load_split_pairs(cfg["test"], d1, d2)
    rec = [((lid, d1[lid]), (rid, d2[rid])) for (lid, rid) in test_pairs]
    sm = {(int(l), int(r)): float(s) for (l, r), s in deduper.score(rec)} if rec else {}
    test_scores = [sm.get((lid, rid), 0.0) for (lid, rid) in test_pairs]

    pw_p, pw_r, pw_f = prf_at_threshold(test_scores, test_labels, t)
    b = testset_bcubed(test_pairs, test_scores, test_labels, t, "A:", "B:")

    return dict(pairwise=(pw_p, pw_r, pw_f), bcubed=b["bcubed"],
                n_entities=b["n_entities"], n_pred_pairs=b["n_pred_pairs"],
                n_gt_pairs=b["n_gt_pairs"],
                dump_pairs=b["dump_pairs"], entities=b["entities"])


# ===========================================================================
# DER (dirty ER) evaluation of one dataset.
# ===========================================================================
def eval_der(ds, cfg, params):
    import zope
    zope.__path__.append("/home/it2022025/.local/lib/python3.10/site-packages/zope")
    import dedupe

    rng = random.Random(SEED)

    records = cfg["loader"](cfg["data"])
    train_pairs, train_labels = load_split_pairs(
        cfg["test"].replace("test_set.csv", "train_set.csv"), records, records)

    deduper = dedupe.Dedupe(cfg["fields"](dedupe))

    matches, distinct = [], []
    for (id1, id2), label in zip(train_pairs, train_labels):
        (matches if label == 1 else distinct).append((records[id1], records[id2]))

    if len(matches) > MAX_MATCHES:
        matches = rng.sample(matches, MAX_MATCHES)
    neg_ratio = float(params["neg_ratio"])
    n_keep = min(int(round(len(matches) * neg_ratio)), MAX_DISTINCT, len(distinct))
    if 0 < n_keep < len(distinct):
        distinct = rng.sample(distinct, n_keep)

    training_file = io.StringIO()
    json.dump({"match": matches, "distinct": distinct}, training_file)
    training_file.seek(0)

    if cfg["der_sample"] is not None:                      # CDDB: subsample for prepare_training
        sampled = rng.sample(list(records.keys()), min(cfg["der_sample"], len(records)))
        training_records = {k: records[k] for k in sampled}
    else:                                                  # CORA: full records dict
        training_records = records

    deduper.prepare_training(training_records, training_file=training_file)
    deduper.train(recall=float(params["recall"]),
                  index_predicates=bool(params["index_predicates"]))

    t = float(params["chosen_threshold"])
    # Score the fixed test pairs ONCE -- used for BOTH pairwise and test-set
    # B-cubed. No full-dataset partition: test-set B-cubed reuses these scores,
    # always finishes, and stays consistent with pairwise (same universe).
    # Symmetric fill (matches the DER worker: single source, unordered pairs).
    test_pairs, test_labels = load_split_pairs(cfg["test"], records, records)
    rec = [((lid, records[lid]), (rid, records[rid])) for (lid, rid) in test_pairs]
    sm = {}
    if rec:
        for (l, r), s in deduper.score(rec):
            sm[(int(l), int(r))] = float(s)
            sm[(int(r), int(l))] = float(s)
    test_scores = [sm.get((lid, rid), 0.0) for (lid, rid) in test_pairs]

    pw_p, pw_r, pw_f = prf_at_threshold(test_scores, test_labels, t)
    b = testset_bcubed(test_pairs, test_scores, test_labels, t, "", "")

    return dict(pairwise=(pw_p, pw_r, pw_f), bcubed=b["bcubed"],
                n_entities=b["n_entities"], n_pred_pairs=b["n_pred_pairs"],
                n_gt_pairs=b["n_gt_pairs"],
                dump_pairs=b["dump_pairs"], entities=b["entities"])


# ===========================================================================
# Best-config lookup (params are the same 3 hypers for both families).
# ===========================================================================
def read_best_config(ds):
    path = os.path.join(RESULTS_DIR, f"dedupe_{ds}_configs.csv")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("status") == "OK" and r.get("test_f1") not in (None, "")]
    if not rows:
        raise RuntimeError(f"{ds}: no OK rows in {path}")
    best = max(rows, key=lambda r: float(r["test_f1"]))
    params = dict(
        neg_ratio=float(best["neg_ratio"]),
        recall=float(best["recall"]),
        index_predicates=str(best["index_predicates"]).strip().lower() == "true",
        chosen_threshold=float(best["chosen_threshold"]),
    )
    return params, best["config_id"], float(best["test_f1"])


# ===========================================================================
# Single-dataset driver (subprocess entry point).
# ===========================================================================
def run_single(ds):
    cfg = DATASETS[ds]
    params, config_id, csv_test_f1 = read_best_config(ds)
    t0 = time.time()
    out = eval_ccer(ds, cfg, params) if cfg["family"] == "ccer" else eval_der(ds, cfg, params)

    os.makedirs(PAIRS_DIR, exist_ok=True)
    with open(os.path.join(PAIRS_DIR, f"dedupe_{ds}_pred_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["left_id", "right_id"])
        w.writerows(out["dump_pairs"])
    with open(os.path.join(PAIRS_DIR, f"dedupe_{ds}_entities.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entity_id"])
        w.writerows([[e] for e in out["entities"]])

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

    # Per-dataset summary row -> its OWN file, so parallel single-dataset jobs
    # never clobber each other or the combined dedupe_bestconfig_eval.csv.
    row = dict(result, status="OK")
    per_ds_csv = os.path.join(RESULTS_DIR, f"dedupe_bestconfig_eval_{ds}.csv")
    with open(per_ds_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerow({c: row.get(c) for c in SUMMARY_COLS})
    return result


# ===========================================================================
# All-datasets driver.
# ===========================================================================
SUMMARY_COLS = ["dataset", "config_id", "family", "csv_test_f1",
                "pairwise_precision", "pairwise_recall", "pairwise_f1",
                "bcubed_precision", "bcubed_recall", "bcubed_f1",
                "n_entities", "n_pred_pairs", "n_gt_pairs", "time_sec", "status"]


# Per-dataset walltime guard. B-cubed is now computed on the test set (no
# full-dataset clustering), so every dataset finishes in seconds/minutes -- this
# is just a safety net: if one ever blows past it, mark it TIMEOUT and keep going
# so a single dataset can never starve the rest of the batch.
PER_DS_TIMEOUT_SEC = 2 * 60 * 60


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for ds in ALL_DATASETS:
        print(f"\n=== {ds} ===", flush=True)
        try:
            proc = subprocess.run([sys.executable, os.path.abspath(__file__), ds],
                                  capture_output=True, text=True,
                                  timeout=PER_DS_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            print(f"  -> TIMEOUT (> {PER_DS_TIMEOUT_SEC}s)", flush=True)
            rows.append({"dataset": ds, "status": "TIMEOUT"})
            with open(SUMMARY_CSV, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
                w.writeheader()
                for r in rows:
                    w.writerow({c: r.get(c) for c in SUMMARY_COLS})
            continue
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
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c) for c in SUMMARY_COLS})
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
