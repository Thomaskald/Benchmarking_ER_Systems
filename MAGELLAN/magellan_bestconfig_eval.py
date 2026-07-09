#!/usr/bin/env python3
"""
Best-config evaluation at TWO levels: pairwise + cluster-level (B-cubed).

Magellan does NOT block+cluster the whole dataset; it trains a supervised matcher
over the pre-generated candidate pairs found in the train/valid/test split files.
So the "predicted clustering" for B-cubed is built from every split pair the best
classifier labels as a match (connected components over the full entity universe).

For each dataset it:
  1. reads the BEST config (max test_f1 among status==OK rows) straight from the
     existing results/magellan_<DS>_configs.csv,
  2. reruns exactly that one config's pipeline (imports the dataset's own worker
     module and reuses its build_candset / tfidf_cosine / prep_matrix /
     make_classifier, so preprocessing & features are identical),
  3. reports PAIRWISE P/R/F1 the same way the workers do (probs_test >= best_t vs
     the test labels) as a sanity cross-check against the existing test_f1,
  4. reports CLUSTER-LEVEL B-cubed P/R/F1: the predicted clustering (connected
     components of all predicted-match pairs over train+valid+test) vs the full
     ground-truth clustering from gt.csv, over EVERY entity in both tables,
  5. dumps the predicted match pairs + the entity id universe under results/pairs/.

Usage
-----
  # evaluate a single dataset (prints RESULT_JSON, writes pair dumps):
  python3 magellan_bestconfig_eval.py D2

  # evaluate ALL datasets (each in its own subprocess) and write the summary CSV:
  python3 magellan_bestconfig_eval.py

Output
------
  results/magellan_bestconfig_eval.csv         one row per dataset, both levels
  results/pairs/magellan_<DS>_pred_pairs.csv    predicted matches (left_id,right_id)
  results/pairs/magellan_<DS>_entities.csv      full entity id universe (tagged)

Notes
-----
* The predicted clustering is built from ALL split pairs (train+valid+test) the
  best classifier scores as a match. Training pairs were used to fit the model, so
  B-cubed *precision* is mildly optimistic -- this is inherent to reconstructing a
  clustering for a supervised matcher and is documented here for transparency.
* If a gt file has a different delimiter/header, fix gt_sep/gt_header below --
  load_gt_pairs fails loudly on a mis-parse and check_gt_connects refuses to
  report bogus B-cubed.
"""
import os
import sys
import csv
import json
import time
import warnings
import subprocess
import importlib
from collections import Counter

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PAIRS_DIR = os.path.join(RESULTS_DIR, "pairs")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "magellan_bestconfig_eval.csv")

DATA_ROOT = "/home/it2022025/er_scalability/datasets"

if HERE not in sys.path:
    sys.path.insert(0, HERE)


# ===========================================================================
# Per-dataset table loaders. Each replicates EXACTLY the loading+cleaning block
# at the top of that dataset's worker main(), and returns (ltable, rtable) in the
# same order the worker feeds build_candset -> left_id indexes ltable, right_id
# indexes rtable. `w` is the imported worker module (for its *_PATH constants and
# custom load_* helpers), so only the cleaning lines live here.
# ===========================================================================
def _load_D2(w):
    import py_entitymatching as em
    import pandas as pd
    abt = em.read_csv_metadata(w.ABT_PATH, key="id", sep="|")
    buy = em.read_csv_metadata(w.BUY_PATH, key="id", sep="|")
    for df in [abt, buy]:
        df["name"] = df["name"].str.lower().str.strip().fillna("")
        df["description"] = df["description"].str.lower().str.strip().fillna("")
    abt["price"] = pd.to_numeric(abt["price"], errors="coerce").fillna(0)
    buy["price"] = pd.to_numeric(buy["price"], errors="coerce").fillna(0)
    return abt, buy


def _load_D3(w):
    import py_entitymatching as em
    import pandas as pd
    amazon = em.read_csv_metadata(w.AMAZON_PATH, key="id", sep="#")
    gp = em.read_csv_metadata(w.GP_PATH, key="id", sep="#")
    for df in [amazon, gp]:
        df["title"] = df["title"].str.lower().str.strip().fillna("")
        df["description"] = df["description"].str.lower().str.strip().fillna("")
        df["manufacturer"] = df["manufacturer"].str.lower().str.strip().fillna("")
    amazon["price"] = pd.to_numeric(amazon["price"], errors="coerce").fillna(0)
    gp["price"] = pd.to_numeric(gp["price"], errors="coerce").fillna(0)
    return amazon, gp


def _load_D4(w):
    import py_entitymatching as em
    import pandas as pd
    acm = em.read_csv_metadata(w.ACM_PATH, key="id", sep="%")
    dblp = em.read_csv_metadata(w.DBLP_PATH, key="id", sep="%")
    for df in [acm, dblp]:
        df["title"] = df["title"].str.lower().str.strip().fillna("")
        df["authors"] = df["authors"].str.lower().str.strip().fillna("")
        df["venue"] = df["venue"].str.lower().str.strip().fillna("")
    acm["year"] = pd.to_numeric(acm["year"], errors="coerce").fillna(0)
    dblp["year"] = pd.to_numeric(dblp["year"], errors="coerce").fillna(0)
    return dblp, acm  # worker feeds build_candset(TRAIN, dblp, acm)


def _load_D5(w):
    import py_entitymatching as em
    import pandas as pd
    imdb = w.load_base_table(w.IMDB_PATH)
    tmdb = w.load_base_table(w.TMDB_PATH)
    em.set_key(imdb, "id")
    em.set_key(tmdb, "id")
    for df in [imdb, tmdb]:
        df["title"] = df["title"].fillna("").astype(str).str.lower().str.strip()
        df["name"] = df["name"].fillna("").astype(str).str.lower().str.strip()
        df["genre_list"] = df["genre_list"].fillna("").astype(str).str.lower().str.strip()
    for df in [imdb, tmdb]:
        df["episodeNumber"] = pd.to_numeric(df["episodeNumber"], errors="coerce").fillna(0)
        df["seasonNumber"] = pd.to_numeric(df["seasonNumber"], errors="coerce").fillna(0)
    return imdb, tmdb


def _load_D6(w):
    import py_entitymatching as em
    import pandas as pd
    imdb = w.load_base_table(w.IMDB_PATH)
    tvdb = w.load_base_table(w.TVDB_PATH)
    em.set_key(imdb, "id")
    em.set_key(tvdb, "id")
    for df in [imdb, tvdb]:
        df["title"] = df["title"].fillna("").astype(str).str.lower().str.strip()
        df["name"] = df["name"].fillna("").astype(str).str.lower().str.strip()
    for df in [imdb, tvdb]:
        df["episodeNumber"] = pd.to_numeric(df["episodeNumber"], errors="coerce").fillna(0)
        df["seasonNumber"] = pd.to_numeric(df["seasonNumber"], errors="coerce").fillna(0)
    return imdb, tvdb


def _load_D7(w):
    import py_entitymatching as em
    import pandas as pd
    tmdb = w.load_movie_table(w.TMDB_PATH)
    tvdb = w.load_movie_table(w.TVDB_PATH)
    em.set_key(tmdb, "id")
    em.set_key(tvdb, "id")
    for df in [tmdb, tvdb]:
        df["title"] = df["title"].fillna("").astype(str).str.lower().str.strip()
        df["name"] = df["name"].fillna("").astype(str).str.lower().str.strip()
        df["abstract"] = df["abstract"].fillna("").astype(str).str.lower().str.strip()
        df["releaseDate"] = df["releaseDate"].fillna("").astype(str).str.lower().str.strip()
    for df in [tmdb, tvdb]:
        df["episodeNumber"] = pd.to_numeric(df["episodeNumber"], errors="coerce").fillna(0)
        df["seasonNumber"] = pd.to_numeric(df["seasonNumber"], errors="coerce").fillna(0)
    return tmdb, tvdb


def _load_D8(w):
    import py_entitymatching as em
    import pandas as pd
    amazon = w.load_base_table(w.AMAZON_PATH)
    walmart = w.load_base_table(w.WALMART_PATH)
    em.set_key(amazon, "id")
    em.set_key(walmart, "id")
    for df in [amazon, walmart]:
        df["title"] = df["title"].fillna("").astype(str).str.lower().str.strip()
        df["modelno"] = df["modelno"].fillna("").astype(str).str.lower().str.strip()
        df["brand"] = df["brand"].fillna("").astype(str).str.lower().str.strip()
        df["dimensions"] = df["dimensions"].fillna("").astype(str).str.lower().str.strip()
    for df in [amazon, walmart]:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
        df["shipweight"] = pd.to_numeric(df["shipweight"], errors="coerce").fillna(0)
    return walmart, amazon  # worker feeds build_candset(TRAIN, walmart, amazon)


def _load_D9(w):
    import py_entitymatching as em
    import pandas as pd
    dblp = w.load_base_table(w.DBLP_PATH)
    scholar = w.load_base_table(w.SCHOLAR_PATH)
    em.set_key(dblp, "id")
    em.set_key(scholar, "id")
    for df in [dblp, scholar]:
        df["title"] = df["title"].fillna("").astype(str).str.lower().str.strip()
        df["authors"] = df["authors"].fillna("").astype(str).str.lower().str.strip()
        df["venue"] = df["venue"].fillna("").astype(str).str.lower().str.strip()
    for df in [dblp, scholar]:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0)
    return dblp, scholar


DATASETS = {
    "D2": dict(worker="magellan_ccer_workerD2", loader=_load_D2,
               gt=f"{DATA_ROOT}/D2/gt.csv", gt_sep="|", gt_header=0),
    "D3": dict(worker="magellan_ccer_workerD3", loader=_load_D3,
               gt=f"{DATA_ROOT}/D3/gt.csv", gt_sep="#", gt_header=0),
    "D4": dict(worker="magellan_ccer_workerD4", loader=_load_D4,
               gt=f"{DATA_ROOT}/D4/gt.csv", gt_sep="%", gt_header=0),
    "D5": dict(worker="magellan_ccer_workerD5", loader=_load_D5,
               gt=f"{DATA_ROOT}/D5/gt.csv", gt_sep="|", gt_header=0),
    "D6": dict(worker="magellan_ccer_workerD6", loader=_load_D6,
               gt=f"{DATA_ROOT}/D6/gt.csv", gt_sep="|", gt_header=0),
    "D7": dict(worker="magellan_ccer_workerD7", loader=_load_D7,
               gt=f"{DATA_ROOT}/D7/gt.csv", gt_sep="|", gt_header=0),
    "D8": dict(worker="magellan_ccer_workerD8", loader=_load_D8,
               gt=f"{DATA_ROOT}/D8/gt.csv", gt_sep="|", gt_header=0),
    "D9": dict(worker="magellan_ccer_workerD9", loader=_load_D9,
               gt=f"{DATA_ROOT}/D9/gt.csv", gt_sep=">", gt_header=0),
}
ALL_DATASETS = list(DATASETS.keys())


# ===========================================================================
# Generic metric helpers
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


def check_gt_connects(ds, n_gt_total, n_gt_connected):
    """Guard: if ground-truth pairs don't reference known entities, B-cubed is
    meaningless (recall collapses). Fail loudly instead of reporting junk."""
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
    """Load ground-truth match pairs as a list of (left, right) string tuples.

    First two columns are the id pair; col0 = ltable id, col1 = rtable id (same
    left/right convention as the split files).
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
# Best-config lookup from the existing search CSV
# ===========================================================================
def read_best_config(ds):
    """Return (cfg_dict_for_make_classifier, config_id, csv_test_f1)."""
    path = os.path.join(RESULTS_DIR, f"magellan_{ds}_configs.csv")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("status") == "OK" and r.get("test_f1") not in (None, "")]
    if not rows:
        raise RuntimeError(f"{ds}: no OK rows in {path}")
    best = max(rows, key=lambda r: float(r["test_f1"]))
    cfg = {"matcher": best["matcher"], "config_id": best["config_id"], "seed": 42}
    for k in ["n_estimators", "max_depth", "learning_rate", "C", "class_weight"]:
        v = best.get(k)
        if v not in (None, ""):
            cfg[k] = v
    return cfg, best["config_id"], float(best["test_f1"])


# ===========================================================================
# Reproduce the worker pipeline for one config, then score both levels
# ===========================================================================
def candset_from_df(df, lt, rt):
    """build_candset, but from an in-memory frame (used for the train+valid+test
    union we score to reconstruct the predicted clustering)."""
    import py_entitymatching as em
    df = df.copy()
    df.insert(0, "_id", range(len(df)))
    em.set_key(df, "_id")
    em.set_ltable(df, lt)
    em.set_rtable(df, rt)
    em.set_fk_ltable(df, "left_id")
    em.set_fk_rtable(df, "right_id")
    return df


def eval_dataset(ds, cfg_dict):
    import numpy as np
    import pandas as pd
    import py_entitymatching as em
    from sklearn.metrics import precision_score, recall_score, f1_score

    spec = DATASETS[ds]
    w = importlib.import_module(spec["worker"])
    np.random.seed(int(cfg_dict.get("seed", 42)))

    lt, rt = spec["loader"](w)

    # ---- reproduce the worker's train/feature/threshold block (identical) ----
    train_cand = w.build_candset(w.TRAIN_PATH, lt, rt)
    valid_cand = w.build_candset(w.VALID_PATH, lt, rt)
    test_cand = w.build_candset(w.TEST_PATH, lt, rt)

    match_f = em.get_features_for_matching(lt, rt, validate_inferred_attr_types=False)
    id_cols = match_f[match_f["feature_name"].str.startswith("id_")].index
    match_f = match_f.drop(id_cols)

    H_train = em.extract_feature_vecs(train_cand, feature_table=match_f,
                                      attrs_after=["label"], show_progress=False)
    meta = ["_id", "left_id", "right_id", "label"]
    base_feat = [c for c in H_train.columns if c not in meta]
    col_means = H_train[base_feat].mean()
    for col in base_feat:
        H_train[col] = H_train[col].fillna(col_means.get(col, 0))
    tfidf_tr, fitted = w.tfidf_cosine(train_cand, lt, rt)
    H_train["tfidf_cosine"] = tfidf_tr
    feat_cols_final = [c for c in H_train.columns if c not in meta]

    X_train = H_train[feat_cols_final].values
    y_train = H_train["label"].values

    X_valid, y_valid = w.prep_matrix(valid_cand, lt, rt, match_f, col_means, fitted, feat_cols_final)
    X_test, y_test = w.prep_matrix(test_cand, lt, rt, match_f, col_means, fitted, feat_cols_final)

    clf = w.make_classifier(cfg_dict)
    clf.fit(X_train, y_train)
    probs_valid = clf.predict_proba(X_valid)[:, 1]
    probs_test = clf.predict_proba(X_test)[:, 1]

    best_t, best_valid_f1 = 0.5, -1.0
    for t in w.THRESHOLD_GRID:
        preds = (probs_valid >= t).astype(int)
        f1 = f1_score(y_valid, preds, zero_division=0)
        if f1 > best_valid_f1:
            best_valid_f1, best_t = f1, float(t)

    # ---- PAIRWISE (exactly as the worker reports it: on the test split) ----
    preds_test = (probs_test >= best_t).astype(int)
    pw_p = precision_score(y_test, preds_test, zero_division=0)
    pw_r = recall_score(y_test, preds_test, zero_division=0)
    pw_f = f1_score(y_test, preds_test, zero_division=0)

    # ---- CLUSTER-LEVEL B-cubed ----
    # Predicted match pairs = every split pair (train+valid+test) the matcher
    # scores >= best_t. These are the only candidate pairs Magellan ever sees.
    raw = pd.concat([pd.read_csv(w.TRAIN_PATH), pd.read_csv(w.VALID_PATH),
                     pd.read_csv(w.TEST_PATH)], ignore_index=True)
    raw = raw.drop_duplicates(subset=["left_id", "right_id"]).reset_index(drop=True)
    union_cand = candset_from_df(raw, lt, rt)
    X_union, _ = w.prep_matrix(union_cand, lt, rt, match_f, col_means, fitted, feat_cols_final)
    probs_union = clf.predict_proba(X_union)[:, 1]
    pos = probs_union >= best_t

    l_ids = raw["left_id"].values
    r_ids = raw["right_id"].values
    predicted_pairs = set()          # native (left_id, right_id) for the dump
    pred_pairs_tagged = []           # tagged L:/R: edges for connected components
    for i in range(len(raw)):
        if pos[i]:
            predicted_pairs.add((l_ids[i], r_ids[i]))
            pred_pairs_tagged.append((f"L:{l_ids[i]}", f"R:{r_ids[i]}"))

    lt_ids = lt["id"].tolist()
    rt_ids = rt["id"].tolist()
    universe = [f"L:{x}" for x in lt_ids] + [f"R:{x}" for x in rt_ids]
    universe_set = set(universe)

    gt_pairs_raw = load_gt_pairs(spec["gt"], sep=spec["gt_sep"], header=spec["gt_header"])
    gt_pairs_tagged = [(f"L:{a}", f"R:{b}") for a, b in gt_pairs_raw]
    n_gt_connected = sum(1 for a, b in gt_pairs_tagged
                         if a in universe_set and b in universe_set)
    check_gt_connects(ds, len(gt_pairs_tagged), n_gt_connected)

    entity_to_pred = connected_components(pred_pairs_tagged, universe)
    entity_to_gt = connected_components(gt_pairs_tagged, universe)
    b_p, b_r, b_f = bcubed(entity_to_pred, entity_to_gt)

    return dict(
        pairwise=(pw_p, pw_r, pw_f), bcubed=(b_p, b_r, b_f),
        chosen_threshold=best_t, valid_f1=best_valid_f1,
        n_entities=len(universe), n_pred_pairs=len(predicted_pairs),
        n_gt_pairs=len(gt_pairs_tagged),
        dump_pairs=sorted(predicted_pairs), entities=universe,
    )


# ===========================================================================
# Single-dataset driver (subprocess entry point)
# ===========================================================================
def run_single(ds):
    cfg_dict, config_id, csv_test_f1 = read_best_config(ds)
    t0 = time.time()
    out = eval_dataset(ds, cfg_dict)

    os.makedirs(PAIRS_DIR, exist_ok=True)
    with open(os.path.join(PAIRS_DIR, f"magellan_{ds}_pred_pairs.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["left_id", "right_id"])
        wtr.writerows(out["dump_pairs"])
    with open(os.path.join(PAIRS_DIR, f"magellan_{ds}_entities.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["entity_id"])
        wtr.writerows([[e] for e in out["entities"]])

    pw, bc = out["pairwise"], out["bcubed"]
    result = {
        "dataset": ds, "config_id": config_id, "family": "ccer",
        "csv_test_f1": round(csv_test_f1, 6),
        "chosen_threshold": round(out["chosen_threshold"], 3),
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
# All-datasets driver
# ===========================================================================
SUMMARY_COLS = ["dataset", "config_id", "family", "csv_test_f1", "chosen_threshold",
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
            print(f"  -> ERROR\n{proc.stderr[-1000:]}", flush=True)
            rows.append({"dataset": ds, "status": "ERROR"})
        else:
            result["status"] = "OK"
            delta = result["pairwise_f1"] - result["csv_test_f1"]
            print(f"  -> pairwise F1={result['pairwise_f1']:.4f} "
                  f"(csv={result['csv_test_f1']:.4f}, dF1={delta:+.4f})  "
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
