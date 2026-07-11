#!/usr/bin/env python3
"""
Best-config evaluation at TWO levels: pairwise + test-set B-cubed.

Both metrics are computed on the fixed test set, uniform with the DEDUPE and
pyJedAI evals so the numbers are directly comparable across frameworks.

For each dataset it:
  1. reads the BEST config (max test_f1 among status==OK rows) straight from the
     existing results/magellan_<DS>_configs.csv,
  2. reruns exactly that one config's pipeline (imports the dataset's own worker
     module and reuses its build_candset / tfidf_cosine / prep_matrix /
     make_classifier, so preprocessing & features are identical),
  3. scores the fixed test pairs once (used for BOTH metrics),
  4. reports PAIRWISE P/R/F1 on the test split (matches results/magellan_<DS>_configs.csv),
  5. reports TEST-SET B-cubed P/R/F1: predicted clusters = connected components of
     test pairs scoring >= threshold; true clusters = connected components of test
     pairs with label == 1; B-cubed over the test-set entities,
  6. dumps predicted match pairs + the test-set entity universe under results/pairs/.

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
  results/pairs/magellan_<DS>_entities.csv      test-set entity id universe (tagged)
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
    "D2": dict(worker="magellan_ccer_workerD2", loader=_load_D2),
    "D3": dict(worker="magellan_ccer_workerD3", loader=_load_D3),
    "D4": dict(worker="magellan_ccer_workerD4", loader=_load_D4),
    "D5": dict(worker="magellan_ccer_workerD5", loader=_load_D5),
    "D6": dict(worker="magellan_ccer_workerD6", loader=_load_D6),
    "D7": dict(worker="magellan_ccer_workerD7", loader=_load_D7),
    "D8": dict(worker="magellan_ccer_workerD8", loader=_load_D8),
    "D9": dict(worker="magellan_ccer_workerD9", loader=_load_D9),
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


def testset_metrics(test_cand, probs, labels, threshold, tag_left="A:", tag_right="B:"):
    """Pairwise + test-set B-cubed over the fixed test pairs (uniform with the
    DEDUPE and pyJedAI evals):
      - predicted clusters = connected components of test pairs scored >= threshold
      - true clusters      = connected components of test pairs with label == 1
      - both metrics over the entities that appear in the test set.
    tag_left/tag_right disambiguate the two id spaces (CCER: 'A:'/'B:').
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    left_ids = test_cand["left_id"].values
    right_ids = test_cand["right_id"].values
    universe, pred_edges, true_edges, pred_pairs_out = set(), [], [], []
    y_pred = []
    for i in range(len(test_cand)):
        na, nb = left_ids[i], right_ids[i]
        la, rb = f"{tag_left}{na}", f"{tag_right}{nb}"
        universe.add(la)
        universe.add(rb)
        matched = probs[i] >= threshold
        y_pred.append(1 if matched else 0)
        if matched:
            pred_edges.append((la, rb))
            pred_pairs_out.append((na, nb))
        if int(labels[i]) == 1:
            true_edges.append((la, rb))
    universe = list(universe)
    P, R, F = bcubed(connected_components(pred_edges, universe),
                     connected_components(true_edges, universe))
    pw = (precision_score(labels, y_pred, zero_division=0),
          recall_score(labels, y_pred, zero_division=0),
          f1_score(labels, y_pred, zero_division=0))
    return dict(pairwise=pw, bcubed=(P, R, F), n_entities=len(universe),
                n_pred_pairs=len(pred_pairs_out), n_gt_pairs=len(true_edges),
                dump_pairs=sorted(set(pred_pairs_out)), entities=universe)


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
def eval_dataset(ds, cfg_dict):
    import numpy as np
    import py_entitymatching as em
    from sklearn.metrics import f1_score

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

    return testset_metrics(test_cand, probs_test, y_test, best_t)


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
