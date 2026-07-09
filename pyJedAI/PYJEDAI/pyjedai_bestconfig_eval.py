#!/usr/bin/env python3
"""Best-config evaluation at two levels: pairwise + cluster-level (B-cubed).

Reads each dataset's best config (max test_f1) from results/pyjedai_<DS>_configs.csv,
reruns that one pipeline, and reports pairwise P/R/F1 (on test_set.csv, as the workers
do) plus B-cubed P/R/F1 (predicted vs ground-truth clustering over every entity).

  python3 pyjedai_bestconfig_eval.py D2   # one dataset
  python3 pyjedai_bestconfig_eval.py      # all -> results/pyjedai_bestconfig_eval.csv
                                          #   + results/pairs/pyjedai_<DS>_{pred_pairs,entities}.csv
"""
import os
import sys
import csv
import json
import time
import subprocess
import warnings
import logging
from collections import Counter
from itertools import combinations

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PAIRS_DIR = os.path.join(RESULTS_DIR, "pairs")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "pyjedai_bestconfig_eval.csv")

DATA_ROOT = "/home/it2022025/er_scalability/datasets"
SPLIT_ROOT = "/home/it2022025/er_scalability/train_validation_test_sets"

# family "ccer" = clean-clean pipeline, "der" = embeddings-NN workflow.
# gt_sep/gt_header must match each ground-truth file's actual format.
SCADS_ATTRS = [
    "https://www.scads.de/movieBenchmark/ontology/title",
    "https://www.scads.de/movieBenchmark/ontology/name",
]

DATASETS = {
    "D2": dict(family="ccer", sep="|", attrs=["name"],
               d1=f"{DATA_ROOT}/D2/abt.csv",    d2=f"{DATA_ROOT}/D2/buy.csv",
               test=f"{SPLIT_ROOT}/db2/test_set.csv",
               gt=f"{DATA_ROOT}/D2/gt.csv", gt_sep="|", gt_header=0),
    "D3": dict(family="ccer", sep="#", attrs=["title"],
               d1=f"{DATA_ROOT}/D3/amazon.csv", d2=f"{DATA_ROOT}/D3/gp.csv",
               test=f"{SPLIT_ROOT}/db3/test_set.csv",
               gt=f"{DATA_ROOT}/D3/gt.csv", gt_sep="#", gt_header=0),
    "D4": dict(family="ccer", sep="%", attrs=["title"],
               d1=f"{DATA_ROOT}/D4/dblp.csv",   d2=f"{DATA_ROOT}/D4/acm.csv",
               test=f"{SPLIT_ROOT}/db4/test_set.csv",
               gt=f"{DATA_ROOT}/D4/gt.csv", gt_sep="%", gt_header=0),
    "D5": dict(family="ccer", sep="|", attrs=SCADS_ATTRS,
               d1=f"{DATA_ROOT}/D5/imdb.csv",   d2=f"{DATA_ROOT}/D5/tmdb.csv",
               test=f"{SPLIT_ROOT}/db5/test_set.csv",
               gt=f"{DATA_ROOT}/D5/gt.csv", gt_sep="|", gt_header=0),
    "D6": dict(family="ccer", sep="|", attrs=SCADS_ATTRS,
               d1=f"{DATA_ROOT}/D6/imdb.csv",   d2=f"{DATA_ROOT}/D6/tvdb.csv",
               test=f"{SPLIT_ROOT}/db6/test_set.csv",
               gt=f"{DATA_ROOT}/D6/gt.csv", gt_sep="|", gt_header=0),
    "D7": dict(family="ccer", sep="|", attrs=SCADS_ATTRS,
               d1=f"{DATA_ROOT}/D7/tmdb.csv",   d2=f"{DATA_ROOT}/D7/tvdb.csv",
               test=f"{SPLIT_ROOT}/db7/test_set.csv",
               gt=f"{DATA_ROOT}/D7/gt.csv", gt_sep="|", gt_header=0),
    "D8": dict(family="ccer", sep="|", attrs=["title"],
               d1=f"{DATA_ROOT}/D8/walmart.csv", d2=f"{DATA_ROOT}/D8/amazon.csv",
               test=f"{SPLIT_ROOT}/db8/test_set.csv",
               gt=f"{DATA_ROOT}/D8/gt.csv", gt_sep="|", gt_header=0),
    "D9": dict(family="ccer", sep=">", attrs=["title"],
               d1=f"{DATA_ROOT}/D9/dblp.csv",   d2=f"{DATA_ROOT}/D9/scholar.csv",
               test=f"{SPLIT_ROOT}/db9/test_set.csv",
               gt=f"{DATA_ROOT}/D9/gt.csv", gt_sep=">", gt_header=0),
    "CORA": dict(family="der", sep="|", id_col="Entity Id", attrs=None,
                 data=f"{DATA_ROOT}/cora/cora.csv",
                 test=f"{SPLIT_ROOT}/cora/test_set.csv",
                 gt=f"{DATA_ROOT}/cora/cora_gt.csv", gt_sep="|", gt_header=None),
    "CDDB": dict(family="der", sep=",", id_col="id", attrs=["artist", "title", "year"],
                 data=f"{DATA_ROOT}/CDDB/cddb.csv",
                 test=f"{SPLIT_ROOT}/cddb/test_set.csv",
                 gt=f"{DATA_ROOT}/CDDB/gt.csv", gt_sep=",", gt_header=0),
}

ALL_DATASETS = list(DATASETS.keys())


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


def clusters_to_map(clusters, universe):
    """List-of-sets predicted clusters -> entity->label, with singletons added."""
    m = {}
    for cid, cluster in enumerate(clusters):
        for e in cluster:
            m[e] = cid
    nxt = len(clusters)
    for e in universe:
        if e not in m:
            m[e] = nxt
            nxt += 1
    return m


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


def pairwise_prf(test_df, predicted_pairs, symmetric):
    """Pairwise P/R/F1 on the test split -- same logic as the existing workers."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        lid, rid = row["left_id"], row["right_id"]
        if symmetric:
            lid, rid = str(lid), str(rid)
            hit = (lid, rid) in predicted_pairs or (rid, lid) in predicted_pairs
        else:
            hit = (lid, rid) in predicted_pairs
        y_true.append(int(row["label"]))
        y_pred.append(1 if hit else 0)
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
            f"gt_sep/gt_header/column order. Refusing to report bogus B-cubed.")
    sys.stderr.write(f"[{ds}] GT: {n_gt_connected}/{n_gt_total} pairs connect "
                     f"({frac:.1%}).\n")


def load_gt_pairs(path, sep, header):
    """Ground-truth match pairs as (left, right) string tuples; first two columns."""
    import pandas as pd
    df = pd.read_csv(path, sep=sep, header=header, engine="python", dtype=str)
    df = df.fillna("")
    if df.shape[1] < 2:
        raise RuntimeError(
            f"GT file {path} parsed into {df.shape[1]} column with sep={sep!r} "
            f"(first row: {df.iloc[0,0]!r}). Wrong delimiter -- fix gt_sep for this dataset.")
    left_col, right_col = df.columns[0], df.columns[1]
    return [(str(a), str(b)) for a, b in zip(df[left_col], df[right_col])]


def eval_ccer(ds, cfg, params):
    import pandas as pd
    from pyjedai.datamodel import Data
    from pyjedai.block_building import StandardBlocking
    from pyjedai.block_cleaning import BlockPurging, BlockFiltering
    from pyjedai.comparison_cleaning import CardinalityEdgePruning, CardinalityNodePruning
    from pyjedai.matching import EntityMatching
    from pyjedai.clustering import UniqueMappingClustering, ConnectedComponentsClustering

    CC = {"CardinalityEdgePruning": CardinalityEdgePruning,
          "CardinalityNodePruning": CardinalityNodePruning}
    CL = {"UniqueMappingClustering": UniqueMappingClustering,
          "ConnectedComponentsClustering": ConnectedComponentsClustering}

    d1 = pd.read_csv(cfg["d1"], sep=cfg["sep"], engine="python", na_filter=False)
    d2 = pd.read_csv(cfg["d2"], sep=cfg["sep"], engine="python", na_filter=False)
    data = Data(dataset_1=d1, id_column_name_1="id",
                dataset_2=d2, id_column_name_2="id")
    data.clean_dataset(remove_stopwords=False, remove_punctuation=False,
                       remove_numbers=False, remove_unicodes=False)

    bb = StandardBlocking()
    blocks = bb.build_blocks(data, attributes_1=cfg["attrs"], attributes_2=cfg["attrs"])
    cleaned = BlockPurging().process(blocks, data, tqdm_disable=True)
    filtered = BlockFiltering(ratio=params["ratio"]).process(cleaned, data, tqdm_disable=True)
    mb = CC[params["comparison_cleaner"]](weighting_scheme=params["weighting_scheme"])
    candidates = mb.process(filtered, data, tqdm_disable=True)
    em = EntityMatching(metric=params["metric"], tokenizer=params["tokenizer"],
                        vectorizer=params["vectorizer"], qgram=int(params["qgram"]),
                        similarity_threshold=0.0)
    pairs_graph = em.predict(candidates, data, tqdm_disable=True)

    t = float(params["chosen_threshold"])
    clusters = CL[params["clusterer"]]().process(pairs_graph, data, similarity_threshold=t)

    n1 = len(d1)
    d1_ids = d1["id"].tolist()
    d2_ids = d2["id"].tolist()

    def tag(idx):
        return ("A:" + str(d1_ids[idx])) if idx < n1 else ("B:" + str(d2_ids[idx - n1]))

    # predicted_pairs: native (d1_id, d2_id) for pairwise scoring/dump.
    # pred_clusters_tagged: A:/B:-tagged entities for B-cubed (ids can overlap across sources).
    predicted_pairs = set()
    pred_clusters_tagged = []
    for cl in clusters:
        ids = list(cl)
        a_ids = [i for i in ids if i < n1]
        b_ids = [i for i in ids if i >= n1]
        for a in a_ids:
            for b in b_ids:
                predicted_pairs.add((d1_ids[a], d2_ids[b - n1]))
        pred_clusters_tagged.append({tag(i) for i in ids})

    universe = [f"A:{x}" for x in d1_ids] + [f"B:{x}" for x in d2_ids]
    universe_set = set(universe)

    gt_pairs_raw = load_gt_pairs(cfg["gt"], sep=cfg["gt_sep"], header=cfg["gt_header"])
    gt_pairs_tagged = [(f"A:{a}", f"B:{b}") for a, b in gt_pairs_raw]
    n_gt_connected = sum(1 for a, b in gt_pairs_tagged
                         if a in universe_set and b in universe_set)
    check_gt_connects(ds, len(gt_pairs_tagged), n_gt_connected)

    entity_to_pred = clusters_to_map(pred_clusters_tagged, universe)
    entity_to_gt = connected_components(gt_pairs_tagged, universe)

    b_p, b_r, b_f = bcubed(entity_to_pred, entity_to_gt)

    test_df = pd.read_csv(cfg["test"])
    pw_p, pw_r, pw_f = pairwise_prf(test_df, predicted_pairs, symmetric=False)

    dump_pairs = [(a, b) for (a, b) in predicted_pairs]
    return dict(pairwise=(pw_p, pw_r, pw_f), bcubed=(b_p, b_r, b_f),
                n_entities=len(universe), n_pred_pairs=len(predicted_pairs),
                n_gt_pairs=len(gt_pairs_tagged),
                dump_pairs=dump_pairs, entities=universe)


def eval_der(ds, cfg, params):
    import pandas as pd
    from pyjedai.datamodel import Data
    from pyjedai.workflow import EmbeddingsNNWorkFlow
    from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
    from pyjedai.clustering import ConnectedComponentsClustering

    id_col = cfg["id_col"]
    df = pd.read_csv(cfg["data"], sep=cfg["sep"], engine="python", na_filter=False).astype(str)
    gt_df = pd.read_csv(cfg["gt"], sep=cfg["gt_sep"], header=cfg["gt_header"], engine="python")
    if cfg["gt_header"] is None:
        gt_df.columns = ["left_id", "right_id"]
    else:
        gt_df = gt_df.rename(columns={"id1": "left_id", "id2": "right_id"})

    data = Data(dataset_1=df, id_column_name_1=id_col, ground_truth=gt_df)

    bb = dict(method=EmbeddingsNNBlockBuilding,
              params=dict(vectorizer=params["vectorizer"], similarity_search="faiss"),
              exec_params=dict(top_k=int(params["top_k"]),
                               similarity_distance=params["similarity_distance"],
                               load_embeddings_if_exist=False, save_embeddings=False))
    if cfg["attrs"] is not None:
        bb["attributes_1"] = cfg["attrs"]

    w = EmbeddingsNNWorkFlow(
        block_building=bb,
        clustering=dict(method=ConnectedComponentsClustering,
                        exec_params=dict(similarity_threshold=float(params["similarity_threshold"]))),
        name=f"{ds}-DER-best",
    )
    w.run(data, verbose=False)

    entity_ids = [str(x) for x in df[id_col].tolist()]
    universe = list(entity_ids)

    pred_clusters = [{entity_ids[i] for i in cl} for cl in w.clusters]
    predicted_pairs = set()
    for cl in w.clusters:
        for i, j in combinations(sorted(cl), 2):
            predicted_pairs.add((entity_ids[i], entity_ids[j]))

    gt_pairs = [(str(a), str(b)) for a, b in zip(gt_df["left_id"], gt_df["right_id"])]
    universe_set = set(universe)
    n_gt_connected = sum(1 for a, b in gt_pairs
                         if a in universe_set and b in universe_set)
    check_gt_connects(ds, len(gt_pairs), n_gt_connected)

    entity_to_pred = clusters_to_map(pred_clusters, universe)
    entity_to_gt = connected_components(gt_pairs, universe)

    b_p, b_r, b_f = bcubed(entity_to_pred, entity_to_gt)

    test_df = pd.read_csv(cfg["test"])
    pw_p, pw_r, pw_f = pairwise_prf(test_df, predicted_pairs, symmetric=True)

    dump_pairs = [(a, b) for (a, b) in predicted_pairs]
    return dict(pairwise=(pw_p, pw_r, pw_f), bcubed=(b_p, b_r, b_f),
                n_entities=len(universe), n_pred_pairs=len(predicted_pairs),
                n_gt_pairs=len(gt_pairs),
                dump_pairs=dump_pairs, entities=universe)


def read_best_config(ds):
    """Return (params_dict, config_id) for the max-test_f1 OK row of this dataset."""
    path = os.path.join(RESULTS_DIR, f"pyjedai_{ds}_configs.csv")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("status") == "OK" and r.get("test_f1") not in (None, "")]
    if not rows:
        raise RuntimeError(f"{ds}: no OK rows in {path}")
    best = max(rows, key=lambda r: float(r["test_f1"]))
    family = DATASETS[ds]["family"]
    if family == "ccer":
        params = dict(
            ratio=float(best["ratio"]),
            comparison_cleaner=best["comparison_cleaner"],
            weighting_scheme=best["weighting_scheme"],
            vectorizer=best["vectorizer"], metric=best["metric"],
            tokenizer=best["tokenizer"], qgram=int(best["qgram"]),
            clusterer=best["clusterer"],
            chosen_threshold=float(best["chosen_threshold"]),
        )
    else:
        params = dict(
            vectorizer=best["vectorizer"],
            similarity_distance=best["similarity_distance"],
            top_k=int(best["top_k"]),
            similarity_threshold=float(best["chosen_threshold"]),
        )
    return params, best["config_id"], float(best["test_f1"])


def run_single(ds):
    cfg = DATASETS[ds]
    params, config_id, csv_test_f1 = read_best_config(ds)
    t0 = time.time()
    if cfg["family"] == "ccer":
        out = eval_ccer(ds, cfg, params)
    else:
        out = eval_der(ds, cfg, params)

    os.makedirs(PAIRS_DIR, exist_ok=True)
    with open(os.path.join(PAIRS_DIR, f"pyjedai_{ds}_pred_pairs.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["left_id", "right_id"])
        wtr.writerows(out["dump_pairs"])
    with open(os.path.join(PAIRS_DIR, f"pyjedai_{ds}_entities.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["entity_id"])
        wtr.writerows([[e] for e in out["entities"]])

    pw = out["pairwise"]
    bc = out["bcubed"]
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
