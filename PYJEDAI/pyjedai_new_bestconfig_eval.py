#!/usr/bin/env python3
"""Best-config evaluation for the TWO NEW pyJedAI pipelines.

Same protocol and same output columns as pyjedai_bestconfig_eval.py -- this one
just covers the other half of the 2x2 grid:

    pyjedai_bestconfig_eval.py  (existing, DO NOT RERUN)
        D2..D9  syntactic      CORA/CDDB  semantic
    this script
        D2..D9  SEMANTIC       CORA/CDDB  SYNTACTIC

For each dataset it reads the best config (max test_f1 among status==OK) from
results/pyjedai_{sem,syn}_<DS>_configs.csv, re-runs exactly that configuration,
scores the fixed test pairs once, and reports pairwise AND test-set B-cubed
P/R/F1 -- identical definitions to the existing evaluator, so the numbers are
directly comparable across all frameworks.

Usage
-----
  python3 pyjedai_new_bestconfig_eval.py D2      # one dataset -> RESULT_JSON
  python3 pyjedai_new_bestconfig_eval.py         # all 10 -> both summary CSVs
  python3 pyjedai_new_bestconfig_eval.py --merge # redo the merge step only

Output
------
  results/pyjedai_sem_bestconfig_eval.csv   8 new SEMANTIC CCER rows
                                            + CORA/CDDB copied from the existing
                                              evaluator (they are already semantic)
  results/pyjedai_syn_bestconfig_eval.csv   2 new SYNTACTIC DER rows
                                            + D2..D9 copied from the existing
                                              evaluator (they are already syntactic)
  results/pairs/pyjedai_{sem,syn}_<DS>_{pred_pairs,entities}.csv

The existing results/pyjedai_bestconfig_eval.csv is READ ONLY -- never modified.
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

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PAIRS_DIR = os.path.join(RESULTS_DIR, "pairs")

EXISTING_CSV = os.path.join(RESULTS_DIR, "pyjedai_bestconfig_eval.csv")
SEM_CSV = os.path.join(RESULTS_DIR, "pyjedai_sem_bestconfig_eval.csv")
SYN_CSV = os.path.join(RESULTS_DIR, "pyjedai_syn_bestconfig_eval.csv")

DATA_ROOT = "/home/it2022025/er_scalability/datasets"
SPLIT_ROOT = "/home/it2022025/er_scalability/train_validation_test_sets"

# variant "sem_ccer" = embeddings-NN on two clean sources (the CORA/CDDB method)
# variant "syn_der"  = blocking chain on one dirty source (the D2..D9 method)
DATASETS = {
    "D2": dict(variant="sem_ccer", sep="|",
               d1=f"{DATA_ROOT}/D2/abt.csv",     d2=f"{DATA_ROOT}/D2/buy.csv",
               test=f"{SPLIT_ROOT}/db2/test_set.csv"),
    "D3": dict(variant="sem_ccer", sep="#",
               d1=f"{DATA_ROOT}/D3/amazon.csv",  d2=f"{DATA_ROOT}/D3/gp.csv",
               test=f"{SPLIT_ROOT}/db3/test_set.csv"),
    "D4": dict(variant="sem_ccer", sep="%",
               d1=f"{DATA_ROOT}/D4/dblp.csv",    d2=f"{DATA_ROOT}/D4/acm.csv",
               test=f"{SPLIT_ROOT}/db4/test_set.csv"),
    "D5": dict(variant="sem_ccer", sep="|",
               d1=f"{DATA_ROOT}/D5/imdb.csv",    d2=f"{DATA_ROOT}/D5/tmdb.csv",
               test=f"{SPLIT_ROOT}/db5/test_set.csv"),
    "D6": dict(variant="sem_ccer", sep="|",
               d1=f"{DATA_ROOT}/D6/imdb.csv",    d2=f"{DATA_ROOT}/D6/tvdb.csv",
               test=f"{SPLIT_ROOT}/db6/test_set.csv"),
    "D7": dict(variant="sem_ccer", sep="|",
               d1=f"{DATA_ROOT}/D7/tmdb.csv",    d2=f"{DATA_ROOT}/D7/tvdb.csv",
               test=f"{SPLIT_ROOT}/db7/test_set.csv"),
    "D8": dict(variant="sem_ccer", sep="|",
               d1=f"{DATA_ROOT}/D8/walmart.csv", d2=f"{DATA_ROOT}/D8/amazon.csv",
               test=f"{SPLIT_ROOT}/db8/test_set.csv"),
    "D9": dict(variant="sem_ccer", sep=">",
               d1=f"{DATA_ROOT}/D9/dblp.csv",    d2=f"{DATA_ROOT}/D9/scholar.csv",
               test=f"{SPLIT_ROOT}/db9/test_set.csv"),
    "CORA": dict(variant="syn_der", sep="|", id_col="Entity Id", attrs=None,
                 data=f"{DATA_ROOT}/cora/cora.csv",
                 test=f"{SPLIT_ROOT}/cora/test_set.csv",
                 gt=f"{DATA_ROOT}/cora/cora_gt.csv", gt_sep="|", gt_header=None),
    "CDDB": dict(variant="syn_der", sep=",", id_col="id",
                 attrs=["artist", "title", "year"],
                 data=f"{DATA_ROOT}/CDDB/cddb.csv",
                 test=f"{SPLIT_ROOT}/cddb/test_set.csv",
                 gt=f"{DATA_ROOT}/CDDB/gt.csv", gt_sep=",", gt_header=0),
}
ALL_DATASETS = list(DATASETS.keys())
PREFIX = {"sem_ccer": "sem", "syn_der": "syn"}
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


def norm_id(v):
    """Canonical string id. '123', 123, '123.0' -> '123'. Keeps non-numeric as-is."""
    s = str(v).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


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


def testset_metrics(test_df, is_match, tag_left, tag_right, symmetric):
    """Pairwise + test-set B-cubed over the fixed test pairs (uniform with DEDUPE).

    is_match(na, nb) -> bool tells whether the model co-clustered the two records.
      - predicted clusters = connected components of test pairs called a match
      - true clusters      = connected components of test pairs with label == 1
      - both metrics over the entities that appear in the test set.
    tag_left/tag_right disambiguate id spaces (CCER: 'A:'/'B:'; DER: ''/'').
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    universe, pred_edges, true_edges, pred_pairs_out = set(), [], [], []
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        na, nb = norm_id(row["left_id"]), norm_id(row["right_id"])
        la, rb = f"{tag_left}{na}", f"{tag_right}{nb}"
        universe.add(la)
        universe.add(rb)
        matched = is_match(na, nb) or (symmetric and is_match(nb, na))
        y_true.append(int(row["label"]))
        y_pred.append(1 if matched else 0)
        if matched:
            pred_edges.append((la, rb))
            pred_pairs_out.append((na, nb))
        if int(row["label"]) == 1:
            true_edges.append((la, rb))
    universe = list(universe)
    P, R, F = bcubed(connected_components(pred_edges, universe),
                     connected_components(true_edges, universe))
    pw = (precision_score(y_true, y_pred, zero_division=0),
          recall_score(y_true, y_pred, zero_division=0),
          f1_score(y_true, y_pred, zero_division=0))
    return dict(pairwise=pw, bcubed=(P, R, F), n_entities=len(universe),
                n_pred_pairs=len(pred_pairs_out), n_gt_pairs=len(true_edges),
                dump_pairs=sorted(set(pred_pairs_out)), entities=universe)


def eval_sem_ccer(ds, cfg, params):
    """SEMANTIC pipeline on a CCER dataset -- mirrors pyjedai_sem_ccer_worker<DS>.py."""
    import pandas as pd
    from pyjedai.datamodel import Data
    from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
    from pyjedai.clustering import UniqueMappingClustering, ConnectedComponentsClustering

    CL = {"UniqueMappingClustering": UniqueMappingClustering,
          "ConnectedComponentsClustering": ConnectedComponentsClustering}

    d1 = pd.read_csv(cfg["d1"], sep=cfg["sep"], engine="python", na_filter=False)
    d2 = pd.read_csv(cfg["d2"], sep=cfg["sep"], engine="python", na_filter=False)
    data = Data(dataset_1=d1, id_column_name_1="id",
                dataset_2=d2, id_column_name_2="id")
    data.clean_dataset(remove_stopwords=False, remove_punctuation=False,
                       remove_numbers=False, remove_unicodes=False)

    emb = EmbeddingsNNBlockBuilding(vectorizer=params["vectorizer"],
                                    similarity_search="faiss")
    # with_entity_matching=True -> returns (blocks, graph); clustering wants the graph.
    _blocks, pairs_graph = emb.build_blocks(
        data,
        top_k=int(params["top_k"]),
        similarity_distance=params["similarity_distance"],
        with_entity_matching=True,
        load_embeddings_if_exist=False,
        save_embeddings=False,
        tqdm_disable=True,
    )
    t = float(params["similarity_threshold"])
    clusters = CL[params["clusterer"]]().process(pairs_graph, data, similarity_threshold=t)

    n1 = len(d1)
    d1_ids = d1["id"].tolist()
    d2_ids = d2["id"].tolist()
    predicted_pairs = set()
    for cl in clusters:
        ids = list(cl)
        a_ids = [i for i in ids if i < n1]
        b_ids = [i for i in ids if i >= n1]
        for a in a_ids:
            for b in b_ids:
                predicted_pairs.add((norm_id(d1_ids[a]), norm_id(d2_ids[b - n1])))

    test_df = pd.read_csv(cfg["test"])
    return testset_metrics(test_df, lambda na, nb: (na, nb) in predicted_pairs,
                           "A:", "B:", symmetric=False)


def eval_syn_der(ds, cfg, params):
    """SYNTACTIC pipeline on a DER dataset -- mirrors pyjedai_syn_der_worker<DS>.py."""
    import pandas as pd
    from pyjedai.datamodel import Data
    from pyjedai.block_building import StandardBlocking
    from pyjedai.block_cleaning import BlockPurging, BlockFiltering
    from pyjedai.comparison_cleaning import CardinalityEdgePruning, CardinalityNodePruning
    from pyjedai.matching import EntityMatching
    from pyjedai.clustering import ConnectedComponentsClustering

    CC = {"CardinalityEdgePruning": CardinalityEdgePruning,
          "CardinalityNodePruning": CardinalityNodePruning}

    id_col = cfg["id_col"]
    df = pd.read_csv(cfg["data"], sep=cfg["sep"], engine="python",
                     na_filter=False).astype(str)
    gt_df = pd.read_csv(cfg["gt"], sep=cfg["gt_sep"], header=cfg["gt_header"],
                        engine="python")
    if cfg["gt_header"] is None:
        gt_df.columns = ["left_id", "right_id"]
    else:
        gt_df = gt_df.rename(columns={"id1": "left_id", "id2": "right_id"})

    data = Data(dataset_1=df, id_column_name_1=id_col, ground_truth=gt_df)
    data.clean_dataset(remove_stopwords=False, remove_punctuation=False,
                       remove_numbers=False, remove_unicodes=False)

    blocks = StandardBlocking().build_blocks(data, attributes_1=cfg["attrs"])
    cleaned = BlockPurging().process(blocks, data, tqdm_disable=True)
    filtered = BlockFiltering(ratio=params["ratio"]).process(cleaned, data, tqdm_disable=True)
    mb = CC[params["comparison_cleaner"]](weighting_scheme=params["weighting_scheme"])
    candidates = mb.process(filtered, data, tqdm_disable=True)
    em = EntityMatching(metric=params["metric"], tokenizer=params["tokenizer"],
                        vectorizer=params["vectorizer"], qgram=int(params["qgram"]),
                        similarity_threshold=0.0)
    pairs_graph = em.predict(candidates, data, tqdm_disable=True)

    t = float(params["chosen_threshold"])
    # Dirty ER: connected components only (UniqueMapping enforces 1-to-1).
    clusters = ConnectedComponentsClustering().process(pairs_graph, data,
                                                       similarity_threshold=t)
    entity_ids = [norm_id(x) for x in df[id_col].tolist()]
    ent2cluster = {}
    for cid, cl in enumerate(clusters):
        for i in cl:
            ent2cluster[entity_ids[i]] = cid

    def is_match(na, nb):
        ca = ent2cluster.get(na)
        return ca is not None and ca == ent2cluster.get(nb)

    test_df = pd.read_csv(cfg["test"])
    return testset_metrics(test_df, is_match, "", "", symmetric=True)


def read_best_config(ds):
    """(params, config_id, csv_test_f1) for the max-test_f1 OK row of this dataset."""
    variant = DATASETS[ds]["variant"]
    path = os.path.join(RESULTS_DIR, f"pyjedai_{PREFIX[variant]}_{ds}_configs.csv")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("status") == "OK" and r.get("test_f1") not in (None, "")]
    if not rows:
        raise RuntimeError(f"{ds}: no OK rows in {path}")
    best = max(rows, key=lambda r: float(r["test_f1"]))
    if variant == "sem_ccer":
        params = dict(
            vectorizer=best["vectorizer"],
            similarity_distance=best["similarity_distance"],
            top_k=int(best["top_k"]),
            similarity_threshold=float(best["chosen_threshold"]),
            clusterer=best["clusterer"],
        )
    else:
        params = dict(
            ratio=float(best["ratio"]),
            comparison_cleaner=best["comparison_cleaner"],
            weighting_scheme=best["weighting_scheme"],
            vectorizer=best["vectorizer"], metric=best["metric"],
            tokenizer=best["tokenizer"], qgram=int(best["qgram"]),
            chosen_threshold=float(best["chosen_threshold"]),
        )
    return params, best["config_id"], float(best["test_f1"])


def run_single(ds):
    cfg = DATASETS[ds]
    params, config_id, csv_test_f1 = read_best_config(ds)
    t0 = time.time()
    out = eval_sem_ccer(ds, cfg, params) if cfg["variant"] == "sem_ccer" \
        else eval_syn_der(ds, cfg, params)

    tag = PREFIX[cfg["variant"]]
    os.makedirs(PAIRS_DIR, exist_ok=True)
    with open(os.path.join(PAIRS_DIR, f"pyjedai_{tag}_{ds}_pred_pairs.csv"), "w", newline="") as f:
        wtr = csv.writer(f); wtr.writerow(["left_id", "right_id"]); wtr.writerows(out["dump_pairs"])
    with open(os.path.join(PAIRS_DIR, f"pyjedai_{tag}_{ds}_entities.csv"), "w", newline="") as f:
        wtr = csv.writer(f); wtr.writerow(["entity_id"]); wtr.writerows([[e] for e in out["entities"]])

    pw, bc = out["pairwise"], out["bcubed"]
    result = {
        "dataset": ds, "config_id": config_id,
        "family": "ccer" if cfg["variant"] == "sem_ccer" else "der",
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


def _write(path, rows):
    with open(path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({c: r.get(c) for c in SUMMARY_COLS})


def merge_with_existing(new_rows):
    """Assemble two COMPLETE 10-dataset methods.

    pyJedAI-sem = 8 new SEMANTIC ccer rows + CORA/CDDB copied from the existing
                  evaluator (those were already produced by the semantic workflow)
    pyJedAI-syn = D2..D9 copied from the existing evaluator (already syntactic)
                  + 2 new SYNTACTIC der rows

    The existing CSV is only read. Rows are copied verbatim, never recomputed.
    """
    if not os.path.exists(EXISTING_CSV):
        print(f"  [warn] {EXISTING_CSV} missing -- writing only the new rows")
        old = []
    else:
        with open(EXISTING_CSV) as f:
            old = list(csv.DictReader(f))

    old_ccer = [r for r in old if r.get("family") == "ccer"]   # syntactic
    old_der = [r for r in old if r.get("family") == "der"]     # semantic
    new_ccer = [r for r in new_rows if r.get("family") == "ccer"]  # semantic
    new_der = [r for r in new_rows if r.get("family") == "der"]    # syntactic

    order = {ds: i for i, ds in enumerate(["D2","D3","D4","D5","D6","D7","D8","D9","CORA","CDDB"])}
    sem = sorted(new_ccer + old_der, key=lambda r: order.get(r["dataset"], 99))
    syn = sorted(old_ccer + new_der, key=lambda r: order.get(r["dataset"], 99))
    _write(SEM_CSV, sem)
    _write(SYN_CSV, syn)
    print(f"\n  pyJedAI-sem -> {SEM_CSV}  ({len(new_ccer)} new + {len(old_der)} copied)")
    print(f"  pyJedAI-syn -> {SYN_CSV}  ({len(old_ccer)} copied + {len(new_der)} new)")


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for ds in ALL_DATASETS:
        v = DATASETS[ds]["variant"]
        print(f"\n=== {ds}  [{v}] ===", flush=True)
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), ds],
                              capture_output=True, text=True)
        result = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                result = json.loads(line[len("RESULT_JSON:"):])
        if result is None:
            print(f"  -> ERROR\n{proc.stderr[-800:]}", flush=True)
            rows.append({"dataset": ds, "status": "ERROR",
                         "family": "ccer" if v == "sem_ccer" else "der"})
        else:
            result["status"] = "OK"
            print(f"  -> pairwise F1={result['pairwise_f1']:.4f}  "
                  f"B3 F1={result['bcubed_f1']:.4f}  "
                  f"(P={result['bcubed_precision']:.3f} R={result['bcubed_recall']:.3f})  "
                  f"cfg#{result['config_id']}  {result['time_sec']}s", flush=True)
            rows.append(result)
        # checkpoint the raw new rows after every dataset
        _write(os.path.join(RESULTS_DIR, "pyjedai_new_bestconfig_eval_raw.csv"), rows)
    merge_with_existing([r for r in rows if r.get("status") == "OK"])
    print("\nDone.")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        path = os.path.join(RESULTS_DIR, "pyjedai_new_bestconfig_eval_raw.csv")
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r.get("status") == "OK"]
        merge_with_existing(rows)
    elif len(sys.argv) > 1:
        ds = sys.argv[1]
        if ds not in DATASETS:
            sys.exit(f"Unknown dataset '{ds}'. Choose from: {', '.join(ALL_DATASETS)}")
        run_single(ds)
    else:
        run_all()


if __name__ == "__main__":
    main()
