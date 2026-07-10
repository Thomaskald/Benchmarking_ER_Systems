#!/usr/bin/env python3
"""
Best-config eval for D7: pairwise + cluster-level B-cubed.
Retrains the max-test_f1 config in results/linktransformer_D7_configs.csv.
VERIFY gt/gt_sep/gt_header in CFG -- B-cubed needs a correct ground-truth file.
"""
import os
import sys
import csv
import json
import time
import warnings
import logging
from collections import Counter

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DS = "D7"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PAIRS_DIR = os.path.join(RESULTS_DIR, "pairs")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "linktransformer_bestconfig_evalD7.csv")

# Cluster level: top-TOP_K right matches per left with cosine >= threshold.
TOP_K = 1
ENCODE_BATCH = 256
SEED = 42  # matches the workers' seed=42 for reproducible retraining

# `text`: (column, normalizer) list joined with " ", replicating the worker.
CFG = dict(
    sep='|',
    col_map={'https://www.scads.de/movieBenchmark/ontology/title': 'title', 'https://www.scads.de/movieBenchmark/ontology/name': 'name', 'https://www.scads.de/movieBenchmark/ontology/genre_list': 'genre_list', 'http://dbpedia.org/ontology/episodeNumber': 'episodeNumber', 'http://dbpedia.org/ontology/seasonNumber': 'seasonNumber'},
    d1='/home/it2022025/er_scalability/datasets/D7/tmdb.csv',
    d2='/home/it2022025/er_scalability/datasets/D7/tvdb.csv',
    text=[('title', 'text'), ('name', 'text'), ('episodeNumber', 'numstr'), ('seasonNumber', 'numstr')],
    train='/home/it2022025/er_scalability/train_validation_test_sets/db7/train_set.csv',
    valid='/home/it2022025/er_scalability/train_validation_test_sets/db7/valid_set.csv',
    test='/home/it2022025/er_scalability/train_validation_test_sets/db7/test_set.csv',
    gt='/home/it2022025/er_scalability/datasets/D7/gt.csv',
    gt_sep='|',
    gt_header=0,
)


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


def check_gt_connects(n_gt_total, n_gt_connected):
    """Guard: refuse bogus B-cubed if GT ids don't line up with the universe."""
    if n_gt_total == 0:
        raise RuntimeError(f"{DS}: ground-truth file loaded 0 pairs -- check gt path/format.")
    frac = n_gt_connected / n_gt_total
    if frac < 0.5:
        raise RuntimeError(
            f"{DS}: only {n_gt_connected}/{n_gt_total} GT pairs match dataset ids "
            f"({frac:.1%}). GT ids don't line up -- check gt_sep/gt_header/column order.")
    sys.stderr.write(f"[{DS}] GT: {n_gt_connected}/{n_gt_total} pairs connect "
                     f"({frac:.1%}).\n")


def load_gt_pairs(path, sep, header):
    """Ground-truth match pairs as (left, right) string tuples; first two columns."""
    import pandas as pd
    df = pd.read_csv(path, sep=sep, header=header, engine="python", dtype=str).fillna("")
    if df.shape[1] < 2:
        raise RuntimeError(
            f"GT file {path} parsed into {df.shape[1]} column with sep={sep!r} "
            f"(first row: {df.iloc[0, 0]!r}). Wrong delimiter -- fix gt_sep.")
    lc, rc = df.columns[0], df.columns[1]
    return [(str(a), str(b)) for a, b in zip(df[lc], df[rc])]


def _normalize(series, kind):
    import pandas as pd
    if kind == "text":
        return series.str.lower().str.strip().fillna("")
    if kind == "str":
        return series.astype(str).str.lower().str.strip().fillna("")
    if kind == "numstr":
        return (pd.to_numeric(series, errors="coerce").fillna("").astype(str)
                .str.replace(r"\.0$", "", regex=True))
    raise ValueError(f"unknown normalizer {kind!r}")


def load_table_with_text(path, sep, col_map, text_spec):
    """Read one base table, normalize, build '_text' = ' '.join(text cols)."""
    import pandas as pd
    df = pd.read_csv(path, delimiter=sep, engine="python")
    if col_map:
        df = df.rename(columns=col_map)
    parts = None
    for col, kind in text_spec:
        norm = _normalize(df[col], kind)
        parts = norm if parts is None else (parts + " " + norm)
    df["_text"] = parts
    df["id"] = df["id"].astype(str)
    return df[["id", "_text"]]


def build_paired(split_path, text1, text2):
    """left_text/right_text/label frame for lt.train_model / lt.evaluate_pairs."""
    import pandas as pd
    df = pd.read_csv(split_path)
    df["left_text"] = df["left_id"].astype(str).map(text1).fillna("")
    df["right_text"] = df["right_id"].astype(str).map(text2).fillna("")
    return df[["left_text", "right_text", "label"]]


def read_best_config():
    """(params, config_id, csv_test_f1) for the max-test_f1 OK row of D7."""
    path = os.path.join(RESULTS_DIR, f"linktransformer_{DS}_configs.csv")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("status") == "OK" and r.get("test_f1") not in (None, "")]
    if not rows:
        raise RuntimeError(f"{DS}: no OK rows in {path}")
    best = max(rows, key=lambda r: float(r["test_f1"]))
    params = dict(
        base_model=best["base_model"], loss_type=best["loss_type"],
        num_epochs=int(best["num_epochs"]), train_batch_size=int(best["train_batch_size"]),
        learning_rate=float(best["learning_rate"]), warm_up_perc=float(best["warm_up_perc"]),
        chosen_threshold=float(best["chosen_threshold"]),
    )
    return params, best["config_id"], float(best["test_f1"])


def eval_dataset(params):
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score
    import linktransformer as lt
    from sentence_transformers import SentenceTransformer, util

    import random as _r
    _r.seed(SEED); np.random.seed(SEED)
    try:
        import torch; torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    except Exception:
        pass

    d1 = load_table_with_text(CFG["d1"], CFG["sep"], CFG["col_map"], CFG["text"])
    d2 = load_table_with_text(CFG["d2"], CFG["sep"], CFG["col_map"], CFG["text"])
    text1 = dict(zip(d1["id"], d1["_text"]))
    text2 = dict(zip(d2["id"], d2["_text"]))

    train_paired = build_paired(CFG["train"], text1, text2)
    valid_paired = build_paired(CFG["valid"], text1, text2)
    test_paired = build_paired(CFG["test"], text1, text2)

    saved_model_path = lt.train_model(
        model_path=params["base_model"],
        train_data=train_paired, val_data=valid_paired, test_data=test_paired,
        left_col_names=["left_text"], right_col_names=["right_text"],
        label_col_name="label",
        training_args={
            "num_epochs": params["num_epochs"],
            "train_batch_size": params["train_batch_size"],
            "loss_type": params["loss_type"],
            "lr": params["learning_rate"],
            "warmup_ratio": params["warm_up_perc"],
        },
        log_wandb=False,
    )
    t = params["chosen_threshold"]

    # PAIRWISE (should reproduce CSV test_f1)
    test_ft = lt.evaluate_pairs(test_paired.copy(), model=saved_model_path,
                                left_on=["left_text"], right_on=["right_text"])
    y_test = test_paired["label"].values
    preds = (test_ft["score"].values >= t).astype(int)
    pw = (precision_score(y_test, preds, zero_division=0),
          recall_score(y_test, preds, zero_division=0),
          f1_score(y_test, preds, zero_division=0))

    # CLUSTER LEVEL: full-table NN matching with the fine-tuned model
    model = SentenceTransformer(saved_model_path)
    d1_ids, d2_ids = d1["id"].tolist(), d2["id"].tolist()
    emb1 = model.encode(d1["_text"].tolist(), batch_size=ENCODE_BATCH,
                        normalize_embeddings=True, convert_to_tensor=True,
                        show_progress_bar=False)
    emb2 = model.encode(d2["_text"].tolist(), batch_size=ENCODE_BATCH,
                        normalize_embeddings=True, convert_to_tensor=True,
                        show_progress_bar=False)
    hits = util.semantic_search(emb1, emb2, top_k=TOP_K)

    predicted_pairs = set()
    pred_pairs_tagged = []
    for i, hit_list in enumerate(hits):
        for h in hit_list:
            if h["score"] >= t:
                a, b = d1_ids[i], d2_ids[h["corpus_id"]]
                predicted_pairs.add((a, b))
                pred_pairs_tagged.append((f"A:{a}", f"B:{b}"))

    universe = [f"A:{x}" for x in d1_ids] + [f"B:{x}" for x in d2_ids]
    universe_set = set(universe)

    gt_pairs_raw = load_gt_pairs(CFG["gt"], sep=CFG["gt_sep"], header=CFG["gt_header"])
    gt_pairs_tagged = [(f"A:{a}", f"B:{b}") for a, b in gt_pairs_raw]
    n_gt_connected = sum(1 for a, b in gt_pairs_tagged
                         if a in universe_set and b in universe_set)
    check_gt_connects(len(gt_pairs_tagged), n_gt_connected)

    entity_to_pred = connected_components(pred_pairs_tagged, universe)
    entity_to_gt = connected_components(gt_pairs_tagged, universe)
    bc = bcubed(entity_to_pred, entity_to_gt)

    return dict(pairwise=pw, bcubed=bc, n_entities=len(universe),
                n_pred_pairs=len(predicted_pairs), n_gt_pairs=len(gt_pairs_tagged),
                dump_pairs=sorted(predicted_pairs), entities=universe)


SUMMARY_COLS = ["dataset", "config_id", "csv_test_f1",
                "pairwise_precision", "pairwise_recall", "pairwise_f1",
                "bcubed_precision", "bcubed_recall", "bcubed_f1",
                "n_entities", "n_pred_pairs", "n_gt_pairs", "time_sec"]


def main():
    params, config_id, csv_test_f1 = read_best_config()
    t0 = time.time()
    out = eval_dataset(params)

    os.makedirs(PAIRS_DIR, exist_ok=True)
    with open(os.path.join(PAIRS_DIR, f"linktransformer_{DS}_pred_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["left_id", "right_id"]); w.writerows(out["dump_pairs"])
    with open(os.path.join(PAIRS_DIR, f"linktransformer_{DS}_entities.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["entity_id"]); w.writerows([[e] for e in out["entities"]])

    pw, bc = out["pairwise"], out["bcubed"]
    result = {
        "dataset": DS, "config_id": config_id, "csv_test_f1": round(csv_test_f1, 6),
        "pairwise_precision": round(pw[0], 6), "pairwise_recall": round(pw[1], 6),
        "pairwise_f1": round(pw[2], 6),
        "bcubed_precision": round(bc[0], 6), "bcubed_recall": round(bc[1], 6),
        "bcubed_f1": round(bc[2], 6),
        "n_entities": out["n_entities"], "n_pred_pairs": out["n_pred_pairs"],
        "n_gt_pairs": out["n_gt_pairs"], "time_sec": round(time.time() - t0, 2),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS); w.writeheader()
        w.writerow({c: result.get(c) for c in SUMMARY_COLS})

    print("RESULT_JSON:" + json.dumps(result))
    print(f"[{DS}] pairwise F1={result['pairwise_f1']:.4f} (csv {result['csv_test_f1']:.4f})  "
          f"B3 F1={result['bcubed_f1']:.4f} "
          f"(P={result['bcubed_precision']:.3f} R={result['bcubed_recall']:.3f})  "
          f"cfg#{config_id}  {result['time_sec']}s")


if __name__ == "__main__":
    main()
