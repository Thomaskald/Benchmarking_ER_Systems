#!/usr/bin/env python3
"""
Best-config eval for D9: pairwise + test-set B-cubed. Both metrics are computed
on the fixed test pairs only -- uniform with the DEDUPE / pyJedAI / Magellan evals
so the numbers are directly comparable across frameworks.
Retrains the max-test_f1 config in results/linktransformer_D9_configs.csv.
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

DS = "D9"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PAIRS_DIR = os.path.join(RESULTS_DIR, "pairs")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "linktransformer_bestconfig_evalD9.csv")
SEED = 42  # matches the workers' seed=42 for reproducible retraining

# `text`: (column, normalizer) list joined with " ", replicating the worker.
CFG = dict(
    sep='>',
    col_map=None,
    d1='/home/it2022025/er_scalability/datasets/D9/dblp.csv',
    d2='/home/it2022025/er_scalability/datasets/D9/scholar.csv',
    text=[('title', 'text'), ('authors', 'text'), ('venue', 'text')],
    train='/home/it2022025/er_scalability/train_validation_test_sets/db9/train_set.csv',
    valid='/home/it2022025/er_scalability/train_validation_test_sets/db9/valid_set.csv',
    test='/home/it2022025/er_scalability/train_validation_test_sets/db9/test_set.csv',
)


def norm_id(v):
    """Canonical string id. '123', 123, '123.0' -> '123'. Keeps non-numeric as-is."""
    s = str(v).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


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


def testset_metrics(left_ids, right_ids, probs, labels, threshold):
    """Pairwise + test-set B-cubed over the fixed test pairs (CCER tags A:/B:):
      - predicted clusters = connected components of test pairs scored >= threshold
      - true clusters      = connected components of test pairs with label == 1
      - both metrics over the entities that appear in the test set.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    universe, pred_edges, true_edges, pred_pairs_out, y_pred = set(), [], [], [], []
    for i in range(len(labels)):
        na, nb = norm_id(left_ids[i]), norm_id(right_ids[i])
        la, rb = f"A:{na}", f"B:{nb}"
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


def build_paired(path, text1, text2):
    """Test/train/valid split as a frame with left_id/right_id + left_text/right_text/label."""
    import pandas as pd
    df = pd.read_csv(path)
    df["left_text"] = df["left_id"].astype(str).map(text1).fillna("")
    df["right_text"] = df["right_id"].astype(str).map(text2).fillna("")
    return df


def read_best_config():
    """(params, config_id, csv_test_f1) for the max-test_f1 OK row of D9."""
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
    )
    return params, best["config_id"], float(best["test_f1"])


def eval_dataset(params):
    import numpy as np
    from sklearn.metrics import f1_score
    import linktransformer as lt

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
    cols = ["left_text", "right_text", "label"]

    saved_model_path = lt.train_model(
        model_path=params["base_model"],
        train_data=train_paired[cols], val_data=valid_paired[cols], test_data=test_paired[cols],
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

    # threshold tuned on the valid split (same grid as the worker)
    valid_ft = lt.evaluate_pairs(valid_paired[cols].copy(), model=saved_model_path,
                                 left_on=["left_text"], right_on=["right_text"])
    probs_valid = valid_ft["score"].values
    y_valid = valid_paired["label"].values
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.0, 1.001, 0.01):
        f1 = f1_score(y_valid, (probs_valid >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    test_ft = lt.evaluate_pairs(test_paired[cols].copy(), model=saved_model_path,
                                left_on=["left_text"], right_on=["right_text"])
    probs_test = test_ft["score"].values

    return testset_metrics(test_paired["left_id"].values, test_paired["right_id"].values,
                           probs_test, test_paired["label"].values, best_t)


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
