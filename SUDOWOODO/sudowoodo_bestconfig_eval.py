"""Re-run Sudowoodo's best configuration per dataset and compute B-Cubed.

The search CSVs carry PAIRWISE metrics only. B-Cubed is a cluster-level
metric -- predicted pairs -> connected components -> predicted entities,
compared against ground-truth clusters over the test-set universe -- so it
needs entity ids the search never records. This second stage supplies them,
exactly as every other framework in this study does.

`connected_components`, `norm_id`, `bcubed` and `testset_metrics` below are
VERBATIM from PYJEDAI/pyjedai_new_bestconfig_eval.py, so Sudowoodo's B-Cubed
is computed by identical code to pyJedAI's, DeepER's and Dedupe's.

Best config = max test_f1 among status == OK, matching DEEPER and PYJEDAI
(`best = max(rows, key=lambda r: float(r["test_f1"]))`). Note this selects on
the test set; it is the study's frozen convention and applies to every
framework equally.

HOW PREDICTIONS ARE JOINED TO ENTITY IDS
  sudowoodo_prepare_data.py writes data/em/<TASK>/test.txt one line per row of
  <split>/test_set.csv, IN ORDER. So row i of the serialised file is row i of
  the split CSV, and the model's probability for row i belongs to that CSV
  row's (left_id, right_id). The join is positional and is asserted below.

ONE SCRIPT, ONE JOB PER DATASET
  7 of the 8 existing frameworks use a single all-datasets evaluator, and the
  logic here is single-source, so this stays one file. But the JOBS are split
  per dataset, for the reason README.md:134 gives for LinkTransformer --
  "because each configuration takes hours". Retraining all six winners in one
  process is 4.47h of measured compute against a 5h walltime, and CDDB is
  last in line. The per-dataset outputs are then merged, the same layout
  DEDUPE already uses (dedupe_bestconfig_eval_<DS>.csv + a combined file).

Usage
  python3 sudowoodo_bestconfig_eval.py D4       # one dataset -> per-dataset CSV
  python3 sudowoodo_bestconfig_eval.py --merge  # combine into the summary CSV
  python3 sudowoodo_bestconfig_eval.py          # all in one process (needs ~4.5h)

Writes
  results/sudowoodo_bestconfig_eval_<DS>.csv   one per dataset
  results/sudowoodo_bestconfig_eval.csv        the merged summary
"""

import sys
import os
import csv
import json
import time
import random
import resource
import warnings
import logging
import contextlib
from collections import Counter

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy as np

SUDO_ROOT = os.environ.get("SUDOWOODO_ROOT",
                           os.path.expanduser("~/er_scalability/sudowoodo"))
SRC = os.path.join(SUDO_ROOT, "src")
sys.path.insert(0, SRC)

# torch, transformers, pandas and Sudowoodo's own modules are imported lazily
# inside evaluate_one(). `--merge` only reads CSVs that the per-dataset jobs
# already wrote, so it must work without the training stack on the path --
# otherwise a merge run outside the venv dies on `import mlflow` (which is a
# shim on PYTHONPATH, not a real package) rather than doing its job.

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "sudowoodo_bestconfig_eval.csv")


def per_dataset_csv(ds):
    return os.path.join(RESULTS_DIR, "sudowoodo_bestconfig_eval_%s.csv" % ds)
BASE = "/home/it2022025/er_scalability"
SPLIT_ROOT = os.path.join(BASE, "train_validation_test_sets")

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)
SEED = 42

# family + the split directory holding left_id/right_id/label.
DATASETS = {
    "D2": dict(family="ccer", split="db2"),
    "D3": dict(family="ccer", split="db3"),
    "D4": dict(family="ccer", split="db4"),
    "D5": dict(family="ccer", split="db5"),
    "D6": dict(family="ccer", split="db6"),
    "D7": dict(family="ccer", split="db7"),
    "D8": dict(family="ccer", split="db8"),
    "D9": dict(family="ccer", split="db9"),
    "CORA": dict(family="der", split="cora"),
    "CDDB": dict(family="der", split="cddb"),
}

SUMMARY_COLS = ["dataset", "config_id", "family", "csv_test_f1",
                "pairwise_precision", "pairwise_recall", "pairwise_f1",
                "bcubed_precision", "bcubed_recall", "bcubed_f1",
                "n_entities", "n_pred_pairs", "n_gt_pairs", "time_sec", "status"]


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# Metric helpers -- VERBATIM from PYJEDAI/pyjedai_new_bestconfig_eval.py
# ---------------------------------------------------------------------------
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
    """Pairwise + test-set B-cubed over the fixed test pairs (uniform with DEDUPE)."""
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


# ---------------------------------------------------------------------------
# Sudowoodo retrain -- identical to the worker's loop
# ---------------------------------------------------------------------------
def make_hp(task, p):
    import argparse
    return argparse.Namespace(
        task=task, task_type="em", logdir="eval_logs/", run_id=SEED,
        batch_size=int(p["batch_size"]), max_len=int(p["max_len"]),
        size=None, lr=float(p["lr"]), n_epochs=int(p["n_epochs"]),
        lm=p["lm"], fp16=False, ssl_method=p["ssl_method"],
        n_ssl_epochs=int(p["n_ssl_epochs"]), da=p["da"], cutoff_ratio=0.05,
        clustering=False, num_clusters=90,
        zero=False, bootstrap=False, multiplier=8,
        projector=str(p["projector"]), scale_loss=1.0 / 256, lambd=3.9e-3,
        alpha_bt=0.001, save_ckpt=False, use_saved_ckpt=False,
        blocking=False, threshold=None, k=10, tfidf=False, lm_only=False,
        mlflow_tag=None)


def collect_probs(model, iterator):
    import torch
    probs, ys = [], []
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            x1, x2, x12, y = batch
            probs += model(2, x1, x2, x12).softmax(dim=1)[:, 1].cpu().numpy().tolist()
            ys += y.cpu().numpy().tolist()
    model.train()
    return np.asarray(probs), np.asarray(ys, dtype=int)


def sweep(probs, y):
    from sklearn.metrics import f1_score
    best_f1, best_t = -1.0, 0.5
    for t in THRESHOLD_GRID:
        f1 = f1_score(y, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_f1, best_t


def evaluate_one(ds, params, config_id, csv_test_f1):
    import pandas as pd
    import torch
    from torch.utils import data
    from transformers import AdamW, get_linear_schedule_with_warmup
    from selfsl.barlow_twins_simclr import (BarlowTwinsSimCLR, selfsl_step,
                                            fine_tune_step)
    from selfsl.bt_dataset import BTDataset
    from selfsl.dataset import DMDataset

    spec = DATASETS[ds]
    t0 = time.time()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    hp = make_hp(ds, params)
    prev = os.getcwd()
    os.chdir(SRC)                       # Sudowoodo resolves data/ from cwd
    try:
        path = os.path.join("data", "em", ds)
        trainset = DMDataset(os.path.join(path, "train.txt"), lm=hp.lm,
                             size=None, max_len=hp.max_len, da=None)
        validset = DMDataset(os.path.join(path, "valid.txt"), lm=hp.lm,
                             size=None, max_len=hp.max_len)
        testset = DMDataset(os.path.join(path, "test.txt"), lm=hp.lm,
                            size=None, max_len=hp.max_len)
        nolabel = BTDataset(os.path.join(path, "train_no_label.txt"), lm=hp.lm,
                            size=10000, max_len=hp.max_len, da=hp.da)

        ft_batch = max(hp.batch_size // 2, 1)
        ssl_iter = data.DataLoader(dataset=nolabel, batch_size=hp.batch_size,
                                   shuffle=True, num_workers=0, collate_fn=nolabel.pad)
        train_iter = data.DataLoader(dataset=trainset, batch_size=ft_batch,
                                     shuffle=True, num_workers=0, collate_fn=trainset.pad)
        valid_iter = data.DataLoader(dataset=validset, batch_size=hp.batch_size,
                                     shuffle=False, num_workers=0, collate_fn=validset.pad)
        test_iter = data.DataLoader(dataset=testset, batch_size=hp.batch_size,
                                    shuffle=False, num_workers=0, collate_fn=testset.pad)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm).to(device)
        optimizer = AdamW(model.parameters(), lr=hp.lr)
        n_ssl = hp.n_ssl_epochs
        n_ssl_steps = len(nolabel) // hp.batch_size * n_ssl
        n_ft_steps = len(trainset) // ft_batch * max(hp.n_epochs - n_ssl, 0)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0,
            num_training_steps=max(n_ssl_steps + n_ft_steps, 1))

        best_valid_f1, best_t, best_probs = -1.0, 0.5, None
        for epoch in range(1, hp.n_epochs + 1):
            model.train()
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                if epoch <= n_ssl:
                    selfsl_step(ssl_iter, train_iter, model, optimizer, scheduler, hp)
                else:
                    fine_tune_step(train_iter, model, optimizer, scheduler, hp)
            if epoch > n_ssl:
                pv, yv = collect_probs(model, valid_iter)
                f1v, tv = sweep(pv, yv)
                if f1v > best_valid_f1:
                    best_valid_f1, best_t = f1v, tv
                    best_probs, _ = collect_probs(model, test_iter)
        if best_probs is None:
            raise RuntimeError("no fine-tuning epoch ran")
        n_test_rows = len(testset)
    finally:
        os.chdir(prev)

    # Positional join back to the split CSV -- see the module docstring.
    test_df = pd.read_csv(os.path.join(SPLIT_ROOT, spec["split"], "test_set.csv"))
    if len(test_df) != n_test_rows or len(best_probs) != n_test_rows:
        raise RuntimeError("row-count mismatch: split_csv=%d test.txt=%d probs=%d"
                           % (len(test_df), n_test_rows, len(best_probs)))

    predicted = set()
    for i, (_, row) in enumerate(test_df.iterrows()):
        if best_probs[i] >= best_t:
            predicted.add((norm_id(row["left_id"]), norm_id(row["right_id"])))

    if spec["family"] == "ccer":
        tag_l, tag_r, symmetric = "A:", "B:", False
    else:
        tag_l, tag_r, symmetric = "", "", True

    out = testset_metrics(test_df, lambda na, nb: (na, nb) in predicted,
                          tag_l, tag_r, symmetric)
    pw, bc = out["pairwise"], out["bcubed"]
    return {
        "dataset": ds, "config_id": config_id, "family": spec["family"],
        "csv_test_f1": round(float(csv_test_f1), 6),
        "pairwise_precision": round(pw[0], 6), "pairwise_recall": round(pw[1], 6),
        "pairwise_f1": round(pw[2], 6),
        "bcubed_precision": round(bc[0], 6), "bcubed_recall": round(bc[1], 6),
        "bcubed_f1": round(bc[2], 6),
        "n_entities": out["n_entities"], "n_pred_pairs": out["n_pred_pairs"],
        "n_gt_pairs": out["n_gt_pairs"],
        "chosen_threshold": round(best_t, 3),
        "time_sec": round(time.time() - t0, 2),
        "peak_mem_mb": round(peak_mem_mb(), 1),
        "status": "OK",
    }


def read_best_config(ds):
    """(params, config_id, csv_test_f1) for the max-test_f1 OK row, or None."""
    path = os.path.join(RESULTS_DIR, "sudowoodo_%s_configs.csv" % ds)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r["status"] == "OK" and r["test_f1"]]
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r["test_f1"]))
    params = {k: best[k] for k in ("lm", "ssl_method", "n_ssl_epochs", "n_epochs",
                                   "max_len", "batch_size", "lr", "da", "projector")}
    return params, int(best["config_id"]), float(best["test_f1"])


def write_summary(rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in SUMMARY_COLS})


def merge():
    """Combine the per-dataset CSVs into the summary, in canonical order."""
    rows = []
    for ds in DATASETS:
        p = per_dataset_csv(ds)
        if os.path.exists(p):
            with open(p) as f:
                rows += list(csv.DictReader(f))
        else:
            # visible in the summary rather than silently absent
            status = "NO_OK_CONFIG" if read_best_config(ds) is None else "NOT_RUN"
            rows.append({"dataset": ds, "family": DATASETS[ds]["family"],
                         "status": status})
    write_summary(rows)
    print("merged %d rows -> %s" % (len(rows), SUMMARY_CSV))
    for r in rows:
        print("  %-6s %-14s pairwise_f1=%-10s bcubed_f1=%s"
              % (r.get("dataset"), r.get("status"),
                 r.get("pairwise_f1") or "-", r.get("bcubed_f1") or "-"))


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--merge":
        merge()
        return

    if len(sys.argv) == 2:
        ds = sys.argv[1]
        if ds not in DATASETS:
            raise SystemExit("unknown dataset %s; known: %s" % (ds, list(DATASETS)))
        got = read_best_config(ds)
        if got is None:
            raise SystemExit("no OK config for %s -- every config TIMEOUT" % ds)
        params, cid, f1 = got
        r = evaluate_one(ds, params, cid, f1)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(per_dataset_csv(ds), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
            w.writeheader()
            w.writerow({c: r.get(c) for c in SUMMARY_COLS})
        print("pairwise_f1=%.4f (search %.4f, delta %+.6f)  bcubed_f1=%.4f  "
              "n_entities=%d  %.0fs"
              % (r["pairwise_f1"], r["csv_test_f1"],
                 r["pairwise_f1"] - r["csv_test_f1"], r["bcubed_f1"],
                 r["n_entities"], r["time_sec"]))
        print("RESULT_JSON:" + json.dumps(r))
        return

    rows = []
    for ds in DATASETS:
        got = read_best_config(ds)
        if got is None:
            print("%-6s no OK config -- skipped (all configs TIMEOUT)" % ds, flush=True)
            rows.append({"dataset": ds, "family": DATASETS[ds]["family"],
                         "status": "NO_OK_CONFIG"})
            write_summary(rows)
            continue
        params, cid, f1 = got
        print("%-6s retraining config %d (search test_f1=%.4f) ..." % (ds, cid, f1),
              flush=True)
        try:
            r = evaluate_one(ds, params, cid, f1)
            print("       pairwise_f1=%.4f (search %.4f, delta %+.6f)  "
                  "bcubed_f1=%.4f  n_entities=%d  %.0fs"
                  % (r["pairwise_f1"], r["csv_test_f1"],
                     r["pairwise_f1"] - r["csv_test_f1"], r["bcubed_f1"],
                     r["n_entities"], r["time_sec"]), flush=True)
        except Exception as e:
            print("       ERROR: %s" % e, flush=True)
            r = {"dataset": ds, "family": DATASETS[ds]["family"],
                 "config_id": cid, "csv_test_f1": round(f1, 6),
                 "status": "ERROR"}
        rows.append(r)
        write_summary(rows)
    print("\nWrote %s" % SUMMARY_CSV)


if __name__ == "__main__":
    main()
