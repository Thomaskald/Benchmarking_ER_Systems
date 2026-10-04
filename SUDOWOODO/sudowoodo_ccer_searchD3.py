"""Random search over Sudowoodo configurations on D3.

    python sudowoodo_ccer_searchD3.py

B = 5, not 50. This follows the precedent the study already set for
LinkTransformer (README.md:145, :509 -- "Search budgets are not equal.
LinkTransformer ran B = 5, every other framework ran B = 50. Each
LinkTransformer configuration takes hours"). Everything else is the frozen
protocol untouched: SEED=42, TIME_CAP_SEC=90*60, checkpoint after every
config, no resume, OK/TIMEOUT/OOM/ERROR all retained.

THE SEARCH SPACE IS BOUNDED BY MEASUREMENT, not by preference. Jobs 173251,
173294 and 173310 measured per-step cost on all ten datasets. What those
measurements rule out:

  * lm=roberta        -- 2.2x slower than distilbert on identical input.
  * ssl_method=combined -- encodes four times per step against simclr's two,
                        measured at almost exactly 2.0x across all ten
                        datasets. Does not fit on any dataset here.
  * max_len=128       -- only D4 and CDDB fit; 64 brings in D3, D7 and D9.
  * n_ssl_epochs > 1  -- a second SSL epoch alone costs 0.3-0.4h.

Bounding the sampled range is what every driver in this study does (the
DeepER driver caps batch_size, RecordLinkage and Magellan cap n_estimators).
The cap itself is NOT relaxed, and configurations that run over still become
TIMEOUT rows.

WHAT THIS MEASURES IS A LOWER BOUND on Sudowoodo. To fit the budget it gives
up `ssl_method=combined`, `--clustering`, multi-epoch pre-training and the
roberta encoder -- several of which are the paper's headline components.
Report it as such.
"""

import os
import sys
import csv
import json
import time
import random
import subprocess

B = 5                       # LinkTransformer precedent, README.md:509
SEED = 42
TIME_CAP_SEC = 90 * 60      # frozen, NOT relaxed
MEM_CAP_GB = 12             # declared for consistency; never enforced anywhere
                            # in this study (no setrlimit), SLURM's 64G is the
                            # real limit. Measured peak here is 13-20GB.

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "sudowoodo_ccer_workerD3.py")
OUT_DIR = os.path.join(HERE, "results")

DATASET = "D3"

# ONE search space for every dataset, exactly as the DeepER and
# LinkTransformer drivers use one space across their datasets. No per-dataset
# tuning and no pre-screening of which datasets to attempt: every dataset gets
# the same B=5 draw, and the 90-minute cap decides what completes. A dataset
# that returns 0 OK / 5 TIMEOUT is a reported result, the same way
# LinkTransformer's D6 is reported as 1 OK / 3 (README.md:526).
#
# Two global exclusions, both bounded by measurement rather than preference,
# which is what every driver in this study does:
#   lm=roberta          -- measured 2.2x slower than distilbert on identical
#                          input (job 173251). Would TIMEOUT everywhere.
#   ssl_method=combined -- encodes four times per step against simclr's two;
#                          measured at 2.0x across all ten datasets
#                          (job 173310 vs 173294).
#   --bootstrap         -- not a preference: it is undefined without a
#                          labelled/unlabelled split, which full supervision
#                          does not have. See the worker's docstring.
#   max_len=128         -- MEASURED out of budget. Jobs 173391/173392 reached
#                          the end of a single SSL epoch at 71 min on D3 and
#                          TIMEOUT on D2, against a 90-minute cap that still
#                          has to cover fine-tuning and two eval passes. It is
#                          reachable on D4 and CDDB only; keeping it would have
#                          killed 3 of 5 configs on the other eight datasets
#                          under SEED=42, which buys nothing.
SSL_METHODS  = ["simclr", "barlow_twins"]
MAX_LEN      = [32, 64]
EPOCHS_RANGE = (2, 3)
# The README advertises ['del','drop_col','append_col','swap','ins','shuffle']
# plus 'cutoff'. `shuffle` IS NOT IMPLEMENTED -- selfsl/augment.py has no
# branch for it and raises ValueError('DA operator not found') at line 185.
# Confirmed empirically (jobs 173391/173392, config 5 on both). This is the
# authors' documented list minus the operator that does not exist.
DA           = ["del", "drop_col", "append_col", "swap", "ins", "cutoff"]
PROJECTOR    = ["768", "4096"]


def sample_config(rng):
    return {
        "lm": "distilbert",
        "ssl_method": rng.choice(SSL_METHODS),
        "n_ssl_epochs": 1,
        "n_epochs": rng.randint(*EPOCHS_RANGE),   # >= n_ssl_epochs + 1
        "max_len": rng.choice(MAX_LEN),
        "batch_size": 64,
        "lr": round(10 ** rng.uniform(-5, -4), 7),
        "da": rng.choice(DA),
        "projector": rng.choice(PROJECTOR),
    }


def run_one(cfg, config_id):
    cfg = dict(cfg); cfg["config_id"] = config_id; cfg["seed"] = SEED
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, WORKER, json.dumps(cfg)],
                              capture_output=True, text=True,
                              timeout=TIME_CAP_SEC)
    except subprocess.TimeoutExpired:
        return {"config_id": config_id, "params": cfg, "status": "TIMEOUT",
                "time_sec": round(time.time() - t0, 2)}
    elapsed = round(time.time() - t0, 2)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    err_tail = (proc.stderr or "")[-600:]
    status = "OOM" if ("MemoryError" in proc.stderr or proc.returncode == 137) \
        else "ERROR"
    return {"config_id": config_id, "params": cfg, "status": status,
            "time_sec": elapsed, "stderr_tail": err_tail}


COLS = ["config_id", "status", "task", "lm", "ssl_method", "n_ssl_epochs",
        "n_epochs", "max_len", "batch_size", "lr", "da", "projector",
        "chosen_threshold", "valid_f1", "test_precision",
        "test_recall", "test_f1", "best_epoch", "n_train_pairs",
        "time_sec", "peak_mem_mb"]


def write_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            p = r.get("params", {})
            tp = r.get("test_point", {}) or {}
            row = {c: p.get(c) for c in COLS if c in p}
            row.update({
                "config_id": r.get("config_id"), "status": r.get("status"),
                "chosen_threshold": r.get("chosen_threshold"),
                "valid_f1": r.get("valid_f1_at_threshold"),
                "test_precision": tp.get("precision"),
                "test_recall": tp.get("recall"), "test_f1": tp.get("f1"),
                "best_epoch": r.get("best_epoch"),
                "n_train_pairs": r.get("n_train_pairs"),
                "time_sec": r.get("time_sec"),
                "peak_mem_mb": r.get("peak_mem_mb"),
            })
            w.writerow(row)


def main():
    task = DATASET
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "sudowoodo_%s_configs.csv" % task)
    curve_path = os.path.join(OUT_DIR, "sudowoodo_%s_curves.json" % task)

    rng = random.Random(SEED)
    print("Sudowoodo random search | dataset=%s | B=%d | "
          "caps: %.2fh / %dGB per config"
          % (task, B, TIME_CAP_SEC / 3600.0, MEM_CAP_GB), flush=True)

    rows, curves = [], {}
    for i in range(1, B + 1):
        cfg = sample_config(rng)
        print("\n[%d/%d] config: %s" % (i, B, cfg), flush=True)
        res = run_one(cfg, i)
        status = res.get("status")
        if status == "OK":
            tp = res["test_point"]
            print("   -> OK  testF1=%.4f (P=%.3f R=%.3f)  thr=%s "
                  "(valF1=%.3f)  epoch=%s  time=%ss  mem=%sMB"
                  % (tp["f1"], tp["precision"], tp["recall"],
                     res["chosen_threshold"], res["valid_f1_at_threshold"],
                     res.get("best_epoch"), res["time_sec"],
                     res["peak_mem_mb"]), flush=True)
            curves[str(i)] = res.get("pr_curve", [])
        else:
            print("   -> %s  time=%ss" % (status, res.get("time_sec")), flush=True)
            if status == "ERROR":
                print("      stderr:", res.get("stderr_tail", "")[-300:], flush=True)
        rows.append(res)
        write_csv(rows, csv_path)
        with open(curve_path, "w") as f:
            json.dump(curves, f)

    ok = sum(1 for r in rows if r.get("status") == "OK")
    print("\nDone. %d/%d OK. Wrote %s and %s" % (ok, B, csv_path, curve_path))


if __name__ == "__main__":
    main()
