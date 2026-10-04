import os
import sys
import json
import time
import random
import subprocess

B = 50
SEED = 42
DATASET = "D2"

TIME_CAP_SEC = 90 * 60
MEM_CAP_GB   = 12

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "deeper_ccer_workerD2.py")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH   = os.path.join(OUT_DIR, f"deeper_{DATASET}_configs.csv")
CURVE_PATH = os.path.join(OUT_DIR, f"deeper_{DATASET}_curves.json")

# average / idf_average reproduce the simple compositions ("DeepER-Lite");
# lstm is the bi-directional RNN composition from the DeepER paper. Every
# composition now runs PER ATTRIBUTE.
COMPOSITIONS  = ["average", "idf_average", "lstm"]
# "similarity" is the paper's per-attribute similarity vector (2 scalars per
# attribute); the other three keep the dense element-wise features.
PAIR_FEATURES = ["similarity", "abs_diff", "hadamard", "both"]
HIDDEN        = [64, 128, 256, 512]
LSTM_HIDDEN   = [32, 64, 128]
# epochs reaches the authors' published MAX_EPOCHS=32. Their BATCH_SIZE=16 is
# deliberately NOT in the list: small batches add pure Python overhead per step
# without changing the work done, and the earlier DeepER search dropped 32 for
# exactly that reason. batch_size is still sampled at random per config -- only
# the range it is sampled from is bounded, which is what every driver in this
# study does (RecordLinkage and Magellan cap n_estimators at 500, max_depth
# at 20).
BATCH         = [64, 128, 256]
EPOCHS_RANGE  = (5, 32)
# max_len is PER ATTRIBUTE now, not per record, so the useful range is much
# shorter -- most attributes (name, brand, modelno, year) are a handful of
# tokens and only description / abstract run long.
MAX_LEN       = [8, 16, 24]


def sample_config(rng):
    cfg = {
        "composition": rng.choice(COMPOSITIONS),
        "pair_feature": rng.choice(PAIR_FEATURES),
        "hidden_dim": rng.choice(HIDDEN),
        "lr": round(10 ** rng.uniform(-4, -2), 6),
        "epochs": rng.randint(*EPOCHS_RANGE),
        "dropout": round(rng.uniform(0.0, 0.5), 3),
        "batch_size": rng.choice(BATCH),
        "max_len": rng.choice(MAX_LEN),
        "tune_embeddings": rng.choice([True, False]),
    }
    if cfg["composition"] == "lstm":
        cfg["lstm_hidden"] = rng.choice(LSTM_HIDDEN)
    return cfg


def run_one(cfg, config_id):
    cfg = dict(cfg); cfg["config_id"] = config_id; cfg["seed"] = SEED
    payload = json.dumps(cfg)
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, WORKER, payload],
                              capture_output=True, text=True, timeout=TIME_CAP_SEC)
    except subprocess.TimeoutExpired:
        return {"config_id": config_id, "params": cfg, "status": "TIMEOUT",
                "time_sec": round(time.time() - t0, 2)}
    elapsed = round(time.time() - t0, 2)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    err_tail = (proc.stderr or "")[-600:]
    status = "OOM" if ("MemoryError" in proc.stderr or proc.returncode == 137) else "ERROR"
    return {"config_id": config_id, "params": cfg, "status": status,
            "time_sec": elapsed, "stderr_tail": err_tail}


def write_csv(rows):
    import csv
    cols = ["config_id", "status", "composition", "pair_feature", "hidden_dim",
            "lstm_hidden", "lr", "epochs", "dropout", "batch_size", "max_len",
            "tune_embeddings", "chosen_threshold", "valid_f1", "test_precision",
            "test_recall", "test_f1", "n_attrs", "vocab_size",
            "tokens_missing_from_cache", "time_sec", "peak_mem_mb"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            p = r.get("params", {})
            tp = r.get("test_point", {}) or {}
            w.writerow({
                "config_id": r.get("config_id"), "status": r.get("status"),
                "composition": p.get("composition"),
                "pair_feature": p.get("pair_feature"),
                "hidden_dim": p.get("hidden_dim"), "lstm_hidden": p.get("lstm_hidden"),
                "lr": p.get("lr"), "epochs": p.get("epochs"),
                "dropout": p.get("dropout"), "batch_size": p.get("batch_size"),
                "max_len": p.get("max_len"),
                "tune_embeddings": p.get("tune_embeddings"),
                "chosen_threshold": r.get("chosen_threshold"),
                "valid_f1": r.get("valid_f1_at_threshold"),
                "test_precision": tp.get("precision"), "test_recall": tp.get("recall"),
                "test_f1": tp.get("f1"),
                "n_attrs": r.get("n_attrs"), "vocab_size": r.get("vocab_size"),
                "tokens_missing_from_cache": r.get("tokens_missing_from_cache"),
                "time_sec": r.get("time_sec"), "peak_mem_mb": r.get("peak_mem_mb"),
            })


def main():
    rng = random.Random(SEED)
    print(f"DeepER (fastText, per-attribute) CCER random search | "
          f"dataset={DATASET} | B={B} | "
          f"caps: {TIME_CAP_SEC/3600:.2f}h / {MEM_CAP_GB}GB per config")
    rows, curves = [], {}
    for i in range(1, B + 1):
        cfg = sample_config(rng)
        print(f"\n[{i}/{B}] config: {cfg}", flush=True)
        res = run_one(cfg, i)
        status = res.get("status")
        if status == "OK":
            tp = res["test_point"]
            print(f"   -> OK  testF1={tp['f1']:.4f} (P={tp['precision']:.3f} "
                  f"R={tp['recall']:.3f})  thr={res['chosen_threshold']} "
                  f"(valF1={res['valid_f1_at_threshold']:.3f})  "
                  f"attrs={res.get('n_attrs')}  "
                  f"time={res['time_sec']}s  mem={res['peak_mem_mb']}MB", flush=True)
            if res.get("tokens_missing_from_cache"):
                print(f"      WARNING: {res['tokens_missing_from_cache']} tokens "
                      f"absent from the fastText cache -- rebuild it", flush=True)
            curves[str(i)] = res.get("pr_curve", [])
        else:
            print(f"   -> {status}  time={res.get('time_sec')}s", flush=True)
            if status == "ERROR":
                print("      stderr:", res.get("stderr_tail", "")[-300:], flush=True)
        rows.append(res)
        write_csv(rows)
        with open(CURVE_PATH, "w") as f:
            json.dump(curves, f)
    print(f"\nDone. Wrote {CSV_PATH} and {CURVE_PATH}")


if __name__ == "__main__":
    main()
