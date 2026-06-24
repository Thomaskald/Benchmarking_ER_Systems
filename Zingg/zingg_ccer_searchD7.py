import os
import sys
import json
import time
import random
import subprocess

B = 50
SEED = 42
DATASET = "D7"

TIME_CAP_SEC = 90 * 60
MEM_CAP_GB   = 12

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "zingg_ccer_workerD7.py")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH   = os.path.join(OUT_DIR, f"zingg_{DATASET}_configs.csv")
CURVE_PATH = os.path.join(OUT_DIR, f"zingg_{DATASET}_curves.json")

SPACE = {
    "numPartitions":       ("intuniform", 8, 64),
    "labelDataSampleSize": ("uniform", 0.1, 1.0),
    "neg_ratio":           ("uniform", 1.0, 10.0),
}


def sample_config(rng):
    cfg = {}
    for name, spec in SPACE.items():
        if spec[0] == "uniform":
            cfg[name] = round(rng.uniform(spec[1], spec[2]), 4)
        elif spec[0] == "intuniform":
            cfg[name] = rng.randint(spec[1], spec[2])
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
    cols = ["config_id", "status", "numPartitions", "labelDataSampleSize", "neg_ratio",
            "chosen_threshold", "valid_f1", "test_precision", "test_recall", "test_f1",
            "time_sec", "peak_mem_mb"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            p = r.get("params", {}); tp = r.get("test_point", {}) or {}
            w.writerow({
                "config_id": r.get("config_id"), "status": r.get("status"),
                "numPartitions": p.get("numPartitions"),
                "labelDataSampleSize": p.get("labelDataSampleSize"),
                "neg_ratio": p.get("neg_ratio"),
                "chosen_threshold": r.get("chosen_threshold"),
                "valid_f1": r.get("valid_f1_at_threshold"),
                "test_precision": tp.get("precision"), "test_recall": tp.get("recall"),
                "test_f1": tp.get("f1"),
                "time_sec": r.get("time_sec"), "peak_mem_mb": r.get("peak_mem_mb"),
            })


def main():
    rng = random.Random(SEED)
    print(f"Zingg CCER random search | dataset={DATASET} | B={B} | "
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
                  f"time={res['time_sec']}s  mem={res['peak_mem_mb']}MB", flush=True)
            curves[str(i)] = res.get("pr_curve", [])
        else:
            print(f"   -> {status}  time={res.get('time_sec')}s", flush=True)
            if status == "ERROR":
                print("      stderr:", res.get("stderr_tail", "")[-400:], flush=True)
        rows.append(res)
        write_csv(rows)
        with open(CURVE_PATH, "w") as f:
            json.dump(curves, f)
    print(f"\nDone. Wrote {CSV_PATH} and {CURVE_PATH}")


if __name__ == "__main__":
    main()