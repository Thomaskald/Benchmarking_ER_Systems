"""
deeper_scalability_fixed.py
---------------------------
Fixed-config scalability runs for DeepER on synthetic FEBRL datasets.
Uses the single BEST config from the 10K search, selected by argmax(valid_f1)
-- the same rule the dedupe / splink / pyjedai / zingg fixed scripts use --
applied verbatim to every size INCLUDING a frozen decision threshold. No
per-size threshold sweep, no valid set at scale.

10K is excluded (it was the tuning size).
Sizes: 50K, 100K, 200K, 300K, 1M, 2M -- one run each, checkpointing after each.
Metrics per size: runtime, peak memory (VmHWM), test P/R/F1 over the split-file
labelled pairs.
Failure policy: OOM / TIMEOUT / ERROR are recorded, not crashes.

WHAT IS AND IS NOT MEASURED
---------------------------
DeepER is a pair classifier over the fixed split pairs. It does NOT do candidate
generation, so -- unlike Dedupe, Splink, pyJedAI and Zingg, which block over all
N records -- DeepER neither pays blocking's COST nor suffers its LOSS. Both
columns are affected and neither is comparable to the other frameworks':

  runtime : theirs includes blocking over N records, DeepER's does not.
            MEASURED: this turned out to matter less than expected -- DeepER
            still grows near-linearly (685s at 50K -> 10694s at 1M, 15.6x for
            20x the records) because the labelled pair count scales with N, so
            it does O(N) work regardless. Comparable to pyJedAI, and far better
            than Splink's superlinear 18s -> 10238s over the same span.
  F1      : theirs is capped by what blocking recalls -- pyJedAI's recall at 50K
            is 0.4499, i.e. it never gets the chance to judge over half the true
            pairs. DeepER is handed every test pair, so its F1 looks far higher.
            MEASURED: DeepER recall 0.992-0.998 against pyJedAI 0.42-0.45 and
            Splink 0.91-0.93; F1 ~0.995 flat across every size. That is NOT
            evidence that it matches better -- it is the blocking asymmetry,
            and this is the column that must carry the caveat.

This is the same treatment DeepER receives in the CCER and DER experiments,
where every framework is scored on the same test pairs so the asymmetry does not
arise. Here it does, and it must be stated wherever this curve is plotted.
What IS measured honestly here is MEMORY: every record in the table is
tokenised and embedded (as the CCER/DER workers do), so the record tensor grows
linearly with N and the size at which DeepER exhausts the cap is a real result.

`n_clusters` is left empty on purpose: DeepER performs no clustering over the
record set, so there is no count comparable to the other frameworks' column.

Output: deeper_scalability_results.csv
"""
import os
import re
import csv
import sys
import time
import json
import traceback
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTED_DIR = "/path/to/your/er_scalability/converted"
SPLITS_DIR    = "/path/to/your/er_scalability/splits"
EMB_DIR       = "/path/to/your/er_scalability/embeddings"
OUTPUT_CSV    = "/path/to/your/er_scalability/scalability/deeper_scalability_results.csv"
SEARCH_CSV    = os.path.join(HERE, "results", "deeper_10K_configs.csv")

DATASETS = ["50K", "100K", "200K", "300K", "1M", "2M"]
ID_COL = "id"

# Same explicit FEBRL attribute set as the 10K worker -- see the note there on
# why this is a fixed list rather than "every non-id column".
ATTRS = ["given_name", "surname", "address_1", "suburb", "postcode",
         "state", "date_of_birth", "soc_sec_id", "phone_number"]

EMB_DIM = 300
TOKEN_RE = re.compile(r"[a-z0-9]+")
SEED = 42

# Leave as None to read the winner straight out of the 10K search by
# argmax(valid_f1). Set explicitly only to override.
BEST_CONFIG_OVERRIDE = None
BEST_THRESHOLD_OVERRIDE = None

# --- code below is verbatim from deeper_scalability_worker10K.py ---
from deeper_scalability_worker10K import (          # noqa: E402
    peak_mem_mb, norm_id, tokenize, attr_tokens, resolve_attrs,
    load_cache, build_matrix, idf_weights, to_ids, DeepER, probs_for, _t,
)


def mem_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def read_best_config():
    """Winner of the 10K search by argmax(valid_f1) -- the scalability rule."""
    if BEST_CONFIG_OVERRIDE is not None:
        return dict(BEST_CONFIG_OVERRIDE), float(BEST_THRESHOLD_OVERRIDE)
    with open(SEARCH_CSV) as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("status") == "OK" and r.get("valid_f1") not in (None, "")]
    if not rows:
        raise RuntimeError(f"no OK rows in {SEARCH_CSV}")
    best = max(rows, key=lambda r: float(r["valid_f1"]))
    cfg = dict(
        composition=best["composition"],
        pair_feature=best["pair_feature"],
        hidden_dim=int(best["hidden_dim"]),
        lr=float(best["lr"]),
        epochs=int(best["epochs"]),
        dropout=float(best["dropout"]),
        batch_size=int(best["batch_size"]),
        max_len=int(best["max_len"]),
        tune_embeddings=str(best["tune_embeddings"]).strip().lower() == "true",
    )
    if best["composition"] == "lstm":
        cfg["lstm_hidden"] = int(float(best["lstm_hidden"]))
    return cfg, float(best["chosen_threshold"])


def run_one_size(ds, cfg, threshold):
    """Train on this size's train split, score its test pairs at the FROZEN
    threshold. Every record in the table is embedded, so memory tracks N."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    t_load = time.time()
    df = pd.read_csv(os.path.join(CONVERTED_DIR, ds, "profiles.csv"),
                     engine="python", na_filter=False)
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, ds, "train_set.csv"))
    test_df  = pd.read_csv(os.path.join(SPLITS_DIR, ds, "test_set.csv"))
    t_load = time.time() - t_load

    print(f"  Records    : {len(df):,}")
    print(f"  Test pairs : {len(test_df):,}  (pos={int(test_df['label'].sum()):,})")
    print(f"  Load time  : {t_load:.2f}s  |  mem: {mem_mb():.0f} MB", flush=True)

    t0 = time.time()
    attrs, unused = resolve_attrs(df)
    if unused:
        print(f"  columns present but NOT embedded: {unused}", flush=True)
    n_attrs = len(attrs)
    tok = attr_tokens(df, attrs)

    cache_toks, cache_vecs = load_cache(ds)
    vocab, M, missing = build_matrix(tok, cache_toks, cache_vecs)
    del cache_toks, cache_vecs          # ~4.5GB at 2M -- free before training
    if missing:
        print(f"  WARNING: {missing} tokens absent from the fastText cache", flush=True)
    idf = idf_weights(tok, vocab)

    X = to_ids(tok, vocab, int(cfg["max_len"]))
    pos = {norm_id(v): i for i, v in enumerate(df[ID_COL].tolist())}
    print(f"  Vocab      : {len(vocab):,}  |  record tensor: "
          f"{X.nbytes / 1024**2:.0f} MB  |  mem: {mem_mb():.0f} MB", flush=True)

    def rows(sp):
        a = np.array([pos[norm_id(v)] for v in sp["left_id"]], dtype=np.int64)
        b = np.array([pos[norm_id(v)] for v in sp["right_id"]], dtype=np.int64)
        return a, b, sp["label"].astype(int).to_numpy()

    a_tr, b_tr, y_tr = rows(train_df)
    a_te, b_te, y_te = rows(test_df)

    model = DeepER(M, cfg, n_attrs, idf=idf)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=float(cfg["lr"]))
    npos, nneg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(nneg / max(npos, 1), dtype=torch.float32))

    batch = int(cfg["batch_size"])
    yt = torch.from_numpy(y_tr.astype(np.float32))
    order = np.arange(len(y_tr))
    rng = np.random.RandomState(SEED)
    model.train()
    for ep in range(int(cfg["epochs"])):
        rng.shuffle(order)
        for s in range(0, len(order), batch):
            sl = order[s:s + batch]
            opt.zero_grad()
            logits = model(_t(X[a_tr[sl]]), _t(X[b_tr[sl]]))
            loss = lossf(logits, yt[sl])
            loss.backward()
            opt.step()

    probs_test = probs_for(model, X, a_te, b_te, 512)
    preds = (probs_test >= threshold).astype(int)
    t_workflow = time.time() - t0

    return dict(
        n_records=len(df), n_test_pairs=len(test_df),
        precision=precision_score(y_te, preds, zero_division=0),
        recall=recall_score(y_te, preds, zero_division=0),
        f1=f1_score(y_te, preds, zero_division=0),
        time_load=t_load, time_workflow=t_workflow,
        peak_mem_mb=peak_mem_mb(), vocab_size=len(vocab),
        tensor_mb=X.nbytes / 1024 ** 2, n_attrs=n_attrs,
    )


COLS = ["dataset", "n_records", "n_test_pairs", "n_clusters", "precision",
        "recall", "f1", "time_load", "time_workflow", "peak_mem_mb", "status"]


def main():
    cfg, threshold = read_best_config()
    print("\n" + "=" * 60)
    print("  DEEPER SCALABILITY (fixed best config)")
    print(f"  {cfg}")
    print(f"  frozen threshold = {threshold}   (selected by argmax valid_f1 at 10K)")
    print("  NOTE: no candidate generation -- runtime excludes blocking.")
    print("=" * 60, flush=True)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results = []
    for ds in DATASETS:
        profiles = os.path.join(CONVERTED_DIR, ds, "profiles.csv")
        test_p   = os.path.join(SPLITS_DIR, ds, "test_set.csv")
        print(f"\n{'='*60}\n  Dataset: {ds}\n{'='*60}", flush=True)

        if not os.path.exists(profiles) or not os.path.exists(test_p):
            print(f"  [SKIP] missing files for {ds}")
            results.append({"dataset": ds, "status": "SKIP: file not found"})
        else:
            try:
                out = run_one_size(ds, cfg, threshold)
                print(f"\n  --- RESULTS ---")
                print(f"  Precision  : {out['precision']:.4f}  Recall: {out['recall']:.4f}"
                      f"  F1: {out['f1']:.4f}")
                print(f"  Workflow   : {out['time_workflow']:.2f}s "
                      f"({out['time_workflow']/60:.1f} min)")
                print(f"  Peak mem   : {out['peak_mem_mb']:.0f} MB", flush=True)
                results.append({
                    "dataset": ds, "n_records": out["n_records"],
                    "n_test_pairs": out["n_test_pairs"], "n_clusters": "",
                    "precision": round(out["precision"], 4),
                    "recall": round(out["recall"], 4), "f1": round(out["f1"], 4),
                    "time_load": round(out["time_load"], 2),
                    "time_workflow": round(out["time_workflow"], 2),
                    "peak_mem_mb": round(out["peak_mem_mb"], 1), "status": "OK"})
            except MemoryError:
                print("  [OOM]", flush=True)
                results.append({"dataset": ds, "status": "OOM"})
            except Exception as e:
                print(f"  [ERROR] {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                results.append({"dataset": ds, "status": f"ERROR: {type(e).__name__}: {e}"})

        pd.DataFrame(results).reindex(columns=COLS).to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
