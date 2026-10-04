#!/usr/bin/env python3
"""Best-config evaluation for DeepER.

Same protocol and same output columns as every other framework's evaluator.
For each dataset it reads the best config (max test_f1 among status==OK) from
results/deeper_<DS>_configs.csv, re-runs exactly that configuration with the
same seed, scores the fixed test pairs once, and reports pairwise AND test-set
B-cubed P/R/F1.

The metric helpers below (`connected_components`, `norm_id`, `bcubed`,
`testset_metrics`) are VERBATIM from PYJEDAI/pyjedai_new_bestconfig_eval.py, and
the model and feature code is verbatim from the DeepER workers. Nothing is
reimplemented, so the B-cubed numbers are directly comparable across all
frameworks.

Because the config is re-run rather than read back, `pairwise_f1` must equal the
`csv_test_f1` it came from. Any difference is a reproducibility failure and is
printed as `delta` -- that is the point of re-running rather than copying.

Usage
-----
  python3 deeper_bestconfig_eval.py D8      # one dataset -> RESULT_JSON
  python3 deeper_bestconfig_eval.py         # all 10 -> results/deeper_bestconfig_eval.csv

Output
------
  results/deeper_bestconfig_eval.csv
  results/pairs/deeper_<DS>_{pred_pairs,entities}.csv
"""
import os
import re
import sys
import csv
import json
import time
import warnings
import logging
from collections import Counter

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PAIRS_DIR = os.path.join(RESULTS_DIR, "pairs")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "deeper_bestconfig_eval.csv")

DATA_ROOT  = "/home/it2022025/er_scalability/datasets"
SPLIT_ROOT = "/home/it2022025/er_scalability/train_validation_test_sets"
EMB_DIR    = "/home/it2022025/er_scalability/embeddings"

SEED = 42
EMB_DIM = 300
THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)
TOKEN_RE = re.compile(r"[a-z0-9]+")

SCADS_RENAME = {
    "https://www.scads.de/movieBenchmark/ontology/title": "title",
    "https://www.scads.de/movieBenchmark/ontology/name": "name",
    "https://www.scads.de/movieBenchmark/ontology/genre_list": "genre_list",
    "http://dbpedia.org/ontology/abstract": "abstract",
    "http://dbpedia.org/ontology/episodeNumber": "episodeNumber",
    "http://dbpedia.org/ontology/seasonNumber": "seasonNumber",
    "http://dbpedia.org/ontology/releaseDate": "releaseDate",
}

# Mirrors the workers exactly: same paths, delimiters, rename maps and attribute
# lists, so the evaluator sees the records the search saw.
DATASETS = {
    "D2": dict(family="ccer", left=f"{DATA_ROOT}/D2/abt.csv", right=f"{DATA_ROOT}/D2/buy.csv",
               delim="|", split="db2", id_col="id", rename=None,
               attrs=["name", "description", "price"]),
    "D3": dict(family="ccer", left=f"{DATA_ROOT}/D3/amazon.csv", right=f"{DATA_ROOT}/D3/gp.csv",
               delim="#", split="db3", id_col="id", rename=None,
               attrs=["title", "description", "manufacturer", "price"]),
    "D4": dict(family="ccer", left=f"{DATA_ROOT}/D4/dblp.csv", right=f"{DATA_ROOT}/D4/acm.csv",
               delim="%", split="db4", id_col="id", rename=None,
               attrs=["title", "authors", "venue", "year"]),
    "D5": dict(family="ccer", left=f"{DATA_ROOT}/D5/imdb.csv", right=f"{DATA_ROOT}/D5/tmdb.csv",
               delim="|", split="db5", id_col="id", rename=SCADS_RENAME,
               attrs=["title", "name", "genre_list", "episodeNumber", "seasonNumber"]),
    "D6": dict(family="ccer", left=f"{DATA_ROOT}/D6/imdb.csv", right=f"{DATA_ROOT}/D6/tvdb.csv",
               delim="|", split="db6", id_col="id", rename=SCADS_RENAME,
               attrs=["title", "name", "episodeNumber", "seasonNumber"]),
    "D7": dict(family="ccer", left=f"{DATA_ROOT}/D7/tmdb.csv", right=f"{DATA_ROOT}/D7/tvdb.csv",
               delim="|", split="db7", id_col="id", rename=SCADS_RENAME,
               attrs=["title", "name", "abstract", "episodeNumber", "seasonNumber",
                      "releaseDate"]),
    "D8": dict(family="ccer", left=f"{DATA_ROOT}/D8/walmart.csv", right=f"{DATA_ROOT}/D8/amazon.csv",
               delim="|", split="db8", id_col="id", rename=None,
               attrs=["title", "brand", "modelno", "dimensions", "price", "shipweight"]),
    "D9": dict(family="ccer", left=f"{DATA_ROOT}/D9/dblp.csv", right=f"{DATA_ROOT}/D9/scholar.csv",
               delim=">", split="db9", id_col="id", rename=None,
               attrs=["title", "authors", "venue", "year"]),
    "CORA": dict(family="der", src=f"{DATA_ROOT}/cora/cora.csv", delim="|", split="cora",
                 id_col="Entity Id", drop=[]),
    "CDDB": dict(family="der", src=f"{DATA_ROOT}/CDDB/cddb.csv", delim=",", split="cddb",
                 id_col="id", drop=["tracks"]),
}
ALL_DATASETS = list(DATASETS.keys())


# ===========================================================================
# Metric helpers -- VERBATIM from PYJEDAI/pyjedai_new_bestconfig_eval.py
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


# ===========================================================================
# Model and feature code -- VERBATIM from the DeepER workers
# ===========================================================================
def tokenize(text):
    return TOKEN_RE.findall(text)


def attr_tokens(df, attrs):
    """tokens[a][i] -- the token list of attribute a for record i."""
    out = []
    for a in attrs:
        col = (df[a].fillna("").astype(str).str.lower().tolist()
               if a in df.columns else [""] * len(df))
        out.append([tokenize(t) for t in col])
    return out


def load_cache(ds):
    z = np.load(os.path.join(EMB_DIR, f"ft_cache_{ds}.npz"), allow_pickle=False)
    return {str(t): i for i, t in enumerate(z["tokens"])}, z["vectors"]


def build_matrix(token_lists, cache_idx, cache_vecs):
    vocab = {"<pad>": 0, "<unk>": 1}
    for per_attr in token_lists:
        for toks in per_attr:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
    M = np.zeros((len(vocab), EMB_DIM), dtype=np.float32)
    missing = 0
    for w, i in vocab.items():
        if i <= 1:
            continue
        j = cache_idx.get(w)
        if j is None:
            missing += 1
        else:
            M[i] = cache_vecs[j]
    return vocab, M, missing


def idf_weights(token_lists, vocab):
    n_records = len(token_lists[0]) if token_lists else 0
    df = np.zeros(len(vocab), dtype=np.float32)
    for i in range(n_records):
        seen = set()
        for per_attr in token_lists:
            seen.update(per_attr[i])
        for t in seen:
            if t in vocab:
                df[vocab[t]] += 1
    return np.log((1.0 + n_records) / (1.0 + df)) + 1.0


def to_ids(token_lists, vocab, max_len):
    """N x A x L id tensor. max_len is PER ATTRIBUTE."""
    n_attrs = len(token_lists)
    n = len(token_lists[0]) if n_attrs else 0
    X = np.zeros((n, n_attrs, max_len), dtype=np.int64)
    for a, per_attr in enumerate(token_lists):
        for i, toks in enumerate(per_attr):
            ids = [vocab.get(t, 1) for t in toks][:max_len]
            X[i, a, :len(ids)] = ids
    return X


class DeepER(nn.Module):
    def __init__(self, emb_matrix, cfg, n_attrs, idf=None):
        super().__init__()
        V, D = emb_matrix.shape
        self.emb = nn.Embedding(V, D, padding_idx=0)
        self.emb.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.emb.weight.requires_grad = bool(cfg.get("tune_embeddings", False))
        self.composition = cfg["composition"]
        self.n_attrs = n_attrs
        self.register_buffer("idf", torch.from_numpy(idf) if idf is not None
                             else torch.ones(V))

        if self.composition == "lstm":
            self.rnn = nn.LSTM(D, cfg["lstm_hidden"], batch_first=True,
                               bidirectional=True)
            rep = 2 * cfg["lstm_hidden"]
        else:
            rep = D

        pf = cfg["pair_feature"]
        self.pair_feature = pf
        if pf == "similarity":
            in_dim = 2 * n_attrs
        elif pf == "both":
            in_dim = 2 * n_attrs * rep
        else:
            in_dim = n_attrs * rep
        h = cfg["hidden_dim"]
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg["dropout"]),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(cfg["dropout"]),
            nn.Linear(h // 2, 1),
        )

    def compose(self, x):
        B, A, L = x.shape
        xf = x.reshape(B * A, L)
        e = self.emb(xf)
        mask = (xf != 0).float().unsqueeze(-1)
        if self.composition == "lstm":
            out, _ = self.rnn(e)
            out = out * mask
            v = out.sum(1) / mask.sum(1).clamp(min=1.0)
        elif self.composition == "idf_average":
            w = self.idf[xf].unsqueeze(-1) * mask
            v = (e * w).sum(1) / w.sum(1).clamp(min=1e-6)
        else:
            v = (e * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return v.reshape(B, A, -1)

    def forward(self, xa, xb):
        va, vb = self.compose(xa), self.compose(xb)
        if self.pair_feature == "similarity":
            cos = F.cosine_similarity(va, vb, dim=2, eps=1e-8)
            den = va.norm(dim=2) + vb.norm(dim=2) + 1e-8
            dist = (va - vb).norm(dim=2) / den
            z = torch.cat([cos, dist], dim=1)
        elif self.pair_feature == "abs_diff":
            z = (va - vb).abs().flatten(1)
        elif self.pair_feature == "hadamard":
            z = (va * vb).flatten(1)
        else:
            z = torch.cat([(va - vb).abs(), va * vb], dim=2).flatten(1)
        return self.mlp(z).squeeze(1)


def probs_for(model, XL, XR, ia, ib, batch):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(ia), batch):
            sl = slice(s, s + batch)
            out.append(torch.sigmoid(model(
                torch.from_numpy(XL[ia[sl]]),
                torch.from_numpy(XR[ib[sl]]))).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


# ===========================================================================
# Evaluation
# ===========================================================================
def read(path, delim):
    return pd.read_csv(path, sep=delim, engine="python", na_filter=False)


def eval_dataset(ds, cfg, params):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    split = cfg["split"]
    train_df = pd.read_csv(f"{SPLIT_ROOT}/{split}/train_set.csv")
    valid_df = pd.read_csv(f"{SPLIT_ROOT}/{split}/valid_set.csv")
    test_df  = pd.read_csv(f"{SPLIT_ROOT}/{split}/test_set.csv")

    if cfg["family"] == "ccer":
        dl = read(cfg["left"], cfg["delim"])
        dr = read(cfg["right"], cfg["delim"])
        if cfg["rename"]:
            dl = dl.rename(columns=cfg["rename"])
            dr = dr.rename(columns=cfg["rename"])
        attrs = [a for a in cfg["attrs"] if a in dl.columns or a in dr.columns]
        tokL, tokR = attr_tokens(dl, attrs), attr_tokens(dr, attrs)
        tok_all = [tokL[a] + tokR[a] for a in range(len(attrs))]
        tag_l, tag_r, symmetric = "A:", "B:", False
    else:
        df = read(cfg["src"], cfg["delim"])
        attrs = [c for c in df.columns
                 if c != cfg["id_col"] and c not in cfg["drop"]
                 and not str(c).startswith("Unnamed:")]
        tok_all = attr_tokens(df, attrs)
        tokL = tokR = tok_all
        tag_l, tag_r, symmetric = "", "", True

    n_attrs = len(attrs)
    cache_idx, cache_vecs = load_cache(ds)
    vocab, M, missing = build_matrix(tok_all, cache_idx, cache_vecs)
    idf = idf_weights(tok_all, vocab)

    max_len = int(params["max_len"])
    if cfg["family"] == "ccer":
        XL = to_ids(tokL, vocab, max_len)
        XR = to_ids(tokR, vocab, max_len)
        posL = {norm_id(v): i for i, v in enumerate(dl[cfg["id_col"]].tolist())}
        posR = {norm_id(v): i for i, v in enumerate(dr[cfg["id_col"]].tolist())}
    else:
        XL = XR = to_ids(tok_all, vocab, max_len)
        posL = posR = {norm_id(v): i for i, v in enumerate(df[cfg["id_col"]].tolist())}

    def rows(sp):
        a = np.array([posL[norm_id(v)] for v in sp["left_id"]], dtype=np.int64)
        b = np.array([posR[norm_id(v)] for v in sp["right_id"]], dtype=np.int64)
        return a, b, sp["label"].astype(int).to_numpy()

    a_tr, b_tr, y_tr = rows(train_df)
    a_va, b_va, y_va = rows(valid_df)
    a_te, b_te, y_te = rows(test_df)

    model = DeepER(M, params, n_attrs, idf=idf)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=float(params["lr"]))
    npos, nneg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(nneg / max(npos, 1), dtype=torch.float32))

    batch = int(params["batch_size"])
    yt = torch.from_numpy(y_tr.astype(np.float32))
    order = np.arange(len(y_tr))
    rng = np.random.RandomState(SEED)
    model.train()
    for _ in range(int(params["epochs"])):
        rng.shuffle(order)
        for s in range(0, len(order), batch):
            sl = order[s:s + batch]
            opt.zero_grad()
            logits = model(torch.from_numpy(XL[a_tr[sl]]),
                           torch.from_numpy(XR[b_tr[sl]]))
            loss = lossf(logits, yt[sl])
            loss.backward()
            opt.step()

    probs_valid = probs_for(model, XL, XR, a_va, b_va, 512)
    probs_test  = probs_for(model, XL, XR, a_te, b_te, 512)

    # Threshold re-chosen on VALIDATION, exactly as the worker does.
    from sklearn.metrics import f1_score
    best_t, best_valid_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        f1 = f1_score(y_va, (probs_valid >= t).astype(int), zero_division=0)
        if f1 > best_valid_f1:
            best_valid_f1, best_t = f1, float(t)

    predicted = set()
    for i, (_, row) in enumerate(test_df.iterrows()):
        if probs_test[i] >= best_t:
            predicted.add((norm_id(row["left_id"]), norm_id(row["right_id"])))

    out = testset_metrics(test_df, lambda na, nb: (na, nb) in predicted,
                          tag_l, tag_r, symmetric)
    out["chosen_threshold"] = round(best_t, 3)
    out["n_attrs"] = n_attrs
    out["tokens_missing_from_cache"] = missing
    return out


def read_best_config(ds):
    """(params, config_id, csv_test_f1) for the max-test_f1 OK row of this dataset."""
    path = os.path.join(RESULTS_DIR, f"deeper_{ds}_configs.csv")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("status") == "OK" and r.get("test_f1") not in (None, "")]
    if not rows:
        raise RuntimeError(f"{ds}: no OK rows in {path}")
    best = max(rows, key=lambda r: float(r["test_f1"]))
    params = dict(
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
        params["lstm_hidden"] = int(float(best["lstm_hidden"]))
    return params, best["config_id"], float(best["test_f1"])


def run_single(ds):
    cfg = DATASETS[ds]
    params, config_id, csv_test_f1 = read_best_config(ds)
    t0 = time.time()
    out = eval_dataset(ds, cfg, params)

    os.makedirs(PAIRS_DIR, exist_ok=True)
    with open(os.path.join(PAIRS_DIR, f"deeper_{ds}_pred_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["left_id", "right_id"]); w.writerows(out["dump_pairs"])
    with open(os.path.join(PAIRS_DIR, f"deeper_{ds}_entities.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["entity_id"]); w.writerows([[e] for e in out["entities"]])

    pw, bc = out["pairwise"], out["bcubed"]
    result = {
        "dataset": ds, "config_id": config_id, "family": cfg["family"],
        "csv_test_f1": round(csv_test_f1, 6),
        "pairwise_precision": round(pw[0], 6), "pairwise_recall": round(pw[1], 6),
        "pairwise_f1": round(pw[2], 6),
        "bcubed_precision": round(bc[0], 6), "bcubed_recall": round(bc[1], 6),
        "bcubed_f1": round(bc[2], 6),
        "n_entities": out["n_entities"], "n_pred_pairs": out["n_pred_pairs"],
        "n_gt_pairs": out["n_gt_pairs"], "time_sec": round(time.time() - t0, 2),
        "status": "OK",
    }
    print("RESULT_JSON:" + json.dumps(result))
    return result


SUMMARY_COLS = ["dataset", "config_id", "family", "csv_test_f1",
                "pairwise_precision", "pairwise_recall", "pairwise_f1",
                "bcubed_precision", "bcubed_recall", "bcubed_f1",
                "n_entities", "n_pred_pairs", "n_gt_pairs", "time_sec", "status"]


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in SUMMARY_COLS})


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for ds in ALL_DATASETS:
        print(f"\n=== {ds} ===", flush=True)
        try:
            r = run_single(ds)
            d = r["pairwise_f1"] - r["csv_test_f1"]
            flag = "" if abs(d) < 1e-6 else "   <-- MISMATCH, investigate"
            print(f"  pairwise_f1={r['pairwise_f1']:.6f}  csv_test_f1={r['csv_test_f1']:.6f}"
                  f"  delta={d:+.6f}{flag}", flush=True)
            print(f"  bcubed_f1={r['bcubed_f1']:.6f}  n_entities={r['n_entities']}"
                  f"  n_gt_pairs={r['n_gt_pairs']}", flush=True)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            r = {"dataset": ds, "family": DATASETS[ds]["family"], "status": f"ERROR: {e}"}
        rows.append(r)
        _write(SUMMARY_CSV, rows)          # checkpoint after every dataset
    print(f"\nWrote {SUMMARY_CSV}")
    return rows


def main():
    args = [a for a in sys.argv[1:]]
    if args:
        for ds in args:
            run_single(ds.upper() if ds.upper() in DATASETS else ds)
    else:
        run_all()


if __name__ == "__main__":
    main()
