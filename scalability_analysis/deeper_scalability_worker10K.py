"""
deeper_scalability_worker10K.py
-------------------------------
One DeepER config on the synthetic FEBRL 10K dataset -> RESULT_JSON.

Tuning size for the scalability ladder, same role as the other frameworks'
*_scalability_worker10K.py. Protocol is the frozen one: SEED=42, threshold
chosen on VALIDATION over np.arange(0, 1.001, 0.01), RESULT_JSON contract.

DeepER is a pair classifier over the fixed split pairs -- it does NOT do
candidate generation, so unlike Dedupe / Splink / pyJedAI / Zingg it eats no
blocking loss and folds no blocking cost into its runtime. That is the same
treatment DeepER gets in the CCER and DER experiments and must be stated in the
paper; it matters most here, where runtime against N is the metric.

Records are embedded for the WHOLE table, exactly as the CCER/DER workers do,
so memory does grow with N.
"""
import sys
import os
import re
import json
import time
import resource
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

CONVERTED_DIR = "/home/it2022025/er_scalability/converted"
SPLITS_DIR    = "/home/it2022025/er_scalability/splits"
EMB_DIR       = "/home/it2022025/er_scalability/embeddings"
DATASET       = "10K"
ID_COL        = "id"

# The FEBRL attribute set Dedupe uses (df_to_records in dedupe_scalability_fixed.py).
# Listed EXPLICITLY rather than "every non-id column" on purpose: FEBRL profiles
# can carry a record identifier that encodes the true entity, and embedding such
# a column would leak the label. Anything present but unused is reported.
ATTRS = ["given_name", "surname", "address_1", "suburb", "postcode",
         "state", "date_of_birth", "soc_sec_id", "phone_number"]

EMB_DIM = 300
THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def norm_id(v):
    s = str(v).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def tokenize(text):
    return TOKEN_RE.findall(text)


def attr_tokens(df, attrs):
    out = []
    for a in attrs:
        col = (df[a].fillna("").astype(str).str.lower().tolist()
               if a in df.columns else [""] * len(df))
        out.append([tokenize(t) for t in col])
    return out


def resolve_attrs(df):
    present = [a for a in ATTRS if a in df.columns]
    unused = [c for c in df.columns if c != ID_COL and c not in present]
    return present, unused


def load_cache(tag):
    """Returns (sorted token array, vectors).

    The builder writes tokens sorted, so lookup is a binary search rather than a
    python dict. At 2M records the vocabulary is ~3.8M tokens and that dict cost
    several hundred MB on its own.
    """
    z = np.load(os.path.join(EMB_DIR, f"ft_cache_SYN{tag}.npz"), allow_pickle=False)
    return z["tokens"], z["vectors"]


def build_matrix(token_lists, cache_toks, cache_vecs):
    """Vocabulary + embedding matrix, via one vectorised lookup.

    `cache_vecs` is ~4.5GB at 2M records, so rows are gathered in one shot and
    the caller frees the cache immediately afterwards -- holding the cache and
    the matrix at once is what pushes the big sizes over the memory cap.
    """
    vocab = {"<pad>": 0, "<unk>": 1}
    for per_attr in token_lists:
        for toks in per_attr:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
    words = np.array([w for w, i in sorted(vocab.items(), key=lambda kv: kv[1])])
    pos = np.searchsorted(cache_toks, words)
    pos_clipped = np.clip(pos, 0, len(cache_toks) - 1)
    hit = cache_toks[pos_clipped] == words
    hit[:2] = False                                   # <pad>, <unk> stay zero
    M = np.zeros((len(vocab), EMB_DIM), dtype=np.float32)
    M[hit] = cache_vecs[pos_clipped[hit]]
    missing = int((~hit).sum()) - 2
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
    n_attrs = len(token_lists)
    n = len(token_lists[0]) if n_attrs else 0
    # int32: at 2M x 9 attrs this halves the tensor. Cast to int64 per
    # batch at the embedding lookup, which torch requires.
    X = np.zeros((n, n_attrs, max_len), dtype=np.int32)
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


def _t(a):
    """int32 record rows -> the int64 tensor nn.Embedding requires."""
    return torch.from_numpy(a.astype(np.int64))


def probs_for(model, X, ia, ib, batch):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(ia), batch):
            sl = slice(s, s + batch)
            out.append(torch.sigmoid(model(
                _t(X[ia[sl]]), _t(X[ib[sl]]))).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def main():
    cfg = json.loads(sys.argv[1])
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    t_start = time.time()

    df = pd.read_csv(os.path.join(CONVERTED_DIR, DATASET, "profiles.csv"),
                     engine="python", na_filter=False)
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, DATASET, "train_set.csv"))
    valid_df = pd.read_csv(os.path.join(SPLITS_DIR, DATASET, "valid_set.csv"))
    test_df  = pd.read_csv(os.path.join(SPLITS_DIR, DATASET, "test_set.csv"))

    attrs, unused = resolve_attrs(df)
    n_attrs = len(attrs)
    tok = attr_tokens(df, attrs)

    cache_toks, cache_vecs = load_cache(DATASET)
    vocab, M, missing = build_matrix(tok, cache_toks, cache_vecs)
    del cache_toks, cache_vecs
    idf = idf_weights(tok, vocab)

    max_len = int(cfg["max_len"])
    X = to_ids(tok, vocab, max_len)
    pos = {norm_id(v): i for i, v in enumerate(df[ID_COL].tolist())}

    def rows(sp):
        a = np.array([pos[norm_id(v)] for v in sp["left_id"]], dtype=np.int64)
        b = np.array([pos[norm_id(v)] for v in sp["right_id"]], dtype=np.int64)
        return a, b, sp["label"].astype(int).to_numpy()

    a_tr, b_tr, y_tr = rows(train_df)
    a_va, b_va, y_va = rows(valid_df)
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
    rng = np.random.RandomState(seed)
    model.train()
    for _ in range(int(cfg["epochs"])):
        rng.shuffle(order)
        for s in range(0, len(order), batch):
            sl = order[s:s + batch]
            opt.zero_grad()
            logits = model(_t(X[a_tr[sl]]), _t(X[b_tr[sl]]))
            loss = lossf(logits, yt[sl])
            loss.backward()
            opt.step()

    probs_valid = probs_for(model, X, a_va, b_va, 512)
    probs_test  = probs_for(model, X, a_te, b_te, 512)

    best_t, best_valid_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        f1 = f1_score(y_va, (probs_valid >= t).astype(int), zero_division=0)
        if f1 > best_valid_f1:
            best_valid_f1, best_t = f1, float(t)

    preds_test = (probs_test >= best_t).astype(int)
    test_p = precision_score(y_te, preds_test, zero_division=0)
    test_r = recall_score(y_te, preds_test, zero_division=0)
    test_f1 = f1_score(y_te, preds_test, zero_division=0)

    curve = []
    for t in THRESHOLD_GRID:
        pr = (probs_test >= t).astype(int)
        curve.append({"t": round(float(t), 3),
                      "precision": round(precision_score(y_te, pr, zero_division=0), 6),
                      "recall": round(recall_score(y_te, pr, zero_division=0), 6),
                      "f1": round(f1_score(y_te, pr, zero_division=0), 6)})

    pkeys = ["composition", "pair_feature", "hidden_dim", "lr", "epochs",
             "dropout", "batch_size", "max_len", "lstm_hidden", "tune_embeddings"]
    result = {
        "config_id": cfg.get("config_id"),
        "params": {k: cfg.get(k) for k in pkeys if k in cfg},
        "status": "OK",
        "chosen_threshold": round(best_t, 3),
        "valid_f1_at_threshold": round(best_valid_f1, 6),
        "test_point": {"t": round(best_t, 3), "precision": round(test_p, 6),
                       "recall": round(test_r, 6), "f1": round(test_f1, 6)},
        "pr_curve": curve,
        "n_records": len(df), "n_attrs": n_attrs, "attrs": attrs,
        "unused_columns": unused,
        "vocab_size": len(vocab), "tokens_missing_from_cache": missing,
        "time_sec": round(time.time() - t_start, 2),
        "peak_mem_mb": round(peak_mem_mb(), 1),
    }
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
