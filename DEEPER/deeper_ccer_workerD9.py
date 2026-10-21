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

# DeepER (Ebraheem et al., VLDB 2018) on CCER dataset D9.
#
# The implementation follows the authors' design on the two points that decide
# what the model can actually see:
#
#   1. EMBEDDINGS -- fastText subword vectors, not a generic English word
#      vocabulary with a fallback for unknown tokens. Product and
#      bibliographic records are carried by model numbers, SKUs and
#      identifiers that a generic English vocabulary has never seen; fastText
#      composes a vector for them from character n-grams, so an unseen token
#      still lands near other tokens that share its substrings. Vectors are
#      read from a per-dataset cache built once by deeper_ft_build_cache.py
#      (loading the ~7GB model inside every config would dominate runtime).
#
#   2. COMPOSITION -- per attribute, not over the whole record. Each aligned
#      attribute is composed on its own and the two records are compared
#      attribute against attribute, so `title` is matched against `title`
#      rather than against a bag of every field concatenated. The attribute
#      alignment is taken verbatim from the RecordLinkage evaluator, so both
#      frameworks see the same schema mapping.
#
# Composition is SEARCHED and covers both variants from the paper:
#     average / idf_average -> the simple compositions ("DeepER-Lite")
#     lstm                  -> the bi-directional RNN composition, the paper's
#                              main contribution
# The pair feature is SEARCHED too and includes the paper's per-attribute
# similarity vector (2 x num_attributes) as well as the denser element-wise
# variants.
#
# Evaluated as a pair classifier over the fixed train/valid/test split files --
# identical treatment to MAGELLAN and RECORDLINKAGE, so DeepER's own LSH
# blocking is not exercised (consistent with how the other pair classifiers are
# scored).

D1_PATH    = "/path/to/your/er_scalability/datasets/D9/dblp.csv"
D2_PATH    = "/path/to/your/er_scalability/datasets/D9/scholar.csv"
TRAIN_PATH = "/path/to/your/er_scalability/train_validation_test_sets/db9/train_set.csv"
VALID_PATH = "/path/to/your/er_scalability/train_validation_test_sets/db9/valid_set.csv"
TEST_PATH  = "/path/to/your/er_scalability/train_validation_test_sets/db9/test_set.csv"
DELIM      = ">"
ID_COL     = "id"

# Aligned attributes. Taken verbatim from the column names the RecordLinkage
# evaluator compares on this dataset, so the two frameworks see the same schema
# mapping and neither gets an advantage from a better alignment.
RENAME = None
ATTRS  = ["title", "authors", "venue", "year"]

FT_CACHE = "/path/to/your/er_scalability/embeddings/ft_cache_D9.npz"
EMB_DIM  = 300

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def tokenize(text):
    return TOKEN_RE.findall(text)


def norm_key(v):
    """Canonical string id. '123', 123 and '123.0' all key the same row."""
    s = str(v).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def prepare(df):
    return df.rename(columns=RENAME) if RENAME else df


def resolve_attrs(dl, dr):
    """The declared attributes that are actually present on at least one side.

    Both sides get the SAME list in the SAME order -- that is what makes
    attribute j of the left record comparable with attribute j of the right
    one. An attribute missing from one side composes to the zero vector there,
    which the similarity features read as "no evidence".
    """
    present = [a for a in ATTRS if a in dl.columns or a in dr.columns]
    if not present:
        raise RuntimeError(f"no declared attribute found; left={list(dl.columns)} "
                           f"right={list(dr.columns)}")
    return present


def attr_tokens(df, attrs):
    """tokens[a][i] -- the token list of attribute a for record i."""
    out = []
    for a in attrs:
        col = (df[a].fillna("").astype(str).str.lower().tolist()
               if a in df.columns else [""] * len(df))
        out.append([tokenize(t) for t in col])
    return out


def load_cache():
    """Token -> fastText vector, precomputed by deeper_ft_build_cache.py."""
    z = np.load(FT_CACHE, allow_pickle=False)
    toks = z["tokens"]
    vecs = z["vectors"]
    return {str(t): i for i, t in enumerate(toks)}, vecs


def build_matrix(token_lists, cache_idx, cache_vecs):
    """Vocabulary + embedding matrix. Row 0 is PAD, row 1 is UNK.

    Every token in the dataset has a fastText vector in the cache -- subword
    composition means there is no out-of-vocabulary case to paper over. A
    non-zero `missing` here means the cache is stale with respect to the data
    and is reported in RESULT_JSON rather than silently absorbed.
    """
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
    """Corpus-level idf over records (a record is the concatenation of its
    attributes), matching the idf_average composition in the paper."""
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
        # DeepER considers both fixed pre-trained vectors and vectors tuned for
        # the ER task; which one is used is SEARCHED.
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
            in_dim = 2 * n_attrs          # the paper's similarity vector
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
        """x: B x A x L  ->  B x A x rep. Attributes are folded into the batch
        so every attribute is composed independently by the same encoder."""
        B, A, L = x.shape
        xf = x.reshape(B * A, L)
        e = self.emb(xf)                                # BA x L x D
        mask = (xf != 0).float().unsqueeze(-1)          # BA x L x 1
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
        va, vb = self.compose(xa), self.compose(xb)     # B x A x rep
        if self.pair_feature == "similarity":
            # Two scalars per attribute: cosine and a scale-free distance.
            # An empty attribute composes to the zero vector, which cosine
            # reads as 0 -- "no evidence" rather than a spurious match.
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


def main():
    cfg = json.loads(sys.argv[1])
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    t_start = time.time()

    dl = prepare(pd.read_csv(D1_PATH, sep=DELIM, engine="python", na_filter=False))
    dr = prepare(pd.read_csv(D2_PATH, sep=DELIM, engine="python", na_filter=False))
    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    attrs = resolve_attrs(dl, dr)
    n_attrs = len(attrs)
    tokL = attr_tokens(dl, attrs)
    tokR = attr_tokens(dr, attrs)
    # idf and the vocabulary are built over both tables, attribute by attribute.
    tok_all = [tokL[a] + tokR[a] for a in range(n_attrs)]

    cache_idx, cache_vecs = load_cache()
    vocab, M, missing = build_matrix(tok_all, cache_idx, cache_vecs)
    idf = idf_weights(tok_all, vocab)

    max_len = int(cfg["max_len"])
    XL = to_ids(tokL, vocab, max_len)
    XR = to_ids(tokR, vocab, max_len)
    posL = {norm_key(v): i for i, v in enumerate(dl[ID_COL].tolist())}
    posR = {norm_key(v): i for i, v in enumerate(dr[ID_COL].tolist())}

    def rows(sp):
        a = np.array([posL[norm_key(v)] for v in sp["left_id"]], dtype=np.int64)
        b = np.array([posR[norm_key(v)] for v in sp["right_id"]], dtype=np.int64)
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
            logits = model(torch.from_numpy(XL[a_tr[sl]]),
                           torch.from_numpy(XR[b_tr[sl]]))
            loss = lossf(logits, yt[sl])
            loss.backward()
            opt.step()

    probs_valid = probs_for(model, XL, XR, a_va, b_va, 512)
    probs_test  = probs_for(model, XL, XR, a_te, b_te, 512)

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
        "n_attrs": n_attrs,
        "attrs": attrs,
        "vocab_size": len(vocab),
        "tokens_missing_from_cache": missing,
        "time_sec": round(time.time() - t_start, 2),
        "peak_mem_mb": round(peak_mem_mb(), 1),
    }
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
