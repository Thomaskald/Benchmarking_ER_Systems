"""
deeper_build_cache_syn.py
-------------------------
Builds the fastText vector caches for the synthetic FEBRL scalability ladder,
the counterpart of DEEPER/deeper_build_cache.py for the D2..D9 / CORA / CDDB
datasets. Same reasoning: the model is ~7GB and the search spawns a fresh worker
per config, so every token a dataset contains is resolved to its vector once and
written to a small .npz the workers read in under a second.

Writes embeddings/ft_cache_SYN<size>.npz for each size.

Usage:
    python3 deeper_build_cache_syn.py             # all sizes
    python3 deeper_build_cache_syn.py 10K 50K     # only these

Build the small sizes first and check the .npz sizes before committing to 1M/2M:
FEBRL vocabulary is mostly names, suburbs and postcodes, so it grows far more
slowly than the record count, but 2M still has to be tokenised once.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

CONVERTED_DIR = "/path/to/your/er_scalability/converted"
EMB_DIR = "/path/to/your/er_scalability/embeddings"
FT_MODEL = os.path.join(EMB_DIR, "wiki.en.bin")
EMB_DIM = 300

TOKEN_RE = re.compile(r"[a-z0-9]+")
SIZES = ["10K", "50K", "100K", "200K", "300K", "1M", "2M"]
ID_COL = "id"

# Must match ATTRS in deeper_scalability_worker10K.py exactly -- a worker
# reporting tokens_missing_from_cache > 0 means these two have drifted.
ATTRS = ["given_name", "surname", "address_1", "suburb", "postcode",
         "state", "date_of_birth", "soc_sec_id", "phone_number"]


def vocab_of(path):
    df = pd.read_csv(path, engine="python", na_filter=False)
    present = [a for a in ATTRS if a in df.columns]
    unused = [c for c in df.columns if c != ID_COL and c not in present]
    vocab = set()
    for a in present:
        for t in df[a].fillna("").astype(str).str.lower().tolist():
            vocab.update(TOKEN_RE.findall(t))
    return vocab, present, unused, len(df)


def load_model():
    """fastText if the binding is installed, gensim as a fallback."""
    try:
        import fasttext
        m = fasttext.load_model(FT_MODEL)
        return lambda w: m.get_word_vector(w)
    except ImportError:
        pass
    from gensim.models.fasttext import load_facebook_vectors
    kv = load_facebook_vectors(FT_MODEL)
    return lambda w: kv[w]


def main():
    wanted = [a.upper() for a in sys.argv[1:]]
    sizes = [s for s in SIZES if not wanted or s in wanted]

    vocabs = {}
    for s in sizes:
        path = os.path.join(CONVERTED_DIR, s, "profiles.csv")
        if not os.path.exists(path):
            print(f"{s:5s} SKIP -- {path} not found")
            continue
        v, present, unused, n = vocab_of(path)
        vocabs[s] = v
        print(f"{s:5s} {n:,} records  attrs={present}")
        if unused:
            print(f"      columns present but NOT embedded: {unused}")
        missing_attrs = [a for a in ATTRS if a not in present]
        if missing_attrs:
            print(f"      WARNING declared but absent: {missing_attrs}")
        print(f"      {len(v):,} distinct tokens", flush=True)

    if not vocabs:
        print("nothing to build")
        return

    print(f"\nloading {FT_MODEL} (this is the slow part) ...", flush=True)
    vec = load_model()

    os.makedirs(EMB_DIR, exist_ok=True)
    for s, vocab in vocabs.items():
        toks = sorted(vocab)
        M = np.zeros((len(toks), EMB_DIM), dtype=np.float32)
        for i, t in enumerate(toks):
            M[i] = vec(t)
        out = os.path.join(EMB_DIR, f"ft_cache_SYN{s}.npz")
        np.savez(out, tokens=np.array(toks), vectors=M)
        print(f"{s:5s} -> {out}  ({len(toks):,} x {EMB_DIM}, "
              f"{os.path.getsize(out)/1024**2:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
