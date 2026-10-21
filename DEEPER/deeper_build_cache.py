"""Build the per-dataset fastText vector cache used by every DeepER worker.

Run ONCE per dataset, before any search job. The fastText model is ~7GB; the
search drivers spawn a fresh worker process for every one of the 50 configs, so
loading the model inside the worker would cost that load 50 times over and would
sit well above the 12GB per-config memory cap. Instead this script resolves
every token the dataset actually contains to its fastText vector and writes a
small .npz the workers read in well under a second.

fastText composes a vector for ANY token out of its character n-grams, so there
is no out-of-vocabulary case here -- which is the whole reason for moving off
GloVe. The cache therefore covers the dataset vocabulary exactly, and a worker
reporting `tokens_missing_from_cache > 0` means the cache is stale.

Tokenization, the rename map and the attribute list are duplicated from the
workers verbatim; a worker verifies coverage at run time so any drift shows up
as a warning instead of a silent zero vector.

Usage:
    python3 deeper_build_cache.py            # all 10 datasets
    python3 deeper_build_cache.py D8 CORA    # only these
"""
import os
import re
import sys

import numpy as np
import pandas as pd

DATA_ROOT = "/path/to/your/er_scalability/datasets"
EMB_DIR   = "/path/to/your/er_scalability/embeddings"
# The authors use the Wikipedia fastText model. cc.en.300.bin works too -- if
# you switch, say which one in the paper, and rebuild every cache.
FT_MODEL  = os.path.join(EMB_DIR, "wiki.en.bin")
EMB_DIM   = 300

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

CCER = {
    "D2": dict(left="D2/abt.csv", right="D2/buy.csv", delim="|", rename=None,
               attrs=["name", "description", "price"]),
    "D3": dict(left="D3/amazon.csv", right="D3/gp.csv", delim="#", rename=None,
               attrs=["title", "description", "manufacturer", "price"]),
    "D4": dict(left="D4/dblp.csv", right="D4/acm.csv", delim="%", rename=None,
               attrs=["title", "authors", "venue", "year"]),
    "D5": dict(left="D5/imdb.csv", right="D5/tmdb.csv", delim="|",
               rename=SCADS_RENAME,
               attrs=["title", "name", "genre_list", "episodeNumber",
                      "seasonNumber"]),
    "D6": dict(left="D6/imdb.csv", right="D6/tvdb.csv", delim="|",
               rename=SCADS_RENAME,
               attrs=["title", "name", "episodeNumber", "seasonNumber"]),
    "D7": dict(left="D7/tmdb.csv", right="D7/tvdb.csv", delim="|",
               rename=SCADS_RENAME,
               attrs=["title", "name", "abstract", "episodeNumber",
                      "seasonNumber", "releaseDate"]),
    "D8": dict(left="D8/walmart.csv", right="D8/amazon.csv", delim="|", rename=None,
               attrs=["title", "brand", "modelno", "dimensions", "price",
                      "shipweight"]),
    "D9": dict(left="D9/dblp.csv", right="D9/scholar.csv", delim=">", rename=None,
               attrs=["title", "authors", "venue", "year"]),
}

DER = {
    "CORA": dict(src="cora/cora.csv", delim="|", id_col="Entity Id", drop=[]),
    "CDDB": dict(src="CDDB/cddb.csv", delim=",", id_col="id", drop=["tracks"]),
}


def read(path, delim):
    return pd.read_csv(path, sep=delim, engine="python", na_filter=False)


def tokens_of(df, attrs):
    vocab = set()
    for a in attrs:
        if a not in df.columns:
            continue
        for t in df[a].fillna("").astype(str).str.lower().tolist():
            vocab.update(TOKEN_RE.findall(t))
    return vocab


def ccer_vocab(spec):
    dl = read(os.path.join(DATA_ROOT, spec["left"]), spec["delim"])
    dr = read(os.path.join(DATA_ROOT, spec["right"]), spec["delim"])
    if spec["rename"]:
        dl = dl.rename(columns=spec["rename"])
        dr = dr.rename(columns=spec["rename"])
    attrs = [a for a in spec["attrs"] if a in dl.columns or a in dr.columns]
    missing = [a for a in spec["attrs"] if a not in attrs]
    return tokens_of(dl, attrs) | tokens_of(dr, attrs), attrs, missing


def der_vocab(spec):
    df = read(os.path.join(DATA_ROOT, spec["src"]), spec["delim"])
    attrs = [c for c in df.columns
             if c != spec["id_col"] and c not in spec["drop"]
             and not str(c).startswith("Unnamed:")]
    return tokens_of(df, attrs), attrs, []


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
    todo = [(k, v, "ccer") for k, v in CCER.items()] + \
           [(k, v, "der") for k, v in DER.items()]
    if wanted:
        todo = [t for t in todo if t[0] in wanted]

    vocabs = {}
    for name, spec, kind in todo:
        vocab, attrs, missing = (ccer_vocab(spec) if kind == "ccer"
                                 else der_vocab(spec))
        vocabs[name] = vocab
        print(f"{name:5s} attrs={attrs}")
        if missing:
            print(f"      WARNING declared but absent from both sides: {missing}")
        print(f"      {len(vocab)} distinct tokens")

    print(f"\nloading {FT_MODEL} (this is the slow part) ...", flush=True)
    vec = load_model()

    os.makedirs(EMB_DIR, exist_ok=True)
    for name, vocab in vocabs.items():
        toks = sorted(vocab)
        M = np.zeros((len(toks), EMB_DIM), dtype=np.float32)
        for i, t in enumerate(toks):
            M[i] = vec(t)
        out = os.path.join(EMB_DIR, f"ft_cache_{name}.npz")
        # Uncompressed on purpose: every one of the 50 configs reads this file,
        # so load speed matters more than the few hundred MB on disk.
        np.savez(out, tokens=np.array(toks), vectors=M)
        print(f"{name:5s} -> {out}  ({len(toks)} x {EMB_DIM})")


if __name__ == "__main__":
    main()
