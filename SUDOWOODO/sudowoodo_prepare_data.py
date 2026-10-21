"""Write the study's datasets in the COL/VAL format Sudowoodo reads.

Sudowoodo inherits Ditto's serialization: one pair per line, three
tab-separated fields,

    COL name VAL usb cable COL price VAL 9.99 <TAB> COL name VAL usb cord ... <TAB> 1

under data/<task_type>/<task>/{train,valid,test}.txt, plus an unlabeled
corpus train_no_label.txt with one serialized record per line.

Three things are worth stating in the paper:

  * The unlabeled corpus is NOT an extra input. Upstream's convert_data.py
    derives it from train.txt alone, and this script does the same, so
    Sudowoodo's self-supervised pre-training sees exactly the records the
    other frameworks' training split contains -- no test-set records, no
    corpus no other method gets.

  * The attribute set per dataset is taken verbatim from the DeepER workers,
    which took it from the RecordLinkage evaluator. Every framework that
    needs a schema alignment therefore uses the same one.

  * Records are serialized from the FIXED train/valid/test pair splits, so
    Sudowoodo is scored as a pair classifier over the same pairs as Magellan,
    RecordLinkage and DeepER. Its own blocking (--k) is a separate experiment
    and is not exercised here.

Usage (on the cluster, after sourcing activate.sh):

    python sudowoodo_prepare_data.py                # all 10 datasets
    python sudowoodo_prepare_data.py D4 CORA        # just these
"""

import os
import sys

import pandas as pd

BASE = "/path/to/your/er_scalability"
OUT_BASE = os.path.join(BASE, "sudowoodo", "src", "data", "em")

# The SCADS movie benchmark ships ontology URIs as column names. Same map the
# DeepER and RecordLinkage workers use.
SCADS_RENAME = {
    "https://www.scads.de/movieBenchmark/ontology/title": "title",
    "https://www.scads.de/movieBenchmark/ontology/name": "name",
    "https://www.scads.de/movieBenchmark/ontology/genre_list": "genre_list",
    "http://dbpedia.org/ontology/abstract": "abstract",
    "http://dbpedia.org/ontology/episodeNumber": "episodeNumber",
    "http://dbpedia.org/ontology/seasonNumber": "seasonNumber",
    "http://dbpedia.org/ontology/releaseDate": "releaseDate",
}

# task -> spec. `kind` is "ccer" (two tables) or "der" (one dirty table).
DATASETS = {
    "D2": dict(kind="ccer", left="datasets/D2/abt.csv", right="datasets/D2/buy.csv",
               split="train_validation_test_sets/db2", delim="|", id_col="id",
               rename=None, attrs=["name", "description", "price"]),
    "D3": dict(kind="ccer", left="datasets/D3/amazon.csv", right="datasets/D3/gp.csv",
               split="train_validation_test_sets/db3", delim="#", id_col="id",
               rename=None, attrs=["title", "description", "manufacturer", "price"]),
    "D4": dict(kind="ccer", left="datasets/D4/dblp.csv", right="datasets/D4/acm.csv",
               split="train_validation_test_sets/db4", delim="%", id_col="id",
               rename=None, attrs=["title", "authors", "venue", "year"]),
    "D5": dict(kind="ccer", left="datasets/D5/imdb.csv", right="datasets/D5/tmdb.csv",
               split="train_validation_test_sets/db5", delim="|", id_col="id",
               rename=SCADS_RENAME,
               attrs=["title", "name", "genre_list", "episodeNumber", "seasonNumber"]),
    "D6": dict(kind="ccer", left="datasets/D6/imdb.csv", right="datasets/D6/tvdb.csv",
               split="train_validation_test_sets/db6", delim="|", id_col="id",
               rename=SCADS_RENAME,
               attrs=["title", "name", "episodeNumber", "seasonNumber"]),
    "D7": dict(kind="ccer", left="datasets/D7/tmdb.csv", right="datasets/D7/tvdb.csv",
               split="train_validation_test_sets/db7", delim="|", id_col="id",
               rename=SCADS_RENAME,
               attrs=["title", "name", "abstract", "episodeNumber", "seasonNumber",
                      "releaseDate"]),
    "D8": dict(kind="ccer", left="datasets/D8/walmart.csv", right="datasets/D8/amazon.csv",
               split="train_validation_test_sets/db8", delim="|", id_col="id",
               rename=None,
               attrs=["title", "brand", "modelno", "dimensions", "price", "shipweight"]),
    "D9": dict(kind="ccer", left="datasets/D9/dblp.csv", right="datasets/D9/scholar.csv",
               split="train_validation_test_sets/db9", delim=">", id_col="id",
               rename=None, attrs=["title", "authors", "venue", "year"]),
    "CORA": dict(kind="der", src="datasets/cora/cora.csv",
                 split="train_validation_test_sets/cora", delim="|",
                 id_col="Entity Id", rename=None, drop_cols=[]),
    # CDDB's `tracks` field holds very long strings; the pyJedAI and DeepER
    # workers exclude it, so it is excluded here too.
    "CDDB": dict(kind="der", src="datasets/CDDB/cddb.csv",
                 split="train_validation_test_sets/cddb", delim=",",
                 id_col="id", rename=None, drop_cols=["tracks"]),
}


def norm_key(v):
    """Canonical string id -- '123', 123 and '123.0' all key the same row.

    Identical to the DeepER workers, so a pair that joins there joins here.
    """
    s = str(v).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def clean(v):
    """Squash anything that would break the tab-separated line format."""
    s = "" if v is None else str(v)
    return " ".join(s.split())


def serialize(row, attrs):
    """One record as `COL <attr> VAL <value>` segments, Ditto style."""
    out = []
    for a in attrs:
        out.append("COL %s VAL %s" % (a, clean(row.get(a, ""))))
    return " ".join(out)


def resolve_attrs(declared, *frames):
    """Declared attributes present on at least one side, in declared order.

    Both sides get the same list so attribute j lines up across the pair, the
    same rule the DeepER workers apply.
    """
    present = [a for a in declared if any(a in f.columns for f in frames)]
    if not present:
        raise RuntimeError("no declared attribute found; columns=%s"
                           % [list(f.columns) for f in frames])
    return present


def read_table(path, delim, rename):
    df = pd.read_csv(path, sep=delim, engine="python", na_filter=False)
    if rename:
        df = df.rename(columns=rename)
    return df


def build(task, spec):
    out_dir = os.path.join(OUT_BASE, task)
    os.makedirs(out_dir, exist_ok=True)

    if spec["kind"] == "ccer":
        dl = read_table(os.path.join(BASE, spec["left"]), spec["delim"], spec["rename"])
        dr = read_table(os.path.join(BASE, spec["right"]), spec["delim"], spec["rename"])
        attrs = resolve_attrs(spec["attrs"], dl, dr)
        left_ser = {norm_key(r[spec["id_col"]]): serialize(r, attrs)
                    for _, r in dl.iterrows()}
        right_ser = {norm_key(r[spec["id_col"]]): serialize(r, attrs)
                     for _, r in dr.iterrows()}
    else:
        src = read_table(os.path.join(BASE, spec["src"]), spec["delim"], spec["rename"])
        # Dirty ER needs no schema alignment -- both sides of a pair come from
        # the same table, so every column is aligned with itself.
        #
        # `Unnamed: N` is pandas' name for a header-less column, which a
        # trailing delimiter produces. cora.csv has one, empty on all 1295
        # rows. Excluded for the same reason the DeepER worker excludes it:
        # it carries no data, and here it would emit a `COL Unnamed: N VAL `
        # segment on every record, spending max_len budget on nothing.
        attrs = [c for c in src.columns
                 if c != spec["id_col"] and c not in spec["drop_cols"]
                 and not str(c).startswith("Unnamed:")]
        left_ser = {norm_key(r[spec["id_col"]]): serialize(r, attrs)
                    for _, r in src.iterrows()}
        right_ser = left_ser

    counts = {}
    nolabel_lines = []
    for split, fname in (("train_set.csv", "train.txt"),
                         ("valid_set.csv", "valid.txt"),
                         ("test_set.csv", "test.txt")):
        sp = pd.read_csv(os.path.join(BASE, spec["split"], split))
        lines = []
        for _, r in sp.iterrows():
            a = left_ser[norm_key(r["left_id"])]
            b = right_ser[norm_key(r["right_id"])]
            lines.append("%s\t%s\t%d" % (a, b, int(r["label"])))
            if fname == "train.txt":
                # Exactly what upstream convert_data.py emits: both sides of
                # every training pair, duplicates included. BTDataset samples
                # 10000 instances from this, so the duplication is a mild
                # frequency weighting, not a bug -- kept so the corpus matches
                # the authors'.
                nolabel_lines.append(a)
                nolabel_lines.append(b)
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write("\n".join(lines) + "\n")
        counts[fname] = len(lines)

    with open(os.path.join(out_dir, "train_no_label.txt"), "w") as f:
        f.write("\n".join(nolabel_lines) + "\n")

    print("%-6s attrs=%-2d  train=%-6d valid=%-6d test=%-6d  nolabel=%-6d (uniq %d)"
          % (task, len(attrs), counts["train.txt"], counts["valid.txt"],
             counts["test.txt"], len(nolabel_lines), len(set(nolabel_lines))))


def main():
    tasks = sys.argv[1:] or list(DATASETS)
    unknown = [t for t in tasks if t not in DATASETS]
    if unknown:
        raise SystemExit("unknown dataset(s): %s\nknown: %s"
                         % (unknown, list(DATASETS)))
    print("writing COL/VAL data under %s" % OUT_BASE)
    for t in tasks:
        build(t, DATASETS[t])


if __name__ == "__main__":
    main()
