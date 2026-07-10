#!/usr/bin/env python3
"""
Best-config evaluation at TWO levels: pairwise + cluster-level (B-cubed) -- SPLINK.

Mirrors the pyjedai `pyjedai_bestconfig_eval.py` structure, but every pipeline is
the *Splink* pipeline (your existing workers), unchanged. For each dataset it:

  1. reads the BEST config (max test_f1 among status==OK rows) straight from the
     existing results/splink_<DS>_configs.csv,
  2. reruns exactly that one config's Splink pipeline (same steps as your workers:
     estimate u -> supervised m from train labels -> patch match rate -> predict),
  3. reports PAIRWISE P/R/F1 the same way your workers do (score_map + chosen
     threshold on test_set.csv), as a sanity cross-check vs your existing numbers,
  4. reports CLUSTER-LEVEL B-cubed P/R/F1. The predicted clustering comes from
     Splink's OWN clusterer -- linker.clustering.cluster_pairwise_predictions_at_threshold at the
     chosen threshold -- compared against the full ground-truth clustering
     (connected components of the GT pair list) over EVERY entity,
  5. dumps the predicted match pairs + the entity id universe under results/pairs/.

Usage
-----
  # one dataset (prints RESULT_JSON, writes pair dumps):
  python3 splink_bestconfig_eval.py D2

  # ALL datasets (each in its own subprocess) + summary CSV:
  python3 splink_bestconfig_eval.py

Output
------
  results/splink_bestconfig_eval.csv          one row per dataset, both metric levels
  results/pairs/splink_<DS>_pred_pairs.csv     predicted matches (left_id,right_id)
  results/pairs/splink_<DS>_entities.csv       full entity id universe (one col)
"""
import os
import re
import sys
import csv
import json
import time
import subprocess
import warnings
import logging
from collections import Counter
from itertools import combinations

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PAIRS_DIR = os.path.join(RESULTS_DIR, "pairs")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "splink_bestconfig_eval.csv")

DATA_ROOT = "/home/it2022025/er_scalability/datasets"
SPLIT_ROOT = "/home/it2022025/er_scalability/train_validation_test_sets"

SEED = 42  

# Ground-truth pair files, needed for B-cubed (the workers only scored on
# test_set.csv). Paths/sep/header mirror the proven pyjedai eval config.
GT = {
    "D2":   dict(path=f"{DATA_ROOT}/D2/gt.csv",        sep="|", header=0),
    "D3":   dict(path=f"{DATA_ROOT}/D3/gt.csv",        sep="#", header=0),
    "D4":   dict(path=f"{DATA_ROOT}/D4/gt.csv",        sep="%", header=0),
    "D5":   dict(path=f"{DATA_ROOT}/D5/gt.csv",        sep="|", header=0),
    "D6":   dict(path=f"{DATA_ROOT}/D6/gt.csv",        sep="|", header=0),
    "D7":   dict(path=f"{DATA_ROOT}/D7/gt.csv",        sep="|", header=0),
    "D8":   dict(path=f"{DATA_ROOT}/D8/gt.csv",        sep="|", header=0),
    "D9":   dict(path=f"{DATA_ROOT}/D9/gt.csv",        sep=">", header=0),
    "CORA": dict(path=f"{DATA_ROOT}/cora/cora_gt.csv", sep="|", header=None),
    "CDDB": dict(path=f"{DATA_ROOT}/CDDB/gt.csv",      sep=",", header=0),
}

# Source each GT column belongs to, per CCER dataset: (alias for col0, col1).
# D4 (dblp,acm) and D8 (walmart,amazon) list sources in the opposite order to
# the Splink worker aliases, so GT ids must be tagged with their true source.
GT_ORDER = {
    "D2": ("abt", "buy"), "D3": ("amazon", "gp"), "D4": ("dblp", "acm"),
    "D5": ("imdb", "tmdb"), "D6": ("imdb", "tvdb"), "D7": ("tmdb", "tvdb"),
    "D8": ("walmart", "amazon"), "D9": ("dblp", "scholar"),
}

CCER_DATASETS = ["D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
DER_DATASETS = ["CORA", "CDDB"]
ALL_DATASETS = CCER_DATASETS + DER_DATASETS


# ===========================================================================
# Generic metric helpers (identical to the pyjedai eval so numbers are
# directly comparable across tools)
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


def fill_singletons(entity_to_label, universe):
    """Ensure every universe entity has a label; missing ones become singletons.
    Labels may be Splink's string cluster_ids, so give singletons a tuple label
    (unique per entity, can't collide with any cluster_id)."""
    m = dict(entity_to_label)
    for e in universe:
        if e not in m:
            m[e] = ("__singleton__", e)
    return m


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


def pairwise_from_scoremap(test_df, score_map, threshold):
    """Pairwise P/R/F1 on the test split -- same logic as your Splink workers
    (score_map holds both (l,r) and (r,l), so direction doesn't matter)."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    scores = [score_map.get((str(row["left_id"]), str(row["right_id"])), 0.0)
              for _, row in test_df.iterrows()]
    y_true = test_df["label"].astype(int).values
    y_pred = [1 if s >= threshold else 0 for s in scores]
    return (precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0))


def check_gt_connects(ds, n_gt_total, n_gt_connected):
    """Guard: if GT pairs don't reference known entities, B-cubed is meaningless
    (recall collapses). Fail loudly instead of reporting junk."""
    if n_gt_total == 0:
        raise RuntimeError(f"{ds}: ground-truth file loaded 0 pairs -- check gt path/format.")
    frac = n_gt_connected / n_gt_total
    if frac < 0.5:
        raise RuntimeError(
            f"{ds}: only {n_gt_connected}/{n_gt_total} GT pairs match dataset ids "
            f"({frac:.1%}). GT ids don't line up with the entity universe -- check "
            f"gt sep/header/column order in GT[...]. Refusing to report bogus B-cubed.")
    sys.stderr.write(f"[{ds}] GT: {n_gt_connected}/{n_gt_total} pairs connect "
                     f"({frac:.1%}).\n")


def load_gt_pairs(ds):
    """Load ground-truth match pairs as a list of (left, right) string tuples.
    First two columns are the id pair (col0 = dataset-1 id for CCER)."""
    import pandas as pd
    g = GT[ds]
    df = pd.read_csv(g["path"], sep=g["sep"], header=g["header"], engine="python", dtype=str)
    df = df.fillna("")
    if df.shape[1] < 2:
        raise RuntimeError(
            f"{ds}: GT file {g['path']} parsed into {df.shape[1]} column with sep={g['sep']!r}. "
            f"Wrong delimiter -- fix GT[{ds!r}]['sep'].")
    left_col, right_col = df.columns[0], df.columns[1]
    return [(str(a), str(b)) for a, b in zip(df[left_col], df[right_col])]


# ===========================================================================
# Per-dataset feature engineering helpers (copied verbatim from the workers)
# ===========================================================================
def jw_levels(strictness):
    s = float(strictness)
    lv = sorted({round(min(0.98, s), 3),
                 round(min(0.98, s - 0.07), 3),
                 round(min(0.98, s - 0.15), 3)}, reverse=True)
    return [x for x in lv if x > 0]


# --- D2 ---
def d2_extract_brand(name):
    name = str(name).lower().strip()
    w = name.split()
    return w[0] if w else ""


def d2_extract_model(name):
    name = str(name).lower().strip()
    if " - " in name:
        last = name.rsplit(" - ", 1)[-1].strip()
        last = re.sub(r"[^a-z0-9]", "", last)
        if last and re.search(r"\d", last):
            return last
    tokens = re.sub(r"[^a-z0-9\s]", " ", name).split()
    dt = [t for t in tokens if re.search(r"\d", t)]
    return dt[-1] if dt else ""


# --- D3 ---
def d3_normalize_token(text):
    text = re.sub(
        r'\b(inc\.?|llc\.?|corp\.?|ltd\.?|systems|software|technologies|usa)\b',
        '', str(text).lower()).strip()
    words = text.split()
    return words[0] if words else ""


def d3_extract_version(title):
    m = re.search(r'\b(v\.?\s*\d+[\.\d]*|\b20\d{2}\b|\b19\d{2}\b|\d+\.\d+)\b', str(title).lower())
    return re.sub(r'\s+', '', m.group(0)).replace('v.', 'v') if m else ""


# --- D4 / D9 ---
def first_author_token(authors):
    authors = str(authors).strip()
    first = re.split(r'[,;]', authors)[0].strip()
    tokens = re.sub(r"[^a-z\s]", "", first.lower()).split()
    return tokens[-1] if tokens else ""


# --- D5 / D6 / D7 (movie/tv) ---
def movie_clean_text(val):
    import pandas as pd
    return str(val).lower().strip() if pd.notna(val) and str(val).strip() else ""


def movie_extract_year_zero(val):   # D5 variant: missing -> "0"
    import pandas as pd
    s = str(val).strip() if pd.notna(val) else ""
    m = re.match(r"(\d{4})", s)
    return m.group(1) if m else "0"


def movie_extract_year_none(val):   # D6 / D7 variant: missing -> None (NULL)
    import pandas as pd
    s = str(val).strip() if pd.notna(val) else ""
    m = re.match(r"(\d{4})", s)
    return m.group(1) if m else None


def make_ep_key(season, episode):
    import pandas as pd
    if pd.notna(season) and pd.notna(episode):
        return f"s{int(season)}e{int(episode)}"
    return None


# --- D8 ---
def d8_clean_text(val):
    import pandas as pd
    return str(val).lower().strip() if pd.notna(val) and str(val).strip() else ""


def d8_clean_modelno(val):
    import pandas as pd
    s = str(val).lower().strip() if pd.notna(val) else ""
    return re.sub(r"[^a-z0-9]", "", s)


# --- CORA / CDDB (unidecode) ---
def uni_clean_text(text):
    from unidecode import unidecode
    if text is None or str(text).strip() in ("", "nan"):
        return None
    text = unidecode(str(text)).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def uni_extract_year(text):
    if text is None:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", str(text))
    return m.group(0) if m else None


# ===========================================================================
# Per-dataset preparation -> (sources, aliases, settings). Each mirrors the
# corresponding worker exactly, up to (not including) u/m estimation.
# ===========================================================================
def _read_splits(dbdir):
    import pandas as pd
    train = pd.read_csv(f"{SPLIT_ROOT}/{dbdir}/train_set.csv")
    test = pd.read_csv(f"{SPLIT_ROOT}/{dbdir}/test_set.csv")
    for df in (train, test):
        df["left_id"] = df["left_id"].astype(str)
        df["right_id"] = df["right_id"].astype(str)
    return train, test


def prepare_D2(cfg):
    import pandas as pd
    import splink.comparison_library as cl
    from splink import SettingsCreator, block_on
    abt = pd.read_csv(f"{DATA_ROOT}/D2/abt.csv", delimiter="|")
    buy = pd.read_csv(f"{DATA_ROOT}/D2/buy.csv", delimiter="|")
    for df in (abt, buy):
        df["id"] = df["id"].astype(str)
        df["name"] = df["name"].fillna("").str.lower().str.strip()
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        df["brand"] = df["name"].apply(d2_extract_brand)
        df["model"] = df["name"].apply(d2_extract_model)
    abt = abt.rename(columns={"id": "unique_id"})
    buy = buy.rename(columns={"id": "unique_id"})
    keep = ["unique_id", "name", "brand", "model", "price"]
    abt_s, buy_s = abt[keep].copy(), buy[keep].copy()
    abt_s["source_dataset"], buy_s["source_dataset"] = "abt", "buy"
    jw = jw_levels(cfg["comparison_strictness"])
    settings = SettingsCreator(
        link_type="link_only",
        comparisons=[cl.JaroWinklerAtThresholds("name", jw),
                     cl.ExactMatch("brand"), cl.ExactMatch("model")],
        blocking_rules_to_generate_predictions=[
            block_on("brand", "model"), block_on("model"),
            block_on("brand", "substr(name, 1, 3)"), block_on("substr(name, 1, 5)")],
        retain_intermediate_calculation_columns=False, retain_matching_columns=False)
    train, test = _read_splits("db2")
    return dict(sources=[abt_s, buy_s], aliases=["abt", "buy"], settings=settings,
                train=train, test=test)


def prepare_D3(cfg):
    import pandas as pd
    import splink.comparison_library as cl
    from splink import SettingsCreator, block_on
    amazon = pd.read_csv(f"{DATA_ROOT}/D3/amazon.csv", delimiter="#")
    gp = pd.read_csv(f"{DATA_ROOT}/D3/gp.csv", delimiter="#")
    for df in (amazon, gp):
        df["id"] = df["id"].astype(str)
        df["title"] = df["title"].fillna("").str.lower().str.strip()
        df["manufacturer"] = df["manufacturer"].fillna("").str.lower().str.strip()
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        df["version"] = df["title"].apply(d3_extract_version)
    amazon["mfr_block"] = amazon["manufacturer"].apply(d3_normalize_token)
    gp["mfr_block"] = gp["title"].apply(d3_normalize_token)
    amazon = amazon.rename(columns={"id": "unique_id"})
    gp = gp.rename(columns={"id": "unique_id"})
    keep = ["unique_id", "title", "mfr_block", "version", "price"]
    amazon_s, gp_s = amazon[keep].copy(), gp[keep].copy()
    amazon_s["source_dataset"], gp_s["source_dataset"] = "amazon", "gp"
    jw = jw_levels(cfg["comparison_strictness"])
    settings = SettingsCreator(
        link_type="link_only",
        comparisons=[cl.JaroWinklerAtThresholds("title", jw),
                     cl.ExactMatch("mfr_block"), cl.ExactMatch("version")],
        blocking_rules_to_generate_predictions=[
            block_on("mfr_block", "version"), block_on("version"),
            block_on("mfr_block", "substr(title, 1, 3)"), block_on("substr(title, 1, 5)")],
        retain_intermediate_calculation_columns=False, retain_matching_columns=False)
    train, test = _read_splits("db3")
    return dict(sources=[amazon_s, gp_s], aliases=["amazon", "gp"], settings=settings,
                train=train, test=test)


def prepare_D4(cfg):
    import pandas as pd
    import splink.comparison_library as cl
    from splink import SettingsCreator, block_on
    acm = pd.read_csv(f"{DATA_ROOT}/D4/acm.csv", delimiter="%")
    dblp = pd.read_csv(f"{DATA_ROOT}/D4/dblp.csv", delimiter="%")
    for df in (acm, dblp):
        df["id"] = df["id"].astype(str)
        df["title"] = df["title"].fillna("").str.lower().str.strip()
        df["authors"] = df["authors"].fillna("").str.lower().str.strip()
        df["venue"] = df["venue"].fillna("").str.lower().str.strip()
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int).astype(str)
        df["first_author"] = df["authors"].apply(first_author_token)
    acm = acm.rename(columns={"id": "unique_id"})
    dblp = dblp.rename(columns={"id": "unique_id"})
    keep = ["unique_id", "title", "authors", "venue", "year", "first_author"]
    acm_s, dblp_s = acm[keep].copy(), dblp[keep].copy()
    acm_s["source_dataset"], dblp_s["source_dataset"] = "acm", "dblp"
    jw = jw_levels(cfg["comparison_strictness"])
    settings = SettingsCreator(
        link_type="link_only",
        comparisons=[cl.JaroWinklerAtThresholds("title", jw),
                     cl.JaroWinklerAtThresholds("authors", jw),
                     cl.JaroWinklerAtThresholds("venue", jw), cl.ExactMatch("year")],
        blocking_rules_to_generate_predictions=[
            block_on("year", "first_author"), block_on("first_author"),
            block_on("year", "substr(title, 1, 3)"), block_on("substr(title, 1, 5)")],
        retain_intermediate_calculation_columns=False, retain_matching_columns=False)
    train, test = _read_splits("db4")
    return dict(sources=[acm_s, dblp_s], aliases=["acm", "dblp"], settings=settings,
                train=train, test=test)


def _prepare_movies(cfg, dbdir, p1, cols1, yr1, a1, p2, cols2, yr2, a2, year_fn):
    import pandas as pd
    import splink.comparison_library as cl
    from splink import SettingsCreator, block_on
    df1 = pd.read_csv(p1, delimiter="|")
    df2 = pd.read_csv(p2, delimiter="|")
    df1.columns = cols1
    df2.columns = cols2
    for df in (df1, df2):
        df["id"] = df["id"].astype(str)
        df["title"] = df["title"].apply(movie_clean_text)
        df["name"] = df["name"].apply(movie_clean_text)
        df["entity_text"] = df.apply(lambda r: r["title"] if r["title"] else r["name"], axis=1)
        df["episodeNumber"] = pd.to_numeric(df["episodeNumber"], errors="coerce")
        df["seasonNumber"] = pd.to_numeric(df["seasonNumber"], errors="coerce")
        df["ep_key"] = df.apply(lambda r: make_ep_key(r["seasonNumber"], r["episodeNumber"]), axis=1)
    df1["year"] = df1[yr1].apply(year_fn)
    df2["year"] = df2[yr2].apply(year_fn)
    df1 = df1.rename(columns={"id": "unique_id"})
    df2 = df2.rename(columns={"id": "unique_id"})
    keep = ["unique_id", "entity_text", "year", "ep_key"]
    s1, s2 = df1[keep].copy(), df2[keep].copy()
    s1["source_dataset"], s2["source_dataset"] = a1, a2
    jw = jw_levels(cfg["comparison_strictness"])
    settings = SettingsCreator(
        link_type="link_only",
        comparisons=[cl.JaroWinklerAtThresholds("entity_text", jw),
                     cl.ExactMatch("year"), cl.ExactMatch("ep_key")],
        blocking_rules_to_generate_predictions=[
            block_on("year", "substr(entity_text, 1, 3)"),
            block_on("year", "substr(entity_text, 1, 5)"),
            block_on("ep_key"), block_on("substr(entity_text, 1, 8)")],
        retain_intermediate_calculation_columns=False, retain_matching_columns=False)
    train, test = _read_splits(dbdir)
    return dict(sources=[s1, s2], aliases=[a1, a2], settings=settings, train=train, test=test)


IMDB_COLS = ["id", "title", "name", "episodeNumber", "seasonNumber",
             "deathYear", "birthYear", "endYear", "startYear",
             "genre_list", "primaryProfessions", "runtimeMinutes"]
TMDB_COLS = ["id", "title", "name", "abstract", "episodeNumber", "seasonNumber",
             "numberOfSeasons", "numberOfEpisodes", "birthDate", "releaseDate",
             "last_air_date", "release_year", "runtime", "genre_list", "origin_country"]
TVDB_COLS = ["id", "title", "name", "abstract",
             "episodeNumber", "seasonNumber", "releaseDate", "job"]


def prepare_D5(cfg):
    return _prepare_movies(
        cfg, "db5",
        f"{DATA_ROOT}/D5/imdb.csv", IMDB_COLS, "startYear", "imdb",
        f"{DATA_ROOT}/D5/tmdb.csv", TMDB_COLS, "release_year", "tmdb",
        movie_extract_year_zero)


def prepare_D6(cfg):
    return _prepare_movies(
        cfg, "db6",
        f"{DATA_ROOT}/D6/imdb.csv", IMDB_COLS, "startYear", "imdb",
        f"{DATA_ROOT}/D6/tvdb.csv", TVDB_COLS, "releaseDate", "tvdb",
        movie_extract_year_none)


def prepare_D7(cfg):
    return _prepare_movies(
        cfg, "db7",
        f"{DATA_ROOT}/D7/tmdb.csv", TMDB_COLS, "releaseDate", "tmdb",
        f"{DATA_ROOT}/D7/tvdb.csv", TVDB_COLS, "releaseDate", "tvdb",
        movie_extract_year_none)


def prepare_D8(cfg):
    import pandas as pd
    import splink.comparison_library as cl
    from splink import SettingsCreator, block_on
    amazon = pd.read_csv(f"{DATA_ROOT}/D8/amazon.csv", delimiter="|")
    walmart = pd.read_csv(f"{DATA_ROOT}/D8/walmart.csv", delimiter="|")
    for df in (amazon, walmart):
        df["id"] = df["id"].astype(str)
        df["title"] = df["title"].apply(d8_clean_text)
        df["brand"] = df["brand"].apply(d8_clean_text)
        df["modelno"] = df["modelno"].apply(d8_clean_modelno)
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    amazon = amazon.rename(columns={"id": "unique_id"})
    walmart = walmart.rename(columns={"id": "unique_id"})
    keep = ["unique_id", "title", "brand", "modelno", "price"]
    amazon_s, walmart_s = amazon[keep].copy(), walmart[keep].copy()
    amazon_s["source_dataset"], walmart_s["source_dataset"] = "amazon", "walmart"
    jw = jw_levels(cfg["comparison_strictness"])
    settings = SettingsCreator(
        link_type="link_only",
        comparisons=[cl.JaroWinklerAtThresholds("title", jw),
                     cl.ExactMatch("brand"), cl.ExactMatch("modelno")],
        blocking_rules_to_generate_predictions=[
            block_on("brand", "modelno"), block_on("modelno"),
            block_on("brand", "substr(title, 1, 3)"), block_on("substr(title, 1, 5)")],
        retain_intermediate_calculation_columns=False, retain_matching_columns=False)
    train, test = _read_splits("db8")
    return dict(sources=[amazon_s, walmart_s], aliases=["amazon", "walmart"],
                settings=settings, train=train, test=test)


def prepare_D9(cfg):
    import pandas as pd
    import splink.comparison_library as cl
    from splink import SettingsCreator, block_on
    dblp = pd.read_csv(f"{DATA_ROOT}/D9/dblp.csv", delimiter=">")
    scholar = pd.read_csv(f"{DATA_ROOT}/D9/scholar.csv", delimiter=">")
    for df in (dblp, scholar):
        df["id"] = df["id"].astype(str)
        df["title"] = df["title"].fillna("").str.lower().str.strip()
        df["authors"] = df["authors"].fillna("").str.lower().str.strip()
        df["venue"] = df["venue"].fillna("").str.lower().str.strip()
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int).astype(str)
        df["first_author"] = df["authors"].apply(first_author_token)
    dblp = dblp.rename(columns={"id": "unique_id"})
    scholar = scholar.rename(columns={"id": "unique_id"})
    keep = ["unique_id", "title", "authors", "venue", "year", "first_author"]
    dblp_s, scholar_s = dblp[keep].copy(), scholar[keep].copy()
    dblp_s["source_dataset"], scholar_s["source_dataset"] = "dblp", "scholar"
    jw = jw_levels(cfg["comparison_strictness"])
    settings = SettingsCreator(
        link_type="link_only",
        comparisons=[cl.JaroWinklerAtThresholds("title", jw),
                     cl.JaroWinklerAtThresholds("authors", jw),
                     cl.JaroWinklerAtThresholds("venue", jw), cl.ExactMatch("year")],
        blocking_rules_to_generate_predictions=[
            block_on("year", "first_author"), block_on("first_author"),
            block_on("year", "substr(title, 1, 3)"), block_on("substr(title, 1, 5)")],
        retain_intermediate_calculation_columns=False, retain_matching_columns=False)
    train, test = _read_splits("db9")
    return dict(sources=[dblp_s, scholar_s], aliases=["dblp", "scholar"],
                settings=settings, train=train, test=test)


def prepare_CORA(cfg):
    import pandas as pd
    import splink.comparison_library as cl
    from splink import SettingsCreator, block_on
    cora = pd.read_csv(f"{DATA_ROOT}/cora/cora.csv", sep="|", engine="python", na_filter=False)
    for col in ["title", "author", "venue", "publisher", "year"]:
        cora[col] = cora[col].apply(uni_clean_text)
    cora = cora.rename(columns={"Entity Id": "unique_id"})
    cora["unique_id"] = cora["unique_id"].astype(str)
    cora["first_author"] = cora["author"].apply(lambda x: x.split()[0] if x else None)
    cora["year_from_title"] = cora["title"].apply(uni_extract_year)
    cora["year_clean"] = cora["year"].where(cora["year"].notna(), cora["year_from_title"])
    keep = ["unique_id", "title", "author", "first_author", "venue", "publisher", "year_clean"]
    cora_s = cora[keep].copy()
    jw = jw_levels(cfg["comparison_strictness"])
    jw_author = [round(x, 3) for x in jw]
    jw_venue = [round(x, 3) for x in jw[:2]] if len(jw) >= 2 else jw
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[cl.JaroWinklerAtThresholds("title", jw),
                     cl.JaroWinklerAtThresholds("author", jw_author),
                     cl.JaroWinklerAtThresholds("venue", jw_venue),
                     cl.ExactMatch("publisher"), cl.ExactMatch("year_clean")],
        blocking_rules_to_generate_predictions=[
            block_on("first_author"), block_on("year_clean"),
            block_on("substr(title, 1, 3)"), block_on("substr(title, 1, 5)"),
            block_on("substr(title, 1, 8)"), block_on("substr(author, 1, 4)"),
            block_on("substr(venue, 1, 4)"), block_on("publisher"),
            block_on("substr(title, 1, 10)")],
        retain_intermediate_calculation_columns=False, retain_matching_columns=False)
    train, test = _read_splits("cora")
    return dict(sources=[cora_s], aliases=None, settings=settings, train=train, test=test)


def prepare_CDDB(cfg):
    import pandas as pd
    import splink.comparison_library as cl
    from splink import SettingsCreator, block_on
    cddb = pd.read_csv(f"{DATA_ROOT}/CDDB/cddb.csv", engine="python", na_filter=False)
    for col in ["artist", "title", "genre", "category", "year"]:
        cddb[col] = cddb[col].apply(uni_clean_text)
    cddb = cddb.rename(columns={"id": "unique_id"})
    cddb["unique_id"] = cddb["unique_id"].astype(str)
    cddb["first_artist"] = cddb["artist"].apply(lambda x: x.split()[0] if x else None)
    keep = ["unique_id", "artist", "title", "first_artist", "genre", "category", "year"]
    cddb_s = cddb[keep].copy()
    jw = jw_levels(cfg["comparison_strictness"])
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[cl.JaroWinklerAtThresholds("artist", jw),
                     cl.JaroWinklerAtThresholds("title", jw),
                     cl.ExactMatch("genre"), cl.ExactMatch("category"), cl.ExactMatch("year")],
        blocking_rules_to_generate_predictions=[
            block_on("first_artist"), block_on("year"),
            block_on("substr(title, 1, 2)"), block_on("substr(title, 1, 3)"),
            block_on("substr(title, 1, 5)"), block_on("substr(artist, 1, 2)"),
            block_on("substr(artist, 1, 4)"), block_on("genre"), block_on("category")],
        retain_intermediate_calculation_columns=False, retain_matching_columns=False)
    train, test = _read_splits("cddb")
    return dict(sources=[cddb_s], aliases=None, settings=settings, train=train, test=test)


PREPARE = {
    "D2": prepare_D2, "D3": prepare_D3, "D4": prepare_D4, "D5": prepare_D5,
    "D6": prepare_D6, "D7": prepare_D7, "D8": prepare_D8, "D9": prepare_D9,
    "CORA": prepare_CORA, "CDDB": prepare_CDDB,
}


# ===========================================================================
# Shared Splink pipeline tail (u -> m -> patch rate -> predict) + both metrics
# ===========================================================================
def _train_linker(prep, cfg, cid, link_type):
    """Reproduce the workers' training: estimate u, supervised m from train labels,
    patch probability_two_random_records_match, reload from patched model."""
    import json as _json
    import numpy as np
    import pandas as pd
    from splink import DuckDBAPI, Linker
    np.random.seed(SEED)  # match the workers before random-sampling u
    sources = prep["sources"]
    settings = prep["settings"]
    train = prep["train"]
    pos = train[train["label"] == 1]

    db_api = DuckDBAPI()
    if link_type == "link_only":
        linker = Linker(sources, settings, db_api, input_table_aliases=prep["aliases"])
    else:
        linker = Linker(sources[0], settings, db_api)
    linker.training.estimate_u_using_random_sampling(
        max_pairs=float(cfg["estimate_u_max_pairs"]), seed=SEED)

    if link_type == "link_only":
        a1, a2 = prep["aliases"]
        labelled = pd.DataFrame({
            "source_dataset_l": a1, "unique_id_l": pos["left_id"].astype(str),
            "source_dataset_r": a2, "unique_id_r": pos["right_id"].astype(str)})
        linker.table_management.register_table(labelled, "labels", overwrite=True)
        linker.training.estimate_m_from_pairwise_labels("labels")
        total_possible = len(sources[0]) * len(sources[1])
    else:
        labelled = pd.DataFrame({
            "unique_id_l": pos["left_id"].astype(str),
            "unique_id_r": pos["right_id"].astype(str),
            "clerical_match_score": 1.0})
        labels_sdf = linker.table_management.register_labels_table(labelled, overwrite=True)
        linker.training.estimate_m_from_pairwise_labels(labels_sdf)
        n = len(sources[0])
        total_possible = n * (n - 1) / 2

    rate = int(train["label"].sum()) / total_possible
    m_path = f"/tmp/splink_eval_m_cfg{cid}.json"
    m_patched = f"/tmp/splink_eval_m_patched_cfg{cid}.json"
    linker.misc.save_model_to_json(m_path, overwrite=True)
    with open(m_path) as f:
        mj = _json.load(f)
    mj["probability_two_random_records_match"] = rate
    with open(m_patched, "w") as f:
        _json.dump(mj, f)

    db_api2 = DuckDBAPI()
    if link_type == "link_only":
        linker = Linker(sources, m_patched, db_api2, input_table_aliases=prep["aliases"])
    else:
        linker = Linker(sources[0], m_patched, db_api2)
    for pth in (m_path, m_patched):
        try:
            os.remove(pth)
        except OSError:
            pass
    return linker


def run_dataset(ds, cfg, cid):
    """Full evaluation for one dataset: pairwise cross-check + Splink-clustered B-cubed."""
    prep = PREPARE[ds](cfg)
    link_type = "dedupe_only" if ds in DER_DATASETS else "link_only"
    t = float(cfg["chosen_threshold"])

    linker = _train_linker(prep, cfg, cid, link_type)

    # --- inference (SplinkDataFrame kept for the Splink clusterer) ---
    df_predict = linker.inference.predict(threshold_match_probability=0.0)
    rdf = df_predict.as_pandas_dataframe()
    score_map = {}
    for _, row in rdf.iterrows():
        l, r, p = str(row["unique_id_l"]), str(row["unique_id_r"]), float(row["match_probability"])
        score_map[(l, r)] = p
        score_map[(r, l)] = p

    # --- pairwise cross-check (same as workers) ---
    pw_p, pw_r, pw_f = pairwise_from_scoremap(prep["test"], score_map, t)

    # --- predicted match pairs at the chosen threshold (for the dump) ---
    kept = rdf[rdf["match_probability"] >= t]
    predicted_pairs = [(str(a), str(b))
                       for a, b in zip(kept["unique_id_l"], kept["unique_id_r"])]

    # --- CLUSTER LEVEL: Splink's own clusterer at the chosen threshold ---
    clusters_sdf = linker.clustering.cluster_pairwise_predictions_at_threshold(
        df_predict, threshold_match_probability=t)
    cdf = clusters_sdf.as_pandas_dataframe()

    if link_type == "link_only":
        a1, a2 = prep["aliases"]
        d1_ids = prep["sources"][0]["unique_id"].astype(str).tolist()
        d2_ids = prep["sources"][1]["unique_id"].astype(str).tolist()
        universe = [f"{a1}::{x}" for x in d1_ids] + [f"{a2}::{x}" for x in d2_ids]
        has_sd = "source_dataset" in cdf.columns
        entity_to_pred = {}
        for _, row in cdf.iterrows():
            sd = str(row["source_dataset"]) if has_sd else a1
            entity_to_pred[f"{sd}::{str(row['unique_id'])}"] = row["cluster_id"]
        g0, g1 = GT_ORDER[ds]  # tag GT with the source each column truly belongs to
        gt_pairs = [(f"{g0}::{a}", f"{g1}::{b}") for a, b in load_gt_pairs(ds)]
    else:
        ids = prep["sources"][0]["unique_id"].astype(str).tolist()
        universe = list(ids)
        entity_to_pred = {str(row["unique_id"]): row["cluster_id"]
                          for _, row in cdf.iterrows()}
        gt_pairs = [(str(a), str(b)) for a, b in load_gt_pairs(ds)]

    universe_set = set(universe)
    n_gt_connected = sum(1 for a, b in gt_pairs if a in universe_set and b in universe_set)
    check_gt_connects(ds, len(gt_pairs), n_gt_connected)

    entity_to_pred = fill_singletons(entity_to_pred, universe)
    entity_to_gt = connected_components(gt_pairs, universe)
    b_p, b_r, b_f = bcubed(entity_to_pred, entity_to_gt)

    return dict(pairwise=(pw_p, pw_r, pw_f), bcubed=(b_p, b_r, b_f),
                n_entities=len(universe), n_pred_pairs=len(predicted_pairs),
                n_gt_pairs=len(gt_pairs), dump_pairs=predicted_pairs, entities=universe)


# ===========================================================================
# Best-config lookup
# ===========================================================================
def read_best_config(ds):
    """Return (params_dict, config_id, csv_test_f1) for the max-test_f1 OK row."""
    path = os.path.join(RESULTS_DIR, f"splink_{ds}_configs.csv")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("status") == "OK" and r.get("test_f1") not in (None, "")]
    if not rows:
        raise RuntimeError(f"{ds}: no OK rows in {path}")
    best = max(rows, key=lambda r: float(r["test_f1"]))
    params = dict(
        comparison_strictness=float(best["comparison_strictness"]),
        estimate_u_max_pairs=float(best["estimate_u_max_pairs"]),
        chosen_threshold=float(best["chosen_threshold"]),
    )
    return params, best["config_id"], float(best["test_f1"])


# ===========================================================================
# Single-dataset driver (subprocess entry point)
# ===========================================================================
def run_single(ds):
    cfg, config_id, csv_test_f1 = read_best_config(ds)
    t0 = time.time()
    out = run_dataset(ds, cfg, config_id)

    os.makedirs(PAIRS_DIR, exist_ok=True)
    with open(os.path.join(PAIRS_DIR, f"splink_{ds}_pred_pairs.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["left_id", "right_id"])
        wtr.writerows(out["dump_pairs"])
    with open(os.path.join(PAIRS_DIR, f"splink_{ds}_entities.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["entity_id"])
        wtr.writerows([[e] for e in out["entities"]])

    pw, bc = out["pairwise"], out["bcubed"]
    result = {
        "dataset": ds, "config_id": config_id,
        "family": ("der" if ds in DER_DATASETS else "ccer"),
        "csv_test_f1": round(csv_test_f1, 6),
        "pairwise_precision": round(pw[0], 6), "pairwise_recall": round(pw[1], 6),
        "pairwise_f1": round(pw[2], 6),
        "bcubed_precision": round(bc[0], 6), "bcubed_recall": round(bc[1], 6),
        "bcubed_f1": round(bc[2], 6),
        "n_entities": out["n_entities"], "n_pred_pairs": out["n_pred_pairs"],
        "n_gt_pairs": out["n_gt_pairs"], "time_sec": round(time.time() - t0, 2),
    }
    print("RESULT_JSON:" + json.dumps(result))
    return result


# ===========================================================================
# All-datasets driver
# ===========================================================================
SUMMARY_COLS = ["dataset", "config_id", "family", "csv_test_f1",
                "pairwise_precision", "pairwise_recall", "pairwise_f1",
                "bcubed_precision", "bcubed_recall", "bcubed_f1",
                "n_entities", "n_pred_pairs", "n_gt_pairs", "time_sec", "status"]


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for ds in ALL_DATASETS:
        print(f"\n=== {ds} ===", flush=True)
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), ds],
                              capture_output=True, text=True)
        result = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                result = json.loads(line[len("RESULT_JSON:"):])
        if result is None:
            print(f"  -> ERROR\n{proc.stderr[-1200:]}", flush=True)
            rows.append({"dataset": ds, "status": "ERROR"})
        else:
            result["status"] = "OK"
            print(f"  -> pairwise F1={result['pairwise_f1']:.4f}  "
                  f"B3 F1={result['bcubed_f1']:.4f}  "
                  f"(P={result['bcubed_precision']:.3f} R={result['bcubed_recall']:.3f})  "
                  f"cfg#{result['config_id']}  {result['time_sec']}s", flush=True)
            rows.append(result)
        with open(SUMMARY_CSV, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
            wtr.writeheader()
            for r in rows:
                wtr.writerow({c: r.get(c) for c in SUMMARY_COLS})
    print(f"\nDone. Wrote {SUMMARY_CSV}")


def main():
    if len(sys.argv) > 1:
        ds = sys.argv[1]
        if ds not in PREPARE:
            sys.exit(f"Unknown dataset '{ds}'. Choose from: {', '.join(ALL_DATASETS)}")
        run_single(ds)
    else:
        run_all()


if __name__ == "__main__":
    main()
