#!/usr/bin/env python3
"""
Best-config evaluation at TWO levels: pairwise + test-set B-cubed -- SPLINK.

Both metrics are computed on the fixed test set, uniform with the DEDUPE,
MAGELLAN, RECORDLINKAGE and pyJedAI evals (test-set B-cubed, so it always
finishes within the time budget and shares the pairwise universe). For each
dataset it:

  1. reads the BEST config (max test_f1 among status==OK rows) straight from the
     existing results/splink_<DS>_configs.csv,
  2. reruns exactly that one config's Splink pipeline (same steps as your workers:
     estimate u -> supervised m from train labels -> patch match rate -> predict),
  3. scores the fixed test pairs once (used for BOTH metrics),
  4. reports PAIRWISE P/R/F1 on the test split (matches splink_<DS>_configs.csv),
  5. reports TEST-SET B-cubed P/R/F1: predicted clusters = connected components of
     test pairs scoring >= threshold; true clusters = connected components of test
     pairs with label == 1; B-cubed over the test-set entities,
  6. dumps predicted match pairs + the test-set entity universe under results/pairs/.

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
  results/pairs/splink_<DS>_entities.csv       test-set entity id universe (tagged)
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

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PAIRS_DIR = os.path.join(RESULTS_DIR, "pairs")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "splink_bestconfig_eval.csv")

DATA_ROOT = "/home/it2022025/er_scalability/datasets"
SPLIT_ROOT = "/home/it2022025/er_scalability/train_validation_test_sets"

SEED = 42

CCER_DATASETS = ["D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
DER_DATASETS = ["CORA", "CDDB"]
ALL_DATASETS = CCER_DATASETS + DER_DATASETS


# ===========================================================================
# Test-set metric helpers (identical to the DEDUPE / MAGELLAN / RECORDLINKAGE
# evals, so the B-cubed numbers are uniform across frameworks)
# ===========================================================================
def norm_id(v):
    """Canonical string id. '123', 123, '123.0' -> '123'. Keeps non-numeric as-is."""
    s = str(v).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


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


def testset_metrics(test_index, probs, labels, threshold, tag_left, tag_right):
    """Pairwise + test-set B-cubed over the fixed test pairs (uniform with the
    DEDUPE / MAGELLAN / RECORDLINKAGE / pyJedAI evals):
      - predicted clusters = connected components of test pairs scored >= threshold
      - true clusters      = connected components of test pairs with label == 1
      - both metrics over the entities that appear in the test set.
    tag_left/tag_right disambiguate the two id spaces (CCER: 'A:' / 'B:';
    single-source DER: '' / '')."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    universe, pred_edges, true_edges, pred_pairs_out = set(), [], [], []
    y_pred = []
    for (na, nb), s, y in zip(test_index, probs, labels):
        la, rb = f"{tag_left}{norm_id(na)}", f"{tag_right}{norm_id(nb)}"
        universe.add(la)
        universe.add(rb)
        matched = s >= threshold
        y_pred.append(1 if matched else 0)
        if matched:
            pred_edges.append((la, rb))
            pred_pairs_out.append((norm_id(na), norm_id(nb)))
        if int(y) == 1:
            true_edges.append((la, rb))
    universe = list(universe)
    P, R, F = bcubed(connected_components(pred_edges, universe),
                     connected_components(true_edges, universe))
    pw = (precision_score(labels, y_pred, zero_division=0),
          recall_score(labels, y_pred, zero_division=0),
          f1_score(labels, y_pred, zero_division=0))
    return dict(pairwise=pw, bcubed=(P, R, F), n_entities=len(universe),
                n_pred_pairs=len(pred_pairs_out), n_gt_pairs=len(true_edges),
                dump_pairs=sorted(set(pred_pairs_out)), entities=universe)


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
    """Full evaluation for one dataset: run the best config's Splink pipeline,
    then pairwise + test-set B-cubed over the fixed test pairs."""
    prep = PREPARE[ds](cfg)
    link_type = "dedupe_only" if ds in DER_DATASETS else "link_only"
    t = float(cfg["chosen_threshold"])

    linker = _train_linker(prep, cfg, cid, link_type)

    # inference -> score_map over predicted pairs (both directions, as the workers)
    df_predict = linker.inference.predict(threshold_match_probability=0.0)
    rdf = df_predict.as_pandas_dataframe()
    score_map = {}
    for _, row in rdf.iterrows():
        l, r, p = str(row["unique_id_l"]), str(row["unique_id_r"]), float(row["match_probability"])
        score_map[(l, r)] = p
        score_map[(r, l)] = p

    # score the fixed test pairs, then compute both metrics on the test set
    test_df = prep["test"]
    test_index = list(zip(test_df["left_id"].astype(str), test_df["right_id"].astype(str)))
    probs = [score_map.get((a, b), 0.0) for a, b in test_index]
    labels = test_df["label"].astype(int).tolist()
    tag_left, tag_right = ("A:", "B:") if link_type == "link_only" else ("", "")
    return testset_metrics(test_index, probs, labels, t, tag_left, tag_right)


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
