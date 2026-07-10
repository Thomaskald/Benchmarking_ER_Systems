import sys
import os
import re
import json
import time
import resource
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

TMDB_PATH  = "/home/it2022025/er_scalability/datasets/D7/tmdb.csv"
TVDB_PATH  = "/home/it2022025/er_scalability/datasets/D7/tvdb.csv"
TRAIN_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/db7/train_set.csv"
VALID_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/db7/valid_set.csv"
TEST_PATH  = "/home/it2022025/er_scalability/train_validation_test_sets/db7/test_set.csv"

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)

def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def clean_text(val):
    return str(val).lower().strip() if pd.notna(val) and str(val).strip() else ""

def extract_year(val):
    s = str(val).strip() if pd.notna(val) else ""
    m = re.match(r"(\d{4})", s)
    return m.group(1) if m else None  # NULL so missing years don't block

def make_ep_key(season, episode):
    if pd.notna(season) and pd.notna(episode):
        return f"s{int(season)}e{int(episode)}"
    return None  # NULL so DuckDB blocking skips missing values

def jw_levels(strictness):
    s = float(strictness)
    lv = sorted({round(min(0.98, s), 3),
                 round(min(0.98, s - 0.07), 3),
                 round(min(0.98, s - 0.15), 3)}, reverse=True)
    return [x for x in lv if x > 0]


def main():
    cfg = json.loads(sys.argv[1])
    np.random.seed(int(cfg.get("seed", 42)))
    t_start = time.time()
    cid = cfg.get("config_id", 0)

    tmdb_df = pd.read_csv(TMDB_PATH, delimiter="|")
    tvdb_df = pd.read_csv(TVDB_PATH, delimiter="|")
    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Rename long URI column headers to short names
    tmdb_df.columns = [
        "id", "title", "name", "abstract", "episodeNumber", "seasonNumber",
        "numberOfSeasons", "numberOfEpisodes", "birthDate", "releaseDate",
        "last_air_date", "release_year", "runtime", "genre_list", "origin_country",
    ]
    tvdb_df.columns = [
        "id", "title", "name", "abstract",
        "episodeNumber", "seasonNumber", "releaseDate", "job",
    ]

    for df in [tmdb_df, tvdb_df]:
        df["id"] = df["id"].astype(str)
        df["title"] = df["title"].apply(clean_text)
        df["name"] = df["name"].apply(clean_text)
        # unified text: prefer title, fall back to name
        df["entity_text"] = df.apply(
            lambda r: r["title"] if r["title"] else r["name"], axis=1
        )
        df["episodeNumber"] = pd.to_numeric(df["episodeNumber"], errors="coerce")
        df["seasonNumber"] = pd.to_numeric(df["seasonNumber"], errors="coerce")
        df["ep_key"] = df.apply(
            lambda r: make_ep_key(r["seasonNumber"], r["episodeNumber"]), axis=1
        )
        df["year"] = df["releaseDate"].apply(extract_year)

    tmdb_df = tmdb_df.rename(columns={"id": "unique_id"})
    tvdb_df = tvdb_df.rename(columns={"id": "unique_id"})
    for df in [train_df, valid_df, test_df]:
        df["left_id"] = df["left_id"].astype(str)
        df["right_id"] = df["right_id"].astype(str)

    keep = ["unique_id", "entity_text", "year", "ep_key"]
    tmdb_s = tmdb_df[keep].copy()
    tvdb_s = tvdb_df[keep].copy()
    tmdb_s["source_dataset"] = "tmdb"
    tvdb_s["source_dataset"] = "tvdb"

    jw = jw_levels(cfg["comparison_strictness"])
    settings = SettingsCreator(
        link_type="link_only",
        comparisons=[
            cl.JaroWinklerAtThresholds("entity_text", jw),
            cl.ExactMatch("year"),
            cl.ExactMatch("ep_key"),
        ],
        blocking_rules_to_generate_predictions=[
            block_on("year", "substr(entity_text, 1, 3)"),
            block_on("year", "substr(entity_text, 1, 5)"),
            block_on("ep_key"),  # only fires for non-NULL ep_key (actual episodes)
            block_on("substr(entity_text, 1, 8)"),
        ],
        retain_intermediate_calculation_columns=False,
        retain_matching_columns=False,
    )
    db_api = DuckDBAPI()
    linker = Linker([tmdb_s, tvdb_s], settings, db_api,
                    input_table_aliases=["tmdb", "tvdb"])

    linker.training.estimate_u_using_random_sampling(max_pairs=float(cfg["estimate_u_max_pairs"]))

    pos = train_df[train_df["label"] == 1]
    labelled = pd.DataFrame({
        "source_dataset_l": "tmdb",
        "unique_id_l": pos["left_id"].astype(str),
        "source_dataset_r": "tvdb",
        "unique_id_r": pos["right_id"].astype(str),
    })
    linker.table_management.register_table(labelled, "labels", overwrite=True)
    linker.training.estimate_m_from_pairwise_labels("labels")

    total_possible = len(tmdb_s) * len(tvdb_s)
    n_true = int(train_df["label"].sum())
    rate = n_true / total_possible
    # per-config temp model files (avoid any cross-config cl: unique by config_id)
    m_path = f"/tmp/splink_m_cfg{cid}.json"
    m_patched = f"/tmp/splink_m_patched_cfg{cid}.json"
    linker.misc.save_model_to_json(m_path, overwrite=True)
    with open(m_path) as f:
        mj = json.load(f)
    mj["probability_two_random_records_match"] = rate
    with open(m_patched, "w") as f:
        json.dump(mj, f)
    db_api2 = DuckDBAPI()
    linker = Linker([tmdb_s, tvdb_s], m_patched, db_api2,
                    input_table_aliases=["tmdb", "tvdb"])

    res = linker.inference.predict(threshold_match_probability=0.0)
    rdf = res.as_pandas_dataframe()
    score_map = {}
    for _, row in rdf.iterrows():
        l, r, p = str(row["unique_id_l"]), str(row["unique_id_r"]), float(row["match_probability"])
        score_map[(l, r)] = p
        score_map[(r, l)] = p

    def scores_for(split_df):
        return np.array([score_map.get((str(row["left_id"]), str(row["right_id"])), 0.0)
                         for _, row in split_df.iterrows()])

    valid_scores = scores_for(valid_df)
    test_scores = scores_for(test_df)
    y_valid = valid_df["label"].values
    y_test = test_df["label"].values

    best_t, best_valid_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        preds = (valid_scores >= t).astype(int)
        f1 = f1_score(y_valid, preds, zero_division=0)
        if f1 > best_valid_f1:
            best_valid_f1, best_t = f1, float(t)

    preds_test = (test_scores >= best_t).astype(int)
    test_p = precision_score(y_test, preds_test, zero_division=0)
    test_r = recall_score(y_test, preds_test, zero_division=0)
    test_f1 = f1_score(y_test, preds_test, zero_division=0)

    curve = []
    for t in THRESHOLD_GRID:
        preds = (test_scores >= t).astype(int)
        curve.append({"t": round(float(t), 3),
                      "precision": round(precision_score(y_test, preds, zero_division=0), 6),
                      "recall": round(recall_score(y_test, preds, zero_division=0), 6),
                      "f1": round(f1_score(y_test, preds, zero_division=0), 6)})

    result = {
        "config_id": cid,
        "params": {"comparison_strictness": cfg["comparison_strictness"],
                   "estimate_u_max_pairs": cfg["estimate_u_max_pairs"],
                   "jw_levels": jw},
        "status": "OK",
        "chosen_threshold": round(best_t, 3),
        "valid_f1_at_threshold": round(best_valid_f1, 6),
        "test_point": {"t": round(best_t, 3), "precision": round(test_p, 6),
                       "recall": round(test_r, 6), "f1": round(test_f1, 6)},
        "pr_curve": curve,
        "time_sec": round(time.time() - t_start, 2),
        "peak_mem_mb": round(peak_mem_mb(), 1),
    }
    # cleanup temp files
    for pth in (m_path, m_patched):
        try: os.remove(pth)
        except OSError: pass
    print("RESULT_JSON:" + json.dumps(result))

if __name__ == "__main__":
    main()