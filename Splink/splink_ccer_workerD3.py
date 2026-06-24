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

AMAZON_PATH = "/home/it2022025/er_scalability/datasets/D3/amazon.csv"
GP_PATH     = "/home/it2022025/er_scalability/datasets/D3/gp.csv"
TRAIN_PATH  = "/home/it2022025/er_scalability/train_validation_test_sets/db3/train_set.csv"
VALID_PATH  = "/home/it2022025/er_scalability/train_validation_test_sets/db3/valid_set.csv"
TEST_PATH   = "/home/it2022025/er_scalability/train_validation_test_sets/db3/test_set.csv"

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)

def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def normalize_token(text):
    text = re.sub(
        r'\b(inc\.?|llc\.?|corp\.?|ltd\.?|systems|software|technologies|usa)\b',
        '', str(text).lower()
    ).strip()
    words = text.split()
    return words[0] if words else ""

def extract_version(title):
    m = re.search(
        r'\b(v\.?\s*\d+[\.\d]*|\b20\d{2}\b|\b19\d{2}\b|\d+\.\d+)\b',
        str(title).lower()
    )
    return re.sub(r'\s+', '', m.group(0)).replace('v.', 'v') if m else ""

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

    amazon_df = pd.read_csv(AMAZON_PATH, delimiter="#")
    gp_df = pd.read_csv(GP_PATH, delimiter="#")
    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)

    for df in [amazon_df, gp_df]:
        df["id"] = df["id"].astype(str)
        df["title"] = df["title"].fillna("").str.lower().str.strip()
        df["manufacturer"] = df["manufacturer"].fillna("").str.lower().str.strip()
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        df["version"] = df["title"].apply(extract_version)

    # mfr_block: Amazon -> normalized manufacturer; GP -> normalized first word
    # of title (manufacturer is ~94% null on GP)
    amazon_df["mfr_block"] = amazon_df["manufacturer"].apply(normalize_token)
    gp_df["mfr_block"] = gp_df["title"].apply(normalize_token)

    amazon_df = amazon_df.rename(columns={"id": "unique_id"})
    gp_df = gp_df.rename(columns={"id": "unique_id"})
    for df in [train_df, valid_df, test_df]:
        df["left_id"] = df["left_id"].astype(str)
        df["right_id"] = df["right_id"].astype(str)

    keep = ["unique_id", "title", "mfr_block", "version", "price"]
    amazon_s = amazon_df[keep].copy()
    gp_s = gp_df[keep].copy()
    amazon_s["source_dataset"] = "amazon"
    gp_s["source_dataset"] = "gp"

    jw = jw_levels(cfg["comparison_strictness"])
    settings = SettingsCreator(
        link_type="link_only",
        comparisons=[
            cl.JaroWinklerAtThresholds("title", jw),
            cl.ExactMatch("mfr_block"),
            cl.ExactMatch("version"),
        ],
        blocking_rules_to_generate_predictions=[
            block_on("mfr_block", "version"),
            block_on("version"),
            block_on("mfr_block", "substr(title, 1, 3)"),
            block_on("substr(title, 1, 5)"),
        ],
        retain_intermediate_calculation_columns=False,
        retain_matching_columns=False,
    )
    db_api = DuckDBAPI()
    linker = Linker([amazon_s, gp_s], settings, db_api,
                    input_table_aliases=["amazon", "gp"])

    linker.training.estimate_u_using_random_sampling(max_pairs=float(cfg["estimate_u_max_pairs"]))

    pos = train_df[train_df["label"] == 1]
    labelled = pd.DataFrame({
        "source_dataset_l": "amazon",
        "unique_id_l": pos["left_id"].astype(str),
        "source_dataset_r": "gp",
        "unique_id_r": pos["right_id"].astype(str),
    })
    linker.table_management.register_table(labelled, "labels", overwrite=True)
    linker.training.estimate_m_from_pairwise_labels("labels")

    total_possible = len(amazon_s) * len(gp_s)
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
    linker = Linker([amazon_s, gp_s], m_patched, db_api2,
                    input_table_aliases=["amazon", "gp"])

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