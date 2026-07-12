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

# Synthetic FEBRL 10K (scalability tuning size). Single-source dirty ER.
PROFILES_PATH = "/home/it2022025/er_scalability/converted/10K/profiles.csv"
TRAIN_PATH    = "/home/it2022025/er_scalability/splits/10K/train_set.csv"
VALID_PATH    = "/home/it2022025/er_scalability/splits/10K/valid_set.csv"
TEST_PATH     = "/home/it2022025/er_scalability/splits/10K/test_set.csv"

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def clean_text(text):
    if text is None or str(text).strip() in ("", "nan"):
        return None
    text = re.sub(r"\s+", " ", str(text).lower()).strip()
    return text if text else None


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

    df = pd.read_csv(PROFILES_PATH, engine="python", na_filter=False)  # comma-delimited
    for col in ["given_name", "surname", "address_1", "suburb", "postcode", "state", "date_of_birth", "soc_sec_id", "phone_number"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    df = df.rename(columns={"id": "unique_id"})
    df["unique_id"] = df["unique_id"].astype(str)

    # derived blocking features (from the scalability script)
    df["surname_2"]  = df["surname"].str[:2]
    df["surname_4"]  = df["surname"].str[:4]
    df["postcode_3"] = df["postcode"].str[:3]

    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)
    for d in [train_df, valid_df, test_df]:
        d["left_id"] = d["left_id"].astype(str)
        d["right_id"] = d["right_id"].astype(str)
    y_valid = valid_df["label"].values
    y_test = test_df["label"].values

    # comparison_strictness drives ALL JaroWinkler comparison levels (searched)
    jw = jw_levels(cfg["comparison_strictness"])

    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            cl.JaroWinklerAtThresholds("given_name", jw),
            cl.JaroWinklerAtThresholds("surname", jw),
            cl.JaroWinklerAtThresholds("address_1", jw),
            cl.ExactMatch("postcode"),
            cl.ExactMatch("suburb"),
            cl.ExactMatch("state"),
            cl.ExactMatch("date_of_birth"),
            cl.ExactMatch("soc_sec_id"),
            cl.ExactMatch("phone_number"),
        ],
        blocking_rules_to_generate_predictions=[
            block_on("surname_2"),
            block_on("surname_4"),
            block_on("postcode"),
            block_on("postcode_3"),
            block_on("suburb"),
            block_on("given_name"),
        ],
        retain_intermediate_calculation_columns=False,
        retain_matching_columns=False,
    )
    db_api = DuckDBAPI()
    linker = Linker(df, settings, db_api)

    linker.training.estimate_u_using_random_sampling(max_pairs=float(cfg["estimate_u_max_pairs"]))

    # supervised m from TRAIN labels only (three-way split: valid stays held out)
    pos = train_df[train_df["label"] == 1]
    labelled = pd.DataFrame({
        "unique_id_l": pos["left_id"].astype(str),
        "unique_id_r": pos["right_id"].astype(str),
        "clerical_match_score": 1.0,
    })
    labels_sdf = linker.table_management.register_labels_table(labelled, overwrite=True)
    linker.training.estimate_m_from_pairwise_labels(labels_sdf)

    total_possible = len(df) * (len(df) - 1) / 2
    n_true = int(train_df["label"].sum())
    rate = n_true / total_possible
    m_path = f"/tmp/splink_scale10k_m_cfg{cid}.json"
    m_patched = f"/tmp/splink_scale10k_m_patched_cfg{cid}.json"
    linker.misc.save_model_to_json(m_path, overwrite=True)
    with open(m_path) as f:
        mj = json.load(f)
    mj["probability_two_random_records_match"] = rate
    with open(m_patched, "w") as f:
        json.dump(mj, f)
    db_api2 = DuckDBAPI()
    linker = Linker(df, m_patched, db_api2)

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
    for pth in (m_path, m_patched):
        try: os.remove(pth)
        except OSError: pass
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()