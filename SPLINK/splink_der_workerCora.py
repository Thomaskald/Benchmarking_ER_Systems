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
from unidecode import unidecode
from sklearn.metrics import precision_score, recall_score, f1_score

import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

CORA_PATH  = "/home/it2022025/er_scalability/datasets/cora/cora.csv"
TRAIN_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/cora/train_set.csv"
VALID_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/cora/valid_set.csv"
TEST_PATH  = "/home/it2022025/er_scalability/train_validation_test_sets/cora/test_set.csv"

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def clean_text(text):
    if text is None or str(text).strip() in ("", "nan"):
        return None
    text = unidecode(str(text)).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def extract_year(text):
    if text is None:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", str(text))
    return m.group(0) if m else None


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

    cora = pd.read_csv(CORA_PATH, sep="|", engine="python", na_filter=False)
    for col in ["title", "author", "venue", "publisher", "year"]:
        cora[col] = cora[col].apply(clean_text)
    cora = cora.rename(columns={"Entity Id": "unique_id"})
    cora["unique_id"] = cora["unique_id"].astype(str)
    cora["first_author"] = cora["author"].apply(lambda x: x.split()[0] if x else None)
    cora["year_from_title"] = cora["title"].apply(extract_year)
    cora["year_clean"] = cora["year"].where(cora["year"].notna(), cora["year_from_title"])

    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)
    for df in [train_df, valid_df, test_df]:
        df["left_id"] = df["left_id"].astype(str)
        df["right_id"] = df["right_id"].astype(str)
    y_valid = valid_df["label"].values
    y_test = test_df["label"].values

    keep = ["unique_id", "title", "author", "first_author", "venue", "publisher", "year_clean"]
    cora_s = cora[keep].copy()

    # comparison_strictness drives ALL JaroWinkler comparison levels (searched)
    jw = jw_levels(cfg["comparison_strictness"])
    jw_author = [round(x, 3) for x in jw]      # same searched levels reused
    jw_venue = [round(x, 3) for x in jw[:2]] if len(jw) >= 2 else jw

    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            cl.JaroWinklerAtThresholds("title", jw),
            cl.JaroWinklerAtThresholds("author", jw_author),
            cl.JaroWinklerAtThresholds("venue", jw_venue),
            cl.ExactMatch("publisher"),
            cl.ExactMatch("year_clean"),
        ],
        blocking_rules_to_generate_predictions=[
            block_on("first_author"),
            block_on("year_clean"),
            block_on("substr(title, 1, 3)"),
            block_on("substr(title, 1, 5)"),
            block_on("substr(title, 1, 8)"),
            block_on("substr(author, 1, 4)"),
            block_on("substr(venue, 1, 4)"),
            block_on("publisher"),
            block_on("substr(title, 1, 10)"),
        ],
        retain_intermediate_calculation_columns=False,
        retain_matching_columns=False,
    )
    db_api = DuckDBAPI()
    linker = Linker(cora_s, settings, db_api)

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

    total_possible = len(cora_s) * (len(cora_s) - 1) / 2
    n_true = int(train_df["label"].sum())
    rate = n_true / total_possible
    m_path = f"/tmp/splink_cora_m_cfg{cid}.json"
    m_patched = f"/tmp/splink_cora_m_patched_cfg{cid}.json"
    linker.misc.save_model_to_json(m_path, overwrite=True)
    with open(m_path) as f:
        mj = json.load(f)
    mj["probability_two_random_records_match"] = rate
    with open(m_patched, "w") as f:
        json.dump(mj, f)
    db_api2 = DuckDBAPI()
    linker = Linker(cora_s, m_patched, db_api2)

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