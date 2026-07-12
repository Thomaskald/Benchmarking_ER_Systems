"""
splink_scalability_fixed.py
---------------------------
Fixed-config scalability runs for Splink on synthetic FEBRL datasets.
Uses the single BEST config chosen from the 10K B=50 search (by valid_f1),
applied verbatim to every size -- INCLUDING a frozen decision threshold,
exactly like pyjedai_scalability_fixed.py. No per-dataset threshold sweep,
no valid set at scale.

    BEST config (10K search, config #21):
        comparison_strictness = 0.8973
        estimate_u_max_pairs  = 123494
        threshold             = 0.06     (valid_f1 = 0.973695)

10K is excluded (it was the tuning size).
Loops: 50K, 100K, 200K, 300K, 1M, 2M -- one serial job, checkpoint after each.
Metrics per size: runtime, peak memory (VmHWM), and test P/R/F1 (scored over the
split-file labelled pairs, consistent with all other tools).
Failure policy: OOM / ERROR recorded per dataset, not crashes.

Output: splink_scalability_results.csv
"""

import os
import re
import json
import time
import traceback
import warnings
import logging

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

CONVERTED_DIR = "/home/it2022025/er_scalability/converted"
SPLITS_DIR    = "/home/it2022025/er_scalability/splits"
OUTPUT_CSV    = "/home/it2022025/er_scalability/scalability/splink_scalability_results.csv"

# 10K excluded (tuning size). Rest of the ladder, looped internally:
DATASETS = ["50K", "100K", "200K", "300K", "1M", "2M"]

# ---- BEST CONFIG from the 10K B=50 search (fixed for all sizes) ----
# Selected by argmax(valid_f1)=0.973695 -> config #21 (tie with #25 on valid_f1,
# broken by lowest config_id; both had threshold 0.06 so the pick is immaterial).
BEST_STRICTNESS  = 0.8973
BEST_U_MAX_PAIRS = 123494.0
BEST_THRESHOLD   = 0.06      # frozen from 10K -- NOT re-tuned per dataset


def mem_mb():
    return psutil.Process().memory_info().rss / 1024 ** 2

def peak_mem_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return mem_mb()

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


def run_pipeline(df, train_df, test_df, ds_label):
    """Fit m/u on this dataset's TRAIN with the frozen config, score TEST at the
    frozen threshold. No valid set, no sweep (mirrors pyjedai/dedupe fixed)."""
    workflow_start = time.time()
    jw = jw_levels(BEST_STRICTNESS)

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
    linker = Linker(df, settings, DuckDBAPI())
    linker.training.estimate_u_using_random_sampling(max_pairs=BEST_U_MAX_PAIRS)

    # supervised m from TRAIN labels only
    pos = train_df[train_df["label"] == 1]
    labelled = pd.DataFrame({
        "unique_id_l": pos["left_id"].astype(str),
        "unique_id_r": pos["right_id"].astype(str),
        "clerical_match_score": 1.0,
    })
    labels_sdf = linker.table_management.register_labels_table(labelled, overwrite=True)
    linker.training.estimate_m_from_pairwise_labels(labels_sdf)

    # patch the base match rate from TRAIN, then reload
    total_possible = len(df) * (len(df) - 1) / 2
    rate = int(train_df["label"].sum()) / total_possible
    m_path    = f"/tmp/splink_fixed_m_{ds_label}.json"
    m_patched = f"/tmp/splink_fixed_m_patched_{ds_label}.json"
    linker.misc.save_model_to_json(m_path, overwrite=True)
    with open(m_path) as f:
        mj = json.load(f)
    mj["probability_two_random_records_match"] = rate
    with open(m_patched, "w") as f:
        json.dump(mj, f)
    linker = Linker(df, m_patched, DuckDBAPI())

    # predict at the frozen operating threshold (realistic + memory-bounded;
    # pairs below threshold are non-matches anyway -> identical final preds)
    res = linker.inference.predict(threshold_match_probability=BEST_THRESHOLD)
    rdf = res.as_pandas_dataframe()
    score_map = {}
    for _, row in rdf.iterrows():
        l, r, p = str(row["unique_id_l"]), str(row["unique_id_r"]), float(row["match_probability"])
        score_map[(l, r)] = p
        score_map[(r, l)] = p

    def scores_for(split_df):
        return np.array([score_map.get((str(row["left_id"]), str(row["right_id"])), 0.0)
                         for _, row in split_df.iterrows()])

    test_scores = scores_for(test_df)
    preds_test  = (test_scores >= BEST_THRESHOLD).astype(int)
    y_test = test_df["label"].values
    test_p  = precision_score(y_test, preds_test, zero_division=0)
    test_r  = recall_score(y_test, preds_test, zero_division=0)
    test_f1 = f1_score(y_test, preds_test, zero_division=0)

    for pth in (m_path, m_patched):
        try: os.remove(pth)
        except OSError: pass

    return test_p, test_r, test_f1, time.time() - workflow_start


# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------

print("\n" + "=" * 60)
print("  SPLINK SCALABILITY (fixed best config)")
print(f"  strictness={BEST_STRICTNESS} | u_max_pairs={BEST_U_MAX_PAIRS} | thr={BEST_THRESHOLD}")
print("=" * 60)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
results = []

for ds in DATASETS:
    profiles_path = os.path.join(CONVERTED_DIR, ds, "profiles.csv")
    train_path    = os.path.join(SPLITS_DIR,    ds, "train_set.csv")
    test_path     = os.path.join(SPLITS_DIR,    ds, "test_set.csv")

    print(f"\n{'='*60}\n  Dataset: {ds}\n{'='*60}", flush=True)

    if not (os.path.exists(profiles_path) and os.path.exists(train_path)
            and os.path.exists(test_path)):
        print(f"  [SKIP] missing files for {ds}")
        results.append({"dataset": ds, "status": "SKIP: file not found"})
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        continue

    try:
        t_load = time.time()
        df = pd.read_csv(profiles_path, engine="python", na_filter=False)  # comma-delimited
        for col in ["given_name", "surname", "address_1", "suburb", "postcode",
                    "state", "date_of_birth", "soc_sec_id", "phone_number"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)
        df = df.rename(columns={"id": "unique_id"})
        df["unique_id"] = df["unique_id"].astype(str)
        df["surname_2"]  = df["surname"].str[:2]
        df["surname_4"]  = df["surname"].str[:4]
        df["postcode_3"] = df["postcode"].str[:3]

        train_df = pd.read_csv(train_path, engine="python")
        test_df  = pd.read_csv(test_path,  engine="python")
        for d in [train_df, test_df]:
            d["left_id"]  = d["left_id"].astype(str)
            d["right_id"] = d["right_id"].astype(str)
        t_load = time.time() - t_load

        print(f"  Records    : {len(df):,}")
        print(f"  Test pairs : {len(test_df):,}  (pos={int(test_df['label'].sum()):,})")
        print(f"  Load time  : {t_load:.2f}s  |  mem: {mem_mb():.0f} MB", flush=True)

        test_p, test_r, test_f1, t_workflow = run_pipeline(df, train_df, test_df, ds)
        peak = peak_mem_mb()

        print(f"\n  --- RESULTS ---")
        print(f"  Precision  : {test_p:.4f}  Recall: {test_r:.4f}  F1: {test_f1:.4f}")
        print(f"  Workflow   : {t_workflow:.2f}s  ({t_workflow/60:.1f} min)")
        print(f"  Peak mem   : {peak:.0f} MB", flush=True)

        results.append({
            "dataset": ds, "n_records": len(df), "n_test_pairs": len(test_df),
            "precision": round(test_p, 4), "recall": round(test_r, 4), "f1": round(test_f1, 4),
            "time_load": round(t_load, 2), "time_workflow": round(t_workflow, 2),
            "peak_mem_mb": round(peak, 1), "status": "OK",
        })

    except MemoryError:
        print(f"  [OOM] {ds}", flush=True)
        results.append({"dataset": ds, "status": "OOM"})
    except Exception as e:
        print(f"  [ERROR] {ds}: {e}", flush=True)
        traceback.print_exc()
        results.append({"dataset": ds, "status": f"FAILED: {str(e)[:200]}"})

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"  [checkpoint saved]", flush=True)

pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print("\n" + "=" * 60)
print("  SCALABILITY SUMMARY")
print("=" * 60)
dfr = pd.DataFrame(results)
cols = [c for c in ["dataset","n_records","precision","recall","f1","time_workflow","peak_mem_mb","status"] if c in dfr.columns]
print(dfr[cols].to_string(index=False))
print(f"\nSaved to: {OUTPUT_CSV}")
