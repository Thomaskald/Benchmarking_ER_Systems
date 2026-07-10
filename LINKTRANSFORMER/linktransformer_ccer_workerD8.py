import sys
import os
import json
import time
import resource
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import linktransformer as lt

AMAZON_PATH  = "/home/it2022025/er_scalability/datasets/D8/amazon.csv"
WALMART_PATH = "/home/it2022025/er_scalability/datasets/D8/walmart.csv"
TRAIN_PATH   = "/home/it2022025/er_scalability/train_validation_test_sets/db8/train_set.csv"
VALID_PATH   = "/home/it2022025/er_scalability/train_validation_test_sets/db8/valid_set.csv"
TEST_PATH    = "/home/it2022025/er_scalability/train_validation_test_sets/db8/test_set.csv"

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# NOTE: in D8 the split pairing is left_id -> walmart, right_id -> amazon.
def build_paired_df(split_df, walmart_idx, amazon_idx):
    df = split_df.copy()
    df["left_text"] = (walmart_idx.loc[df["left_id"].values, "title"].values + " " +
                       walmart_idx.loc[df["left_id"].values, "brand"].values + " " +
                       walmart_idx.loc[df["left_id"].values, "modelno"].values)
    df["right_text"] = (amazon_idx.loc[df["right_id"].values, "title"].values + " " +
                        amazon_idx.loc[df["right_id"].values, "brand"].values + " " +
                        amazon_idx.loc[df["right_id"].values, "modelno"].values)
    return df[["left_text", "right_text", "label"]]


def main():
    cfg = json.loads(sys.argv[1])
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    import random as _r; _r.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
    except Exception:
        pass
    t_start = time.time()

    amazon = pd.read_csv(AMAZON_PATH, delimiter="|")
    walmart = pd.read_csv(WALMART_PATH, delimiter="|")
    for df in [amazon, walmart]:
        df["title"] = df["title"].str.lower().str.strip().fillna("")
        df["brand"] = df["brand"].str.lower().str.strip().fillna("")
        df["modelno"] = df["modelno"].astype(str).str.lower().str.strip().fillna("")
    amazon["price"] = pd.to_numeric(amazon["price"], errors="coerce").fillna(0)
    walmart["price"] = pd.to_numeric(walmart["price"], errors="coerce").fillna(0)
    amazon_idx = amazon.set_index("id")
    walmart_idx = walmart.set_index("id")

    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)
    y_valid = valid_df["label"].values
    y_test = test_df["label"].values

    train_paired = build_paired_df(train_df, walmart_idx, amazon_idx)
    valid_paired = build_paired_df(valid_df, walmart_idx, amazon_idx)
    test_paired = build_paired_df(test_df, walmart_idx, amazon_idx)

    saved_model_path = lt.train_model(
        model_path=cfg["base_model"],
        train_data=train_paired,
        val_data=valid_paired,
        test_data=test_paired,
        left_col_names=["left_text"],
        right_col_names=["right_text"],
        label_col_name="label",
        training_args={
            "num_epochs": int(cfg["num_epochs"]),
            "train_batch_size": int(cfg["train_batch_size"]),
            "loss_type": cfg["loss_type"],
            "lr": float(cfg["learning_rate"]),
            "warmup_ratio": float(cfg["warm_up_perc"]),
        },
        log_wandb=False,
    )

    valid_ft = lt.evaluate_pairs(valid_paired.copy(), model=saved_model_path,
                                 left_on=["left_text"], right_on=["right_text"])
    test_ft = lt.evaluate_pairs(test_paired.copy(), model=saved_model_path,
                                left_on=["left_text"], right_on=["right_text"])
    valid_scores = valid_ft["score"].values
    test_scores = test_ft["score"].values

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

    pkeys = ["base_model", "loss_type", "num_epochs", "train_batch_size",
             "learning_rate", "warm_up_perc"]
    result = {
        "config_id": cfg.get("config_id"),
        "params": {k: cfg.get(k) for k in pkeys},
        "status": "OK",
        "chosen_threshold": round(best_t, 3),
        "valid_f1_at_threshold": round(best_valid_f1, 6),
        "test_point": {"t": round(best_t, 3), "precision": round(test_p, 6),
                       "recall": round(test_r, 6), "f1": round(test_f1, 6)},
        "pr_curve": curve,
        "time_sec": round(time.time() - t_start, 2),
        "peak_mem_mb": round(peak_mem_mb(), 1),
    }
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()