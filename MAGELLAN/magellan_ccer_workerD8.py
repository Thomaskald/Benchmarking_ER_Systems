import sys
import json
import time
import csv
import re
import resource
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import py_entitymatching as em
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

AMAZON_PATH  = "/home/it2022025/er_scalability/datasets/D8/amazon.csv"
WALMART_PATH = "/home/it2022025/er_scalability/datasets/D8/walmart.csv"
TRAIN_PATH   = "/home/it2022025/er_scalability/train_validation_test_sets/db8/train_set.csv"
VALID_PATH   = "/home/it2022025/er_scalability/train_validation_test_sets/db8/valid_set.csv"
TEST_PATH    = "/home/it2022025/er_scalability/train_validation_test_sets/db8/test_set.csv"

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def clean_text(text):
    if text is None:
        return None
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()
    return text if text else None


def pre_process_price(price):
    if price is None:
        return None
    text = str(price).replace("$", "").replace(",", "").strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def pre_process_weight(weight):
    if weight is None:
        return None
    text = str(weight).strip().lower().replace("lbs", "").replace("lb", "").strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_base_table(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            try:
                rec_id = int(str(row.get("id")).strip())
            except Exception:
                continue
            rows.append({
                "id": rec_id,
                "title": clean_text(row.get("title")),
                "modelno": clean_text(row.get("modelno")),
                "price": pre_process_price(row.get("price")),
                "shipweight": pre_process_weight(row.get("shipweight")),
                "brand": clean_text(row.get("brand")),
                "dimensions": clean_text(row.get("dimensions")),
            })
    return pd.DataFrame(
        rows,
        columns=["id", "title", "modelno", "price", "shipweight", "brand", "dimensions"],
    )


# left table = Walmart, right table = Amazon (per db8 split convention)
def build_candset(csv_path, walmart, amazon):
    df = pd.read_csv(csv_path)
    df.insert(0, "_id", range(len(df)))
    em.set_key(df, "_id")
    em.set_ltable(df, walmart)
    em.set_rtable(df, amazon)
    em.set_fk_ltable(df, "left_id")
    em.set_fk_rtable(df, "right_id")
    return df


def tfidf_cosine(candset, walmart, amazon, tfidf=None):
    left_ids = candset["left_id"].values
    right_ids = candset["right_id"].values
    left_texts = (walmart.loc[left_ids, "title"].values + " " +
                  walmart.loc[left_ids, "modelno"].values + " " +
                  walmart.loc[left_ids, "brand"].values + " " +
                  walmart.loc[left_ids, "dimensions"].values)
    right_texts = (amazon.loc[right_ids, "title"].values + " " +
                   amazon.loc[right_ids, "modelno"].values + " " +
                   amazon.loc[right_ids, "brand"].values + " " +
                   amazon.loc[right_ids, "dimensions"].values)
    if tfidf is None:
        tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
        tfidf.fit(list(left_texts) + list(right_texts))
    left_mat = tfidf.transform(left_texts)
    right_mat = tfidf.transform(right_texts)
    # Cosine only for corresponding rows (i,i), avoiding an NxN matrix.
    left_norm = normalize(left_mat, norm="l2", axis=1)
    right_norm = normalize(right_mat, norm="l2", axis=1)
    sims = left_norm.multiply(right_norm).sum(axis=1).A1
    return sims, tfidf


def make_classifier(cfg):
    m = cfg["matcher"]
    cw = None if cfg.get("class_weight") == "none" else "balanced"
    if m == "RandomForest":
        return RandomForestClassifier(n_estimators=int(cfg["n_estimators"]),
                                      max_depth=int(cfg["max_depth"]),
                                      class_weight=cw, random_state=42, n_jobs=-1)
    if m == "GradientBoosting":
        return GradientBoostingClassifier(n_estimators=int(cfg["n_estimators"]),
                                          max_depth=int(cfg["max_depth"]),
                                          learning_rate=float(cfg["learning_rate"]),
                                          random_state=42)
    if m == "DecisionTree":
        return DecisionTreeClassifier(max_depth=int(cfg["max_depth"]),
                                      class_weight=cw, random_state=42)
    if m == "LogisticRegression":
        return LogisticRegression(C=float(cfg["C"]), max_iter=1000, class_weight=cw)
    if m == "NaiveBayes":
        return GaussianNB()
    if m == "SVM":
        return SVC(C=float(cfg["C"]), kernel="rbf", probability=True,
                   class_weight=cw, random_state=42)
    raise ValueError(f"unknown matcher {m}")


def prep_matrix(candset, walmart, amazon, match_f, col_means, fitted_tfidf, feat_cols_final):
    H = em.extract_feature_vecs(candset, feature_table=match_f, attrs_after=["label"], show_progress=False)
    meta = ["_id", "left_id", "right_id", "label"]
    for col in [c for c in H.columns if c not in meta]:
        H[col] = H[col].fillna(col_means.get(col, 0))
    sims, _ = tfidf_cosine(candset, walmart, amazon, tfidf=fitted_tfidf)
    H["tfidf_cosine"] = sims
    X = H[feat_cols_final].values
    y = H["label"].values
    return X, y


def main():
    cfg = json.loads(sys.argv[1])
    np.random.seed(int(cfg.get("seed", 42)))
    t_start = time.time()

    amazon = load_base_table(AMAZON_PATH)
    walmart = load_base_table(WALMART_PATH)
    em.set_key(amazon, "id")
    em.set_key(walmart, "id")

    for df in [amazon, walmart]:
        df["title"] = df["title"].fillna("").astype(str).str.lower().str.strip()
        df["modelno"] = df["modelno"].fillna("").astype(str).str.lower().str.strip()
        df["brand"] = df["brand"].fillna("").astype(str).str.lower().str.strip()
        df["dimensions"] = df["dimensions"].fillna("").astype(str).str.lower().str.strip()
    for df in [amazon, walmart]:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
        df["shipweight"] = pd.to_numeric(df["shipweight"], errors="coerce").fillna(0)

    train_cand = build_candset(TRAIN_PATH, walmart, amazon)
    valid_cand = build_candset(VALID_PATH, walmart, amazon)
    test_cand = build_candset(TEST_PATH, walmart, amazon)

    match_f = em.get_features_for_matching(walmart, amazon, validate_inferred_attr_types=False)
    id_cols = match_f[match_f["feature_name"].str.startswith("id_")].index
    match_f = match_f.drop(id_cols)

    H_train = em.extract_feature_vecs(train_cand, feature_table=match_f, attrs_after=["label"], show_progress=False)
    meta = ["_id", "left_id", "right_id", "label"]
    base_feat = [c for c in H_train.columns if c not in meta]
    col_means = H_train[base_feat].mean()
    for col in base_feat:
        H_train[col] = H_train[col].fillna(col_means.get(col, 0))
    tfidf_tr, fitted = tfidf_cosine(train_cand, walmart, amazon)
    H_train["tfidf_cosine"] = tfidf_tr
    feat_cols_final = [c for c in H_train.columns if c not in meta]

    X_train = H_train[feat_cols_final].values
    y_train = H_train["label"].values

    X_valid, y_valid = prep_matrix(valid_cand, walmart, amazon, match_f, col_means, fitted, feat_cols_final)
    X_test,  y_test  = prep_matrix(test_cand,  walmart, amazon, match_f, col_means, fitted, feat_cols_final)

    clf = make_classifier(cfg)
    clf.fit(X_train, y_train)
    probs_valid = clf.predict_proba(X_valid)[:, 1]
    probs_test = clf.predict_proba(X_test)[:, 1]

    best_t, best_valid_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        preds = (probs_valid >= t).astype(int)
        f1 = f1_score(y_valid, preds, zero_division=0)
        if f1 > best_valid_f1:
            best_valid_f1, best_t = f1, float(t)

    preds_test = (probs_test >= best_t).astype(int)
    test_p = precision_score(y_test, preds_test, zero_division=0)
    test_r = recall_score(y_test, preds_test, zero_division=0)
    test_f1 = f1_score(y_test, preds_test, zero_division=0)

    curve = []
    for t in THRESHOLD_GRID:
        preds = (probs_test >= t).astype(int)
        curve.append({"t": round(float(t), 3),
                      "precision": round(precision_score(y_test, preds, zero_division=0), 6),
                      "recall": round(recall_score(y_test, preds, zero_division=0), 6),
                      "f1": round(f1_score(y_test, preds, zero_division=0), 6)})

    pkeys = ["matcher", "n_estimators", "max_depth", "learning_rate", "C", "class_weight"]
    result = {
        "config_id": cfg.get("config_id"),
        "params": {k: cfg.get(k) for k in pkeys if k in cfg},
        "status": "OK",
        "chosen_threshold": round(best_t, 3),
        "valid_f1_at_threshold": round(best_valid_f1, 6),
        "test_point": {"t": round(best_t, 3),
                       "precision": round(test_p, 6),
                       "recall": round(test_r, 6),
                       "f1": round(test_f1, 6)},
        "pr_curve": curve,
        "time_sec": round(time.time() - t_start, 2),
        "peak_mem_mb": round(peak_mem_mb(), 1),
    }
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()