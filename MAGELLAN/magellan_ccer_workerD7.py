import sys
import json
import time
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
from sklearn.metrics.pairwise import cosine_similarity

TMDB_PATH  = "/home/it2022025/er_scalability/datasets/D7/tmdb.csv"
TVDB_PATH  = "/home/it2022025/er_scalability/datasets/D7/tvdb.csv"
TRAIN_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/db7/train_set.csv"
VALID_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/db7/valid_set.csv"
TEST_PATH  = "/home/it2022025/er_scalability/train_validation_test_sets/db7/test_set.csv"

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# --- D7-specific preprocessing: load ontology-URL CSV into clean columns ---
def load_movie_table(path):
    raw = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8")
    title_col = "https://www.scads.de/movieBenchmark/ontology/title"
    name_col = "https://www.scads.de/movieBenchmark/ontology/name"
    abstract_col = "http://dbpedia.org/ontology/abstract"
    episode_col = "http://dbpedia.org/ontology/episodeNumber"
    season_col = "http://dbpedia.org/ontology/seasonNumber"
    release_col = "http://dbpedia.org/ontology/releaseDate"
    df = pd.DataFrame()
    df["id"] = pd.to_numeric(raw["id"], errors="coerce")
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    title = raw.get(title_col)
    name = raw.get(name_col)
    df["title"] = (title if title is not None else name)
    df["name"] = name if name is not None else ""
    df["abstract"] = raw.get(abstract_col, "")
    df["episodeNumber"] = raw.get(episode_col, 0)
    df["seasonNumber"] = raw.get(season_col, 0)
    df["releaseDate"] = raw.get(release_col, "")
    df = df.reset_index(drop=True)
    return df


def build_candset(csv_path, tmdb, tvdb):
    df = pd.read_csv(csv_path)
    df.insert(0, "_id", range(len(df)))
    em.set_key(df, "_id")
    em.set_ltable(df, tmdb)
    em.set_rtable(df, tvdb)
    em.set_fk_ltable(df, "left_id")
    em.set_fk_rtable(df, "right_id")
    return df


def tfidf_cosine(candset, tmdb, tvdb, tfidf=None):
    left_ids = candset["left_id"].values
    right_ids = candset["right_id"].values
    left_texts = (tmdb.loc[left_ids, "title"].values + " " +
                  tmdb.loc[left_ids, "name"].values + " " +
                  tmdb.loc[left_ids, "abstract"].values)
    right_texts = (tvdb.loc[right_ids, "title"].values + " " +
                   tvdb.loc[right_ids, "name"].values + " " +
                   tvdb.loc[right_ids, "abstract"].values)
    if tfidf is None:
        tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
        tfidf.fit(list(left_texts) + list(right_texts))
    sims = cosine_similarity(tfidf.transform(left_texts), tfidf.transform(right_texts)).diagonal()
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


def prep_matrix(candset, tmdb, tvdb, match_f, col_means, fitted_tfidf, feat_cols_final):
    H = em.extract_feature_vecs(candset, feature_table=match_f, attrs_after=["label"], show_progress=False)
    meta = ["_id", "left_id", "right_id", "label"]
    for col in [c for c in H.columns if c not in meta]:
        H[col] = H[col].fillna(col_means.get(col, 0))
    sims, _ = tfidf_cosine(candset, tmdb, tvdb, tfidf=fitted_tfidf)
    H["tfidf_cosine"] = sims
    X = H[feat_cols_final].values
    y = H["label"].values
    return X, y


def main():
    cfg = json.loads(sys.argv[1])
    np.random.seed(int(cfg.get("seed", 42)))
    t_start = time.time()

    # --- D7-specific preprocessing ---
    tmdb = load_movie_table(TMDB_PATH)
    tvdb = load_movie_table(TVDB_PATH)
    em.set_key(tmdb, "id")
    em.set_key(tvdb, "id")
    for df in [tmdb, tvdb]:
        df["title"] = df["title"].fillna("").astype(str).str.lower().str.strip()
        df["name"] = df["name"].fillna("").astype(str).str.lower().str.strip()
        df["abstract"] = df["abstract"].fillna("").astype(str).str.lower().str.strip()
        df["releaseDate"] = df["releaseDate"].fillna("").astype(str).str.lower().str.strip()
    for df in [tmdb, tvdb]:
        df["episodeNumber"] = pd.to_numeric(df["episodeNumber"], errors="coerce").fillna(0)
        df["seasonNumber"] = pd.to_numeric(df["seasonNumber"], errors="coerce").fillna(0)
    # --- end D7-specific preprocessing ---

    train_cand = build_candset(TRAIN_PATH, tmdb, tvdb)
    valid_cand = build_candset(VALID_PATH, tmdb, tvdb)
    test_cand = build_candset(TEST_PATH, tmdb, tvdb)

    match_f = em.get_features_for_matching(tmdb, tvdb, validate_inferred_attr_types=False)
    id_cols = match_f[match_f["feature_name"].str.startswith("id_")].index
    match_f = match_f.drop(id_cols)

    H_train = em.extract_feature_vecs(train_cand, feature_table=match_f, attrs_after=["label"], show_progress=False)
    meta = ["_id", "left_id", "right_id", "label"]
    base_feat = [c for c in H_train.columns if c not in meta]
    col_means = H_train[base_feat].mean()
    for col in base_feat:
        H_train[col] = H_train[col].fillna(col_means.get(col, 0))
    tfidf_tr, fitted = tfidf_cosine(train_cand, tmdb, tvdb)
    H_train["tfidf_cosine"] = tfidf_tr
    feat_cols_final = [c for c in H_train.columns if c not in meta]

    X_train = H_train[feat_cols_final].values
    y_train = H_train["label"].values

    X_valid, y_valid = prep_matrix(valid_cand, tmdb, tvdb, match_f, col_means, fitted, feat_cols_final)
    X_test,  y_test  = prep_matrix(test_cand,  tmdb, tvdb, match_f, col_means, fitted, feat_cols_final)

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