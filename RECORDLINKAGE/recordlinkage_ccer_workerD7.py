import sys
import json
import time
import resource
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import recordlinkage
from recordlinkage.preprocessing import clean
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

TMDB_PATH  = "/home/it2022025/er_scalability/datasets/D7/tmdb.csv"
TVDB_PATH  = "/home/it2022025/er_scalability/datasets/D7/tvdb.csv"
TRAIN_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/db7/train_set.csv"
VALID_PATH = "/home/it2022025/er_scalability/train_validation_test_sets/db7/valid_set.csv"
TEST_PATH  = "/home/it2022025/er_scalability/train_validation_test_sets/db7/test_set.csv"

THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)

def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def rowwise_cosine(left_matrix, right_matrix):
    dot = np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).ravel()
    ln = np.sqrt(np.asarray(left_matrix.multiply(left_matrix).sum(axis=1)).ravel())
    rn = np.sqrt(np.asarray(right_matrix.multiply(right_matrix).sum(axis=1)).ravel())
    denom = ln * rn
    denom[denom == 0] = 1.0
    return dot / denom

def get_texts(df, ids, cols):
    return (df.loc[ids, cols].fillna("").astype(str).apply(" ".join, axis=1).to_numpy())

def build_tfidf_sim(pairs, left_df, right_df, vec, cols, label):
    lv = vec.transform(get_texts(left_df, pairs.get_level_values(0), cols))
    rv = vec.transform(get_texts(right_df, pairs.get_level_values(1), cols))
    return pd.Series(rowwise_cosine(lv, rv), index=pairs, name=label)

def make_classifier(cfg, neg, pos):
    m = cfg["matcher"]
    cw = None if cfg.get("class_weight") == "none" else "balanced"
    if m == "LogisticRegression":
        return LogisticRegression(C=float(cfg["C"]), max_iter=1000, class_weight=cw)
    if m == "RandomForest":
        return RandomForestClassifier(n_estimators=int(cfg["n_estimators"]),
                                      max_depth=int(cfg["max_depth"]),
                                      class_weight=cw, random_state=42, n_jobs=-1)
    if m == "GradientBoosting":
        return GradientBoostingClassifier(n_estimators=int(cfg["n_estimators"]),
                                          max_depth=int(cfg["max_depth"]),
                                          learning_rate=float(cfg["learning_rate"]),
                                          random_state=42)
    raise ValueError(f"unknown matcher {m}")

def main():
    cfg = json.loads(sys.argv[1])
    np.random.seed(int(cfg.get("seed", 42)))
    t_start = time.time()

    tmdb = pd.read_csv(TMDB_PATH, delimiter="|")
    tvdb = pd.read_csv(TVDB_PATH, delimiter="|")

    rename_map = {
        "https://www.scads.de/movieBenchmark/ontology/title": "title",
        "https://www.scads.de/movieBenchmark/ontology/name": "name",
        "http://dbpedia.org/ontology/abstract": "abstract",
        "http://dbpedia.org/ontology/episodeNumber": "episodeNumber",
        "http://dbpedia.org/ontology/seasonNumber": "seasonNumber",
        "http://dbpedia.org/ontology/releaseDate": "releaseDate",
    }
    tmdb = tmdb.rename(columns=rename_map)
    tvdb = tvdb.rename(columns=rename_map)

    for df in [tmdb, tvdb]:
        df["title"] = clean(df["title"].fillna(""))
        df["name"] = clean(df["name"].fillna(""))
        df["abstract"] = clean(df["abstract"].fillna(""))
    tmdb["episodeNumber"] = pd.to_numeric(tmdb["episodeNumber"], errors="coerce")
    tmdb["seasonNumber"] = pd.to_numeric(tmdb["seasonNumber"], errors="coerce")
    tvdb["episodeNumber"] = pd.to_numeric(tvdb["episodeNumber"], errors="coerce")
    tvdb["seasonNumber"] = pd.to_numeric(tvdb["seasonNumber"], errors="coerce")
    tmdb["releaseYear"] = pd.to_datetime(tmdb["releaseDate"], errors="coerce").dt.year
    tvdb["releaseYear"] = pd.to_datetime(tvdb["releaseDate"], errors="coerce").dt.year
    tmdb.set_index("id", inplace=True)
    tvdb.set_index("id", inplace=True)

    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)
    y_train = train_df["label"].values
    y_valid = valid_df["label"].values
    y_test = test_df["label"].values
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()

    train_index = pd.MultiIndex.from_arrays([train_df["left_id"], train_df["right_id"]])
    valid_index = pd.MultiIndex.from_arrays([valid_df["left_id"], valid_df["right_id"]])
    test_index = pd.MultiIndex.from_arrays([test_df["left_id"], test_df["right_id"]])

    compare = recordlinkage.Compare()
    compare.string("title", "title", method="cosine", label="title_cosine")
    compare.string("title", "title", method="jarowinkler", label="title_jw")
    compare.string("name", "name", method="cosine", label="name_cosine")
    compare.string("name", "name", method="jarowinkler", label="name_jw")
    compare.string("abstract", "abstract", method="cosine", label="abstract_cosine")
    compare.exact("episodeNumber", "episodeNumber", label="episode_exact")
    compare.exact("seasonNumber", "seasonNumber", label="season_exact")
    compare.exact("releaseYear", "releaseYear", label="year_exact")
    features_train = compare.compute(train_index, tmdb, tvdb)
    features_valid = compare.compute(valid_index, tmdb, tvdb)
    features_test = compare.compute(test_index, tmdb, tvdb)

    train_left_ids = pd.Index(train_df["left_id"].unique())
    train_right_ids = pd.Index(train_df["right_id"].unique())
    word_corpus = np.concatenate([get_texts(tmdb, train_left_ids, ["title", "name", "abstract"]),
                                  get_texts(tvdb, train_right_ids, ["title", "name", "abstract"])])
    char_corpus = np.concatenate([get_texts(tmdb, train_left_ids, ["title", "name"]),
                                  get_texts(tvdb, train_right_ids, ["title", "name"])])
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2).fit(word_corpus)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit(char_corpus)

    for pidx, fdf in [(train_index, features_train), (valid_index, features_valid), (test_index, features_test)]:
        fdf["tfidf_word"] = build_tfidf_sim(pidx, tmdb, tvdb, word_vec, ["title", "name", "abstract"], "tfidf_word")
        fdf["tfidf_char"] = build_tfidf_sim(pidx, tmdb, tvdb, char_vec, ["title", "name"], "tfidf_char")

    features_train = features_train.fillna(0)
    features_valid = features_valid.fillna(0)
    features_test = features_test.fillna(0)

    clf = make_classifier(cfg, neg, pos)
    if cfg["matcher"] == "GradientBoosting":
        sw = np.where(y_train == 1, neg / pos, 1.0)
        clf.fit(features_train, y_train, sample_weight=sw)
    else:
        clf.fit(features_train, y_train)

    probs_valid = clf.predict_proba(features_valid)[:, 1]
    probs_test = clf.predict_proba(features_test)[:, 1]

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
        "test_point": {"t": round(best_t, 3), "precision": round(test_p, 6),
                       "recall": round(test_r, 6), "f1": round(test_f1, 6)},
        "pr_curve": curve,
        "time_sec": round(time.time() - t_start, 2),
        "peak_mem_mb": round(peak_mem_mb(), 1),
    }
    print("RESULT_JSON:" + json.dumps(result))

if __name__ == "__main__":
    main()