"""
generate_splits.py
-------------------
Generates entity-disjoint train/valid/test splits for each
synthetic FEBRL dataset (10K, 50K, 100K, 200K, 300K, 1M, 2M).

Split strategy (same as DER benchmarking):
  - 70% train / 15% valid / 15% test by entities
  - Entity-disjoint: connected components from GT pairs are
    assigned entirely to one split
  - Negatives: hard (same postcode) + random
  - Negative ratio: 3 negatives per positive (scaled down from
    CDDB's 10x because synthetic datasets are much larger)

Output per dataset:
  <splits_dir>/<size>/train_set.csv
  <splits_dir>/<size>/valid_set.csv
  <splits_dir>/<size>/test_set.csv

Each CSV has columns: left_id, right_id, label (1=match, 0=non-match)
"""

import os
import random
from collections import defaultdict

import pandas as pd

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

CONVERTED_DIR = "/home/it2022025/er_scalability/converted"
SPLITS_DIR    = "/home/it2022025/er_scalability/splits"

DATASETS = ["10K", "50K", "100K", "200K", "300K", "1M", "2M"]

SEED   = 42
random.seed(SEED)

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO  = 0.15

# Negatives per positive — use 3x for large datasets to keep files manageable
NEGATIVE_RATIO = 3

# Fraction of negatives sampled as "hard" (same postcode)
HARD_NEGATIVE_FRACTION = 0.30

# -------------------------------------------------------
# UTILITIES (same logic as CDDB split script)
# -------------------------------------------------------

def canon_pair(a, b):
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


def build_connected_components(ids, pairs):
    parent = {x: x for x in ids}
    rank   = {x: 0 for x in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1

    for a, b in pairs:
        if a in parent and b in parent:
            union(a, b)

    comp_map = defaultdict(list)
    for x in ids:
        comp_map[find(x)].append(x)
    return list(comp_map.values())


def split_entities_by_components(components, train_ratio, valid_ratio):
    random.shuffle(components)
    total_entities = sum(len(c) for c in components)
    train_target   = int(total_entities * train_ratio)
    valid_target   = int(total_entities * valid_ratio)

    train_entities = set()
    valid_entities = set()
    test_entities  = set()

    for comp in components:
        comp_set = set(comp)
        if len(train_entities) < train_target:
            train_entities |= comp_set
        elif len(valid_entities) < valid_target:
            valid_entities |= comp_set
        else:
            test_entities |= comp_set

    return train_entities, valid_entities, test_entities


def sample_hard_negatives(entity_ids, id_to_postcode, positive_pairs,
                          target, max_attempts):
    """Sample hard negatives: pairs sharing the same postcode but not in GT."""
    negatives    = set()
    post_groups  = defaultdict(list)
    for eid in entity_ids:
        pc = id_to_postcode.get(eid, "")
        if pc and pc != "nan":
            post_groups[pc].append(eid)

    candidate_posts = [p for p, g in post_groups.items() if len(g) >= 2]
    attempts = 0
    while len(negatives) < target and attempts < max_attempts and candidate_posts:
        attempts += 1
        pc   = random.choice(candidate_posts)
        a, b = random.sample(post_groups[pc], 2)
        pair = canon_pair(a, b)
        if pair not in positive_pairs:
            negatives.add(pair)
    return negatives


def sample_random_negatives(entity_ids, positive_pairs, target,
                            max_attempts, seed_pairs=None):
    """Sample random negatives not in GT."""
    negatives = set(seed_pairs or set())
    ids       = list(entity_ids)
    attempts  = 0
    while len(negatives) < target and attempts < max_attempts:
        attempts += 1
        a, b = random.sample(ids, 2)
        pair = canon_pair(a, b)
        if pair not in positive_pairs:
            negatives.add(pair)
    return negatives


def build_split_df(split_name, split_entities, split_positive_pairs,
                   id_to_postcode):
    requested_neg    = len(split_positive_pairs) * NEGATIVE_RATIO
    n                = len(split_entities)
    max_possible_neg = max(0, (n * (n - 1)) // 2 - len(split_positive_pairs))
    target_neg       = min(requested_neg, max_possible_neg)
    target_hard      = int(target_neg * HARD_NEGATIVE_FRACTION)
    max_attempts     = max(10000, target_neg * 20)

    if target_neg < requested_neg:
        print(f"  [INFO] {split_name}: requested {requested_neg} negatives, "
              f"capped to {target_neg}")

    hard_neg  = sample_hard_negatives(
        entity_ids=split_entities,
        id_to_postcode=id_to_postcode,
        positive_pairs=split_positive_pairs,
        target=target_hard,
        max_attempts=max_attempts,
    )
    negatives = sample_random_negatives(
        entity_ids=split_entities,
        positive_pairs=split_positive_pairs,
        target=target_neg,
        max_attempts=max_attempts,
        seed_pairs=hard_neg,
    )

    if len(negatives) < target_neg:
        print(f"  [WARN] {split_name}: requested {target_neg} negatives, "
              f"generated {len(negatives)}")

    rows = []
    for a, b in split_positive_pairs:
        rows.append({"left_id": a, "right_id": b, "label": 1})
    for a, b in negatives:
        rows.append({"left_id": a, "right_id": b, "label": 0})

    random.shuffle(rows)
    return pd.DataFrame(rows)


def pair_in_split(pair, split_entities):
    return pair[0] in split_entities and pair[1] in split_entities


# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------

print("\n" + "=" * 60)
print("  SPLIT GENERATION FOR SYNTHETIC DATASETS")
print("=" * 60)

os.makedirs(SPLITS_DIR, exist_ok=True)

summary = []

for ds in DATASETS:
    profiles_path = os.path.join(CONVERTED_DIR, ds, "profiles.csv")
    gt_path       = os.path.join(CONVERTED_DIR, ds, "ground_truth.csv")
    out_dir       = os.path.join(SPLITS_DIR, ds)

    print(f"\n{'='*60}")
    print(f"  Dataset: {ds}")
    print(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)

    # -- Load --
    df    = pd.read_csv(profiles_path, engine="python", na_filter=False)
    gt_df = pd.read_csv(gt_path, engine="python")

    print(f"  Records  : {len(df):,}")
    print(f"  GT pairs : {len(gt_df):,}")

    # -- Build metadata --
    entity_ids     = set(df["id"].astype(int))
    id_to_postcode = {
        int(r["id"]): str(r.get("postcode", "")).strip()
        for _, r in df.iterrows()
    }

    # -- Build positive pairs --
    positive_pairs = set()
    for _, row in gt_df.iterrows():
        pair = canon_pair(row["id1"], row["id2"])
        if pair[0] in entity_ids and pair[1] in entity_ids:
            positive_pairs.add(pair)

    print(f"  Valid GT pairs: {len(positive_pairs):,}")

    # -- Entity-disjoint split --
    components = build_connected_components(entity_ids, positive_pairs)
    train_ent, valid_ent, test_ent = split_entities_by_components(
        components, TRAIN_RATIO, VALID_RATIO
    )

    print(f"  Entities — train: {len(train_ent):,}  "
          f"valid: {len(valid_ent):,}  test: {len(test_ent):,}")

    # Verify disjoint
    assert train_ent.isdisjoint(valid_ent)
    assert train_ent.isdisjoint(test_ent)
    assert valid_ent.isdisjoint(test_ent)

    # -- Split positive pairs --
    train_pos = {p for p in positive_pairs if pair_in_split(p, train_ent)}
    valid_pos = {p for p in positive_pairs if pair_in_split(p, valid_ent)}
    test_pos  = {p for p in positive_pairs if pair_in_split(p, test_ent)}

    print(f"  Pos pairs — train: {len(train_pos):,}  "
          f"valid: {len(valid_pos):,}  test: {len(test_pos):,}")

    # -- Build split DataFrames --
    train_df = build_split_df("train", train_ent, train_pos, id_to_postcode)
    valid_df = build_split_df("valid", valid_ent, valid_pos, id_to_postcode)
    test_df  = build_split_df("test",  test_ent,  test_pos,  id_to_postcode)

    # -- Save --
    train_df.to_csv(os.path.join(out_dir, "train_set.csv"), index=False)
    valid_df.to_csv(os.path.join(out_dir, "valid_set.csv"), index=False)
    test_df.to_csv( os.path.join(out_dir, "test_set.csv"),  index=False)

    # -- Print summary --
    for name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        pos = int(split_df["label"].sum())
        neg = len(split_df) - pos
        print(f"  {name:<6}: {len(split_df):>8,} pairs  "
              f"(pos={pos:,}, neg={neg:,}, ratio={neg/pos:.1f}x)")

    print(f"  Saved to: {out_dir}")

    summary.append({
        "dataset"    : ds,
        "n_records"  : len(df),
        "n_gt_pairs" : len(positive_pairs),
        "train_pairs": len(train_df),
        "valid_pairs": len(valid_df),
        "test_pairs" : len(test_df),
        "train_pos"  : int(train_df["label"].sum()),
        "valid_pos"  : int(valid_df["label"].sum()),
        "test_pos"   : int(test_df["label"].sum()),
    })

# -------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
df_sum = pd.DataFrame(summary)
print(df_sum[["dataset", "n_records", "n_gt_pairs",
              "train_pairs", "valid_pairs", "test_pairs"]].to_string(index=False))
print(f"\nSplits saved to: {SPLITS_DIR}")