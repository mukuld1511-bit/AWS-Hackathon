"""
train_xgb_v2.py
===============
Retrain XGBoost Re-Ranker with Enhanced 10-Feature Set

New Features vs v1 (7 features):
  8. us_state_match    — 1 if both S1 and candidate share same 2-letter US state
  9. us_state_conflict — 1 if both have states and they differ (strong FP signal)
 10. addr_jaccard      — Jaccard overlap of address tokens (normalized 0-100)

Pipeline:
  1. Build inverted indices over train S2+S3 (using tight_blocker logic)
  2. For each S1 in train_ground_truth, pair with all candidates from blocking
  3. Label: 1 if candidate is in ground truth, 0 otherwise (hard negative)
  4. Extract 10-feature vector for each pair
  5. Train XGBoost with scale_pos_weight to handle class imbalance
  6. Save model to xgb_reranker_v2.json
  7. Evaluate on 10% holdout
"""
import os
import re
import csv
import sys
import random
import numpy as np
import xgboost as xgb
from collections import defaultdict
from rapidfuzz import fuzz, distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Add src/ to path for tight_blocker
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from tight_blocker import build_indices, get_candidates, clean_name, extract_geo, extract_numbers

DATA_DIR = "student_resource/dataset/train"
OUTPUT_MODEL = "xgb_reranker_v2.json"
MAX_TRAIN_S1 = 500_000  # Use 500K S1 entities for training (covers full dataset)
MAX_CANDIDATES_PER_ENTITY = 10  # Match inference-time cap
NEGATIVE_SAMPLE_RATIO = 5  # For every true positive, sample 5 negatives

US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID',
    'IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO',
    'MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA',
    'RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'
}
RE_US_STATE = re.compile(r'\b([A-Z]{2})\b')

def extract_us_state(addr: str) -> str:
    if not addr: return ""
    for m in RE_US_STATE.finditer(addr):
        if m.group(1) in US_STATES:
            return m.group(1)
    return ""


def extract_features_v2(s1_name, s1_addr, s2_name, s2_addr):
    """10-feature vector for XGBoost v2."""
    features = []

    # === Original 7 features ===
    features.append(fuzz.ratio(s1_name, s2_name))
    features.append(fuzz.token_set_ratio(s1_name, s2_name))
    features.append(fuzz.token_sort_ratio(s1_name, s2_name))
    features.append(fuzz.partial_ratio(s1_name, s2_name))
    features.append(distance.JaroWinkler.normalized_similarity(s1_name, s2_name) * 100)

    s1_addr_tokens = set(s1_addr.lower().split()) if s1_addr else set()
    s2_addr_tokens = set(s2_addr.lower().split()) if s2_addr else set()
    overlap = len(s1_addr_tokens & s2_addr_tokens)
    features.append(overlap)
    features.append(abs(len(s1_name) - len(s2_name)))

    # === New Phase 1 Features ===
    s1_state = extract_us_state(s1_addr)
    s2_state = extract_us_state(s2_addr)

    # Feature 8: US State match
    state_match = 1 if (s1_state and s2_state and s1_state == s2_state) else 0
    features.append(state_match)

    # Feature 9: US State conflict (strong false positive signal)
    state_conflict = 1 if (s1_state and s2_state and s1_state != s2_state) else 0
    features.append(state_conflict)

    # Feature 10: Address token Jaccard ratio (normalized 0-100)
    union_size = len(s1_addr_tokens | s2_addr_tokens)
    addr_jaccard = (overlap / union_size * 100) if union_size > 0 else 0.0
    features.append(addr_jaccard)

    return features


def main():
    print("=" * 70)
    print("🏋️  XGBoost v2 Retraining with 10-Feature Set")
    print("=" * 70)

    # Step 1: Build blocking indices
    print("\n[1/5] Building inverted indices over train S2+S3...")
    idx, records = build_indices([
        os.path.join(DATA_DIR, "train_source2.tsv"),
        os.path.join(DATA_DIR, "train_source3.tsv"),
    ])
    print(f"[+] Indexed {len(records):,} S2+S3 records.")

    # Step 2: Load ground truth
    print("\n[2/5] Loading ground truth...")
    gt = {}
    with open(os.path.join(DATA_DIR, "train_ground_truth.tsv"), encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) >= 2:
                gt[row[0]] = set(row[1].split(',')) if row[1].strip() else set()
    print(f"[+] Loaded ground truth for {len(gt):,} S1 entities.")

    # Step 3: Generate training pairs
    print(f"\n[3/5] Generating training pairs (max {MAX_TRAIN_S1:,} S1 entities)...")
    X, y = [], []
    s1_processed = 0
    pos_count = 0
    neg_count = 0

    with open(os.path.join(DATA_DIR, "train_source1.tsv"), encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if s1_processed >= MAX_TRAIN_S1:
                break
            if len(row) < 4:
                continue

            s1_id = row[0].strip()
            s1_name = clean_name(row[1].strip())
            s1_addr = row[2].strip()

            gt_matches = gt.get(s1_id, set())
            candidates = get_candidates(row, idx, MAX_CANDIDATES_PER_ENTITY)

            if not candidates:
                s1_processed += 1
                continue

            # Generate positive pairs (true matches that appear in candidates)
            pos_cands = [c for c in candidates if c in gt_matches]
            neg_cands = [c for c in candidates if c not in gt_matches]

            # Also add GT matches NOT in candidates as positives (recall boost for training)
            extra_pos = list(gt_matches - set(candidates))[:3]

            all_pos = pos_cands + extra_pos
            # Sample negatives (up to NEGATIVE_SAMPLE_RATIO * positives)
            n_neg = min(len(neg_cands), max(len(all_pos) * NEGATIVE_SAMPLE_RATIO, 3))
            sampled_neg = random.sample(neg_cands, min(n_neg, len(neg_cands)))

            for cid in all_pos:
                if cid not in records:
                    continue
                t_cn, t_addr, _, _ = records[cid]
                feats = extract_features_v2(s1_name, s1_addr, t_cn, t_addr)
                X.append(feats)
                y.append(1)
                pos_count += 1

            for cid in sampled_neg:
                if cid not in records:
                    continue
                t_cn, t_addr, _, _ = records[cid]
                feats = extract_features_v2(s1_name, s1_addr, t_cn, t_addr)
                X.append(feats)
                y.append(0)
                neg_count += 1

            s1_processed += 1
            if s1_processed % 50000 == 0:
                print(f"   Processed {s1_processed:,} S1 | Pairs: {len(X):,} "
                      f"(+{pos_count:,} / -{neg_count:,})")

    print(f"\n[+] Training pairs generated: {len(X):,}")
    print(f"    Positives: {pos_count:,} | Negatives: {neg_count:,}")
    print(f"    Imbalance ratio: 1:{neg_count//max(pos_count,1):.1f}")

    # Step 4: Train XGBoost
    print("\n[4/5] Training XGBoost v2...")
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    scale_pos_weight = neg_count / max(pos_count, 1)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist',
        'device': 'cuda',
        'seed': 42,
    }

    print(f"   Training on {len(X_train):,} pairs | Val on {len(X_val):,} pairs")
    print(f"   scale_pos_weight = {scale_pos_weight:.2f}")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=50,
    )

    model.save_model(OUTPUT_MODEL)
    print(f"\n[+] Model saved to {OUTPUT_MODEL}")

    # Step 5: Evaluate on validation set
    print("\n[5/5] Evaluating on validation holdout...")
    probs = model.predict(dval)

    for threshold in [0.85, 0.90, 0.95, 0.99]:
        preds = (probs >= threshold).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        print(f"   Threshold {threshold:.2f}: Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")

    print("\n✅ Done! Use xgb_reranker_v2.json in run_pipeline_v2.py")
    print("   Change: xgb_model.load_model('xgb_reranker_v2.json')")


if __name__ == "__main__":
    main()
