"""
train_heavy_ensemble.py
=======================
Trains the Heavy Tri-Model GBDT Ensemble:
  1. XGBoost GPU (Depth-wise)
  2. LightGBM (Leaf-wise)
  3. CatBoost GPU (Symmetric oblivious)

Uses 18-Dimensional Feature Matrix from src/features_18d.py.
Evaluates on validation holdout to calibrate the optimal Macro F0.5 decision threshold.
Amazon ML Challenge 2026
"""
import os
import csv
import time
import random
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from src.features_18d import (
    super_clean_name,
    extract_features_18d,
    US_STATES,
    RE_US_STATE
)
from src.tight_blocker_v2 import extract_geo, extract_numbers

random.seed(42)
np.random.seed(42)

def extract_us_state(addr: str) -> str:
    if not addr: return ""
    for m in RE_US_STATE.finditer(addr):
        if m.group(1) in US_STATES:
            return m.group(1)
    return ""


def main():
    print("=" * 70)
    print("🚀 TRAINING HEAVY TRI-MODEL GBDT ENSEMBLE (XGB + LGB + CAT)")
    print("=" * 70)
    start_time = time.time()

    data_dir = "student_resource/dataset/train"
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")

    # 1. Load Ground Truth for 70,000 entities
    print("[1/5] Loading Ground Truth...")
    gt = {}
    needed_targets = set()
    with open(gt_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            s1 = row[0].strip()
            targets = [t.strip() for t in row[1].split(',') if t.strip()]
            if targets:
                gt[s1] = targets
                needed_targets.update(targets)
            if len(gt) >= 70000:
                break

    print(f"[+] Loaded GT for {len(gt):,} S1 entities ({len(needed_targets):,} unique target EIDs).")

    # 2. Load S1 records
    print("[2/5] Loading S1 records...")
    s1_dict = {}
    with open(s1_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            eid = row[0].strip()
            if eid in gt:
                name, addr, country = row[1].strip(), row[2].strip(), row[3].strip().lower()
                cn = super_clean_name(name)
                geo = extract_geo(addr, country)
                state = extract_us_state(addr) if country == 'us' else ""
                s1_dict[eid] = (cn, addr, geo, state)

    # 3. Load Target records (S2 + S3)
    print("[3/5] Loading Target records (S2 + S3)...")
    target_dict = {}
    distractor_pool = []
    for sf in [s2_path, s3_path]:
        with open(sf, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for i, row in enumerate(reader):
                eid = row[0].strip()
                is_target = eid in needed_targets
                is_distractor = (len(distractor_pool) < 150000 and i % 30 == 0)

                if is_target or is_distractor:
                    name, addr, country = row[1].strip(), row[2].strip(), row[3].strip().lower()
                    cn = super_clean_name(name)
                    geo = extract_geo(addr, country)
                    state = extract_us_state(addr) if country == 'us' else ""
                    info = (cn, addr, geo, state)

                    if is_target:
                        target_dict[eid] = info
                    elif is_distractor:
                        distractor_pool.append(info)

    print(f"[+] Loaded {len(target_dict):,} true target records, {len(distractor_pool):,} distractor records.")

    # 4. Build Dataset (Positives + Hard Negatives + Random Distractors)
    print("[4/5] Extracting 18-D Features...")
    X = []
    y = []

    # Holdout split
    s1_keys = list(gt.keys())
    random.shuffle(s1_keys)
    split_idx = int(len(s1_keys) * 0.8)
    train_s1 = s1_keys[:split_idx]
    val_s1 = s1_keys[split_idx:]

    def build_subset(s1_list):
        sub_X = []
        sub_y = []
        for s1_id in s1_list:
            if s1_id not in s1_dict:
                continue
            s1_info = s1_dict[s1_id]
            pos_targets = gt[s1_id]

            # True Positives
            for tid in pos_targets:
                if tid in target_dict:
                    t_info = target_dict[tid]
                    feats = extract_features_18d(
                        s1_info[0], s1_info[1],
                        t_info[0], t_info[1],
                        s1_info[2], t_info[2],
                        s1_info[3], t_info[3]
                    )
                    sub_X.append(feats)
                    sub_y.append(1)

            # Negatives: 1 hard negative from distractor pool per positive
            for _ in range(len(pos_targets)):
                neg_info = random.choice(distractor_pool)
                feats = extract_features_18d(
                    s1_info[0], s1_info[1],
                    neg_info[0], neg_info[1],
                    s1_info[2], neg_info[2],
                    s1_info[3], neg_info[3]
                )
                sub_X.append(feats)
                sub_y.append(0)

        return np.array(sub_X, dtype=np.float32), np.array(sub_y, dtype=np.int32)

    X_train, y_train = build_subset(train_s1)
    X_val, y_val = build_subset(val_s1)

    print(f"[+] Train dataset: {X_train.shape[0]:,} samples (Pos: {np.sum(y_train):,}, Neg: {len(y_train)-np.sum(y_train):,})")
    print(f"[+] Val dataset:   {X_val.shape[0]:,} samples (Pos: {np.sum(y_val):,}, Neg: {len(y_val)-np.sum(y_val):,})")

    # 5. Train Models
    # Model 1: XGBoost GPU
    print("\n--- Training Model 1: XGBoost GPU (max_depth=9, n_estimators=1000) ---")
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=9,
        learning_rate=0.04,
        tree_method='hist',
        device='cuda',
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    xgb_model.save_model("xgb_heavy.json")
    print("[+] Saved xgb_heavy.json")

    # Model 2: LightGBM
    print("\n--- Training Model 2: LightGBM (num_leaves=127, n_estimators=1000) ---")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    params_lgb = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.04,
        'num_leaves': 127,
        'max_depth': 12,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    lgb_model = lgb.train(
        params_lgb,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    lgb_model.save_model("lgb_heavy.txt")
    print("[+] Saved lgb_heavy.txt")

    # Model 3: CatBoost GPU
    print("\n--- Training Model 3: CatBoost GPU (depth=8, iterations=1000) ---")
    cb_model = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.04,
        loss_function='Logloss',
        task_type='GPU',
        verbose=100,
        random_seed=42
    )
    cb_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=100
    )
    cb_model.save_model("cat_heavy.cbm")
    print("[+] Saved cat_heavy.cbm")

    # 6. Ensemble Validation & Optimal F0.5 Calibration
    print("\n" + "=" * 70)
    print("🎯 EVALUATING TRI-MODEL ENSEMBLE ON HOLDOUT VALIDATION")
    print("=" * 70)

    pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
    pred_lgb = lgb_model.predict(X_val)
    pred_cat = cb_model.predict_proba(X_val)[:, 1]

    # Weighted Ensemble
    pred_ens = 0.40 * pred_xgb + 0.35 * pred_lgb + 0.25 * pred_cat

    best_thresh = 0.65
    best_f05 = 0.0

    print("Threshold Sweep for Ensemble Macro F0.5:")
    for thresh in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        bin_pred = (pred_ens >= thresh).astype(int)
        tp = np.sum((bin_pred == 1) & (y_val == 1))
        fp = np.sum((bin_pred == 1) & (y_val == 0))
        fn = np.sum((bin_pred == 0) & (y_val == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # F0.5 weights precision twice as much as recall: beta = 0.5
        beta = 0.5
        beta_sq = beta ** 2
        f05 = ((1 + beta_sq) * prec * rec) / (beta_sq * prec + rec) if (prec + rec) > 0 else 0.0

        print(f"  Thresh={thresh:.2f} | Prec={prec:.4f} | Rec={rec:.4f} | F0.5={f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh

    print(f"\n🏆 BEST ENSEMBLE THRESHOLD: {best_thresh:.2f} (Holdout F0.5 = {best_f05:.4f})")
    print(f"Total training pipeline completed in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
