import os
import csv
import random
import numpy as np
import xgboost as xgb
from rapidfuzz import fuzz, distance
from tqdm import tqdm

DATA_DIR = "student_resource/dataset/train"

def extract_features(s1_name, s1_addr, s2_name, s2_addr):
    """Extract string similarity and address overlap features for the GBDT."""
    features = []
    
    # Name Features
    features.append(fuzz.ratio(s1_name, s2_name))
    features.append(fuzz.token_set_ratio(s1_name, s2_name))
    features.append(fuzz.token_sort_ratio(s1_name, s2_name))
    features.append(distance.JaroWinkler.normalized_similarity(s1_name, s2_name) * 100)
    
    # Address Features
    s1_addr_tokens = set(s1_addr.lower().split()) if s1_addr else set()
    s2_addr_tokens = set(s2_addr.lower().split()) if s2_addr else set()
    overlap = len(s1_addr_tokens & s2_addr_tokens)
    features.append(overlap)
    
    # Length features
    features.append(abs(len(s1_name) - len(s2_name)))
    
    return features

def main():
    print("="*70)
    print("🚀 TRAINING XGBOOST RE-RANKER (STAGE 2)")
    print("="*70)
    
    # 1. Load Ground Truth to get Positive Pairs
    print("[*] Loading Ground Truth Positive Pairs...")
    gt_path = os.path.join(DATA_DIR, "train_ground_truth.tsv")
    positive_pairs = [] # (s1_id, match_id)
    with open(gt_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for i, row in enumerate(reader):
            if i > 200000: break # Train on 200k subset for speed
            if len(row) < 2: continue
            s1 = row[0].strip()
            matches = [x.strip() for x in row[1].split(',')] if row[1].strip() else []
            for m in matches:
                positive_pairs.append((s1, m))
                
    print(f"[+] Loaded {len(positive_pairs)} positive pairs.")
    
    # 2. Load S1 data for these pairs
    print("[*] Loading S1 Data...")
    s1_data = {}
    with open(os.path.join(DATA_DIR, "train_source1.tsv"), encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            s1_data[row[0].strip()] = (row[1].strip(), row[2].strip())
            
    # 3. Load S2/S3 data for these pairs
    print("[*] Loading S2/S3 Data...")
    target_data = {}
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        with open(os.path.join(DATA_DIR, fname), encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                if len(row) < 3: continue
                target_data[row[0].strip()] = (row[1].strip(), row[2].strip())
                
    # 4. Generate Training Dataset (Positives + Hard Negatives)
    print("[*] Generating Features (Positives)...")
    X = []
    y = []
    
    # Positives
    for s1_id, match_id in tqdm(positive_pairs):
        if s1_id in s1_data and match_id in target_data:
            s1_n, s1_a = s1_data[s1_id]
            t_n, t_a = target_data[match_id]
            X.append(extract_features(s1_n, s1_a, t_n, t_a))
            y.append(1)
            
    # Negatives (Random target samples for simplicity in this baseline)
    print("[*] Generating Features (Negatives)...")
    target_keys = list(target_data.keys())
    for s1_id, _ in tqdm(positive_pairs):
        if s1_id in s1_data:
            s1_n, s1_a = s1_data[s1_id]
            # Pick a random negative
            neg_id = random.choice(target_keys)
            t_n, t_a = target_data[neg_id]
            X.append(extract_features(s1_n, s1_a, t_n, t_a))
            y.append(0)

    # 5. Train XGBoost
    print("\n[*] Training XGBoost Classifier...")
    X = np.array(X)
    y = np.array(y)
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'tree_method': 'hist' # Fast histogram optimized
    }
    
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    model_path = "xgb_reranker.json"
    model.save_model(model_path)
    print(f"[+] Model saved to {model_path}!")

if __name__ == "__main__":
    main()
