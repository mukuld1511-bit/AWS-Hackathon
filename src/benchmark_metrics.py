#!/usr/bin/env python3
"""
src/benchmark_metrics.py
Evaluates accuracy, precision, recall, and Macro F0.5 on a large train sample.
"""
import os
import csv
import sys
import time
import torch
from sentence_transformers import SentenceTransformer

if "HF_HOME" not in os.environ:
    local_cache = os.path.abspath(".venv/hf_cache")
    if os.path.exists(".venv"):
        os.environ["HF_HOME"] = local_cache

TRAIN_S1 = "student_resource/dataset/train/train_source1.tsv"
TRAIN_S2 = "student_resource/dataset/train/train_source2.tsv"
TRAIN_S3 = "student_resource/dataset/train/train_source3.tsv"
TRAIN_GT = "student_resource/dataset/train/train_ground_truth.tsv"
MODEL_PATH = "output/fine_tuned_multilingual_entity_model"

def run_benchmark(sample_size=10_000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading model from {MODEL_PATH} on {device}...")
    model = SentenceTransformer(MODEL_PATH, device=device)

    # 1. Load Ground Truth sample
    gt = {}
    with open(TRAIN_GT, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for i, row in enumerate(r):
            if i >= sample_size: break
            s1_id = row[0].strip()
            matches = set(m.strip() for m in row[1].split(",") if m.strip()) if len(row) > 1 else set()
            gt[s1_id] = matches

    print(f"[+] Loaded {len(gt):,} Ground Truth rows.")

    # 2. Load candidate pool for this sample
    all_matched_eids = set()
    for matches in gt.values():
        all_matched_eids.update(matches)

    print(f"[+] Total target entities in ground truth pool: {len(all_matched_eids):,}")

    target_records = {}
    for p in [TRAIN_S2, TRAIN_S3]:
        with open(p, encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r)
            for row in r:
                if len(row) >= 4 and row[0].strip() in all_matched_eids:
                    target_records[row[0].strip()] = f"{row[1].strip()} | {row[2].strip()}" if row[2].strip() else row[1].strip()

    # 3. Load S1 records
    s1_records = {}
    with open(TRAIN_S1, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row[0].strip() in gt:
                s1_records[row[0].strip()] = f"{row[1].strip()} | {row[2].strip()}" if len(row) > 2 and row[2].strip() else row[1].strip()

    # 4. Score queries against target pool
    target_ids = list(target_records.keys())
    target_texts = [target_records[tid] for tid in target_ids]
    
    print(f"[*] Encoding {len(target_texts):,} candidate targets...")
    t_embs = model.encode(target_texts, batch_size=512, convert_to_tensor=True, device=device, normalize_embeddings=True)

    print(f"[*] Evaluating {len(s1_records):,} Source 1 queries...")
    s1_ids = list(s1_records.keys())
    s1_texts = [s1_records[sid] for sid in s1_ids]
    s1_embs = model.encode(s1_texts, batch_size=512, convert_to_tensor=True, device=device, normalize_embeddings=True)

    # Cosine similarity matrix
    sim_matrix = torch.matmul(s1_embs, t_embs.T).cpu().numpy()

    # Evaluate at threshold 0.65
    tp, fp, fn = 0, 0, 0
    exact_matches = 0
    singletons_correct = 0
    singletons_total = 0

    for i, s1_id in enumerate(s1_ids):
        gt_matches = gt.get(s1_id, set())
        row_sims = sim_matrix[i]
        
        # Select best S2 and best S3 above threshold 0.65
        best_s2, best_s3 = None, None
        for j, sim in enumerate(row_sims):
            if sim >= 0.65:
                tid = target_ids[j]
                if tid.startswith(("S2-", "S2_")) and (best_s2 is None or sim > best_s2[1]):
                    best_s2 = (tid, sim)
                elif tid.startswith(("S3-", "S3_")) and (best_s3 is None or sim > best_s3[1]):
                    best_s3 = (tid, sim)

        pred_matches = set()
        if best_s2: pred_matches.add(best_s2[0])
        if best_s3: pred_matches.add(best_s3[0])

        if not gt_matches:
            singletons_total += 1
            if not pred_matches: singletons_correct += 1
            else: fp += len(pred_matches)
        else:
            cur_tp = len(pred_matches.intersection(gt_matches))
            cur_fp = len(pred_matches - gt_matches)
            cur_fn = len(gt_matches - pred_matches)

            tp += cur_tp
            fp += cur_fp
            fn += cur_fn

            if pred_matches == gt_matches:
                exact_matches += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta = 0.5
    f05 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if ((beta**2 * precision) + recall) > 0 else 0.0
    singleton_acc = singletons_correct / singletons_total if singletons_total > 0 else 1.0
    exact_acc = exact_matches / len(s1_ids)

    print("\n" + "=" * 70)
    print("🏆 MULTILINGUAL EMBEDDING BENCHMARK REPORT (THRESHOLD = 0.65)")
    print("=" * 70)
    print(f"[*] Evaluated Entities:       {len(s1_ids):,}")
    print(f"[*] True Positives (TP):      {tp:,}")
    print(f"[*] False Positives (FP):     {fp:,}")
    print(f"[*] False Negatives (FN):     {fn:,}")
    print("-" * 70)
    print(f"🎯 Precision Score:           {precision * 100:.2f}%")
    print(f"🎯 Recall Score:              {recall * 100:.2f}%")
    print(f"🏆 Macro F0.5 Score:          {f05 * 100:.2f}%")
    print(f"🎯 Singleton Accuracy:        {singleton_acc * 100:.2f}%")
    print(f"🎯 Exact Entity Accuracy:     {exact_acc * 100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark(sample_size=10_000)
