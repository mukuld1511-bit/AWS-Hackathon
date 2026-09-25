#!/usr/bin/env python3
"""
src/evaluate_full_train_ground_truth.py

Rigorous End-to-End Evaluation on Train Ground Truth for Amazon ML Challenge 2026.
Uses Dataset Sources 1, 2, 3 and Ground Truth from student_resource/dataset/train/.

Evaluates:
- Tier 1 (High-Confidence Transliteration: Cosine >= 0.95)
- Tier 2 (Semantic Address Bridge: Overlap >= 7 or Overlap >= 5 + Num, Cosine >= 0.45)
- OVERALL COMBINED PIPELINE (Tier 1 + Tier 2)

Computes exact:
- Precision
- Recall (Indic ground truth matches)
- F1 Score
- Micro F_0.5
- Macro F_0.5 (Official Metric)
- Singleton Accuracy
"""

import os
import re
import csv
import sys
import time
import random
import argparse
from collections import defaultdict, Counter

if "HF_HOME" not in os.environ:
    local_cache = os.path.abspath(".venv/hf_cache")
    if os.path.exists(".venv"):
        os.environ["HF_HOME"] = local_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Paths
TRAIN_S1 = "student_resource/dataset/train/train_source1.tsv"
TRAIN_S2 = "student_resource/dataset/train/train_source2.tsv"
TRAIN_S3 = "student_resource/dataset/train/train_source3.tsv"
TRAIN_GT = "student_resource/dataset/train/train_ground_truth.tsv"
OUTPUT_REPORT = "output/train_ground_truth_evaluation_report.txt"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HIGH_CONF_SIM_THRESHOLD = 0.95
BRIDGE_SIM_THRESHOLD = 0.45
MIN_ADDR_TOKEN_OVERLAP = 7
RELAXED_ADDR_TOKEN_OVERLAP = 5
MAX_MATCHES_PER_ENTITY = 15

RE_INDIC = re.compile(r'[\u0900-\u0D7F]')
RE_WORD = re.compile(r'\w+')
RE_NUM = re.compile(r'\b\d+\b')
STOPS = {'india', 'near', 'opp', 'opposite', 'road', 'rd', 'street', 'st', 'floor', 'flr', 'no', 'plot'}


def is_india(country_str):
    if not country_str:
        return False
    return country_str.strip().lower() == "india"


def extract_address_tokens(addr):
    if not addr:
        return set()
    return {w for w in RE_WORD.findall(addr.lower()) if len(w) > 1 and w not in STOPS}


def extract_numbers(addr):
    if not addr:
        return set()
    return set(RE_NUM.findall(addr.lower()))


def compute_f_beta(precision, recall, beta=0.5):
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return ((1 + beta_sq) * precision * recall) / denom


def evaluate(num_eval_anchors=20000, batch_size=512):
    total_start = time.time()
    os.makedirs("output", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    report_lines = []
    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=" * 85)
    log("RIGOROUS END-TO-END EVALUATION ON TRAIN DATASET & GROUND TRUTH")
    log(f"Device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    log(f"Evaluation Target: {num_eval_anchors:,} India S1 anchors against Train Ground Truth")
    log("=" * 85)

    # -----------------------------------------------------------------------
    # Step 1: Scan Train S2 & S3 for Indic Candidates
    # -----------------------------------------------------------------------
    log("\n[*] Step 1: Scanning Train S2 & S3 for India records with Indic business names...")
    t0 = time.time()
    cand_ids = []
    cand_names = []
    cand_tokens = []
    cand_numbers = []
    cand_id_to_idx = {}

    def load_train_cands(filepath, src_label):
        loaded = 0
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                if is_india(row[3]) and RE_INDIC.search(row[1]):
                    cid_str = row[0].strip()
                    idx = len(cand_ids)
                    cand_ids.append(cid_str)
                    cand_names.append(row[1].strip())
                    cand_tokens.append(extract_address_tokens(row[2]))
                    cand_numbers.append(extract_numbers(row[2]))
                    cand_id_to_idx[cid_str] = idx
                    loaded += 1
        return loaded

    s2_count = load_train_cands(TRAIN_S2, "S2")
    s3_count = load_train_cands(TRAIN_S3, "S3")
    total_cands = len(cand_ids)
    log(f"[+] Loaded {total_cands:,} Indic candidates from Train (S2: {s2_count:,}, S3: {s3_count:,}) in {time.time()-t0:.2f}s")

    # Inverted Index with token pruning
    log("[*] Building address inverted index (pruning non-discriminative tokens > 5000)...")
    token_counts = Counter()
    for toks in cand_tokens:
        token_counts.update(toks)

    inv_index = defaultdict(list)
    for cid, toks in enumerate(cand_tokens):
        for t in toks:
            if token_counts[t] <= 5000:
                inv_index[t].append(cid)
    log(f"[+] Inverted index built with {len(inv_index):,} unique tokens.")

    # -----------------------------------------------------------------------
    # Step 2: Stream Ground Truth
    # -----------------------------------------------------------------------
    log("\n[*] Step 2: Streaming Ground Truth to identify true Indic match pairs...")
    t0 = time.time()
    gt_indic_matches = defaultdict(set) # s1_id -> set of true Indic cand_ids
    all_s1_with_indic_gt = set()

    with open(TRAIN_GT, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            s1_id = row[0].strip()
            raw_matches = row[1].strip()
            if not raw_matches:
                continue
            matched_ids = [m.strip() for m in raw_matches.split(",") if m.strip()]
            for mid in matched_ids:
                if mid in cand_id_to_idx:
                    gt_indic_matches[s1_id].add(mid)
                    all_s1_with_indic_gt.add(s1_id)

    log(f"[+] Found {len(all_s1_with_indic_gt):,} S1 entities in Train with true Indic-script matches in S2/S3 (in {time.time()-t0:.2f}s)")

    # -----------------------------------------------------------------------
    # Step 3: Select Evaluation Anchors (Balanced: Positives + Negatives/Singletons)
    # -----------------------------------------------------------------------
    log(f"\n[*] Step 3: Selecting balanced slice of {num_eval_anchors:,} India S1 anchors from Train...")
    t0 = time.time()
    
    # We select 50% entities with true Indic matches and 50% without (negatives/singletons)
    num_pos = num_eval_anchors // 2
    num_neg = num_eval_anchors - num_pos

    eval_s1_records = [] # (s1_id, name, addr)
    pos_collected = 0
    neg_collected = 0

    with open(TRAIN_S1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            if is_india(row[3]):
                s1_id = row[0].strip()
                has_gt = s1_id in gt_indic_matches
                if has_gt and pos_collected < num_pos:
                    eval_s1_records.append((s1_id, row[1].strip(), row[2].strip() if row[2] else ""))
                    pos_collected += 1
                elif (not has_gt) and neg_collected < num_neg:
                    eval_s1_records.append((s1_id, row[1].strip(), row[2].strip() if row[2] else ""))
                    neg_collected += 1

                if pos_collected >= num_pos and neg_collected >= num_neg:
                    break

    log(f"[+] Selected {len(eval_s1_records):,} evaluation S1 anchors: {pos_collected:,} Positives + {neg_collected:,} Negatives/Singletons")

    # -----------------------------------------------------------------------
    # Step 4: Encode Candidate Embeddings & S1 Embeddings on GPU
    # -----------------------------------------------------------------------
    log(f"\n[*] Step 4: Loading SentenceTransformer ({MODEL_NAME}) on {device}...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=device)
    log(f"[+] Model loaded in {time.time()-t0:.2f}s")

    log(f"[*] Encoding {total_cands:,} Candidate names on GPU...")
    t0 = time.time()
    cand_embeddings = model.encode(
        cand_names,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=True,
        device=device,
    ).to(dtype=torch.float16)
    log(f"[+] Candidate embeddings encoded in {time.time()-t0:.2f}s (Shape: {cand_embeddings.shape})")

    # Free memory
    del cand_names
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Step 5: Run Two-Tier Matching Pipeline & Collect Predictions
    # -----------------------------------------------------------------------
    log(f"\n[*] Step 5: Running Two-Tier Matching Pipeline on {len(eval_s1_records):,} S1 anchors...")
    t0 = time.time()

    tier1_preds = defaultdict(set)
    tier2_preds = defaultdict(set)
    combined_preds = defaultdict(set)

    eval_len = len(eval_s1_records)
    for b_start in range(0, eval_len, batch_size):
        b_end = min(b_start + batch_size, eval_len)
        batch = eval_s1_records[b_start:b_end]
        b_ids = [r[0] for r in batch]
        b_names = [r[1] for r in batch]
        b_addrs = [r[2] for r in batch]

        b_embs = model.encode(
            b_names,
            batch_size=len(b_names),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=device,
        ).to(dtype=torch.float16)

        sims = torch.matmul(b_embs, cand_embeddings.T)

        # Tier 1: Cosine >= 0.95
        high_mask = sims >= HIGH_CONF_SIM_THRESHOLD
        h_rows, h_cols = torch.nonzero(high_mask, as_tuple=True)
        if len(h_rows) > 0:
            h_rows_np = h_rows.cpu().numpy()
            h_cols_np = h_cols.cpu().numpy()
            for r_idx, c_idx in zip(h_rows_np, h_cols_np):
                s1_id = b_ids[r_idx]
                cand_id = cand_ids[c_idx]
                tier1_preds[s1_id].add(cand_id)
                combined_preds[s1_id].add(cand_id)

        # Tier 2: Address Bridge
        candidate_pairs = []
        for local_idx, addr_str in enumerate(b_addrs):
            s_toks = extract_address_tokens(addr_str)
            if not s_toks:
                continue
            s_nums = extract_numbers(addr_str)

            cand_hits = defaultdict(int)
            for t in s_toks:
                if t in inv_index:
                    for cid in inv_index[t]:
                        cand_hits[cid] += 1

            for cid, hits in cand_hits.items():
                has_shared_num = bool(s_nums & cand_numbers[cid]) if (s_nums and cand_numbers[cid]) else False
                if hits >= MIN_ADDR_TOKEN_OVERLAP or (hits >= RELAXED_ADDR_TOKEN_OVERLAP and has_shared_num):
                    candidate_pairs.append((local_idx, cid))

        if candidate_pairs:
            p_rows = torch.tensor([p[0] for p in candidate_pairs], device=device, dtype=torch.long)
            p_cols = torch.tensor([p[1] for p in candidate_pairs], device=device, dtype=torch.long)
            pair_sims = sims[p_rows, p_cols]
            valid_mask = pair_sims >= BRIDGE_SIM_THRESHOLD
            valid_indices = torch.nonzero(valid_mask).squeeze(1).cpu().numpy()
            for v_idx in valid_indices:
                loc_r, c_idx = candidate_pairs[v_idx]
                s1_id = b_ids[loc_r]
                cand_id = cand_ids[c_idx]
                tier2_preds[s1_id].add(cand_id)
                combined_preds[s1_id].add(cand_id)

        del b_embs
        del sims

        if (b_end % 5000 == 0) or b_end == eval_len:
            log(f"    Processed {b_end:>6,} / {eval_len:,} S1 entities... ({(b_end/(time.time()-t0)):.0f} rows/s)")

    # -----------------------------------------------------------------------
    # Step 6: Compute Exact Metrics
    # -----------------------------------------------------------------------
    log("\n" + "=" * 85)
    log("COMPUTING RIGOROUS METRICS AGAINST GROUND TRUTH")
    log("=" * 85)

    def evaluate_predictions(preds_dict, label):
        tp_total = 0
        fp_total = 0
        fn_total = 0
        f05_macro_list = []
        singleton_correct = 0
        singleton_total = 0

        for s1_id, _, _ in eval_s1_records:
            gt_set = gt_indic_matches.get(s1_id, set())
            pred_set = preds_dict.get(s1_id, set())

            # Singleton evaluation
            if len(gt_set) == 0:
                singleton_total += 1
                if len(pred_set) == 0:
                    singleton_correct += 1

            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)

            tp_total += tp
            fp_total += fp
            fn_total += fn

            # Entity-level Macro F0.5
            if len(gt_set) == 0 and len(pred_set) == 0:
                f05_macro_list.append(1.0)
            elif len(gt_set) == 0 and len(pred_set) > 0:
                f05_macro_list.append(0.0)
            elif len(gt_set) > 0 and len(pred_set) == 0:
                f05_macro_list.append(0.0)
            else:
                p_i = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f05_macro_list.append(compute_f_beta(p_i, r_i, beta=0.5))

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        micro_f05 = compute_f_beta(precision, recall, beta=0.5)
        macro_f05 = float(np.mean(f05_macro_list))
        singleton_acc = (singleton_correct / singleton_total * 100) if singleton_total > 0 else 0.0

        return {
            "label": label,
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "micro_f05": micro_f05,
            "macro_f05": macro_f05,
            "singleton_acc": singleton_acc,
            "total_preds": tp_total + fp_total
        }

    m_tier1 = evaluate_predictions(tier1_preds, "Tier 1 Only (Cosine >= 0.95)")
    m_tier2 = evaluate_predictions(tier2_preds, "Tier 2 Only (Address Bridge)")
    m_comb  = evaluate_predictions(combined_preds, "OVERALL COMBINED (Tier 1 + Tier 2)")

    log(f"{'Metric':<32} | {'Tier 1 (Name)':<15} | {'Tier 2 (Address)':<17} | {'OVERALL COMBINED':<17}")
    log("-" * 88)
    log(f"{'Precision':<32} | {m_tier1['precision']*100:>14.2f}% | {m_tier2['precision']*100:>16.2f}% | {m_comb['precision']*100:>16.2f}%")
    log(f"{'Recall (Indic Ground Truth)':<32} | {m_tier1['recall']*100:>14.2f}% | {m_tier2['recall']*100:>16.2f}% | {m_comb['recall']*100:>16.2f}%")
    log(f"{'F1 Score (Harmonic Mean)':<32} | {m_tier1['f1']*100:>14.2f}% | {m_tier2['f1']*100:>16.2f}% | {m_comb['f1']*100:>16.2f}%")
    log(f"{'Micro F_0.5 Score':<32} | {m_tier1['micro_f05']*100:>14.2f}% | {m_tier2['micro_f05']*100:>16.2f}% | {m_comb['micro_f05']*100:>16.2f}%")
    log(f"{'Macro F_0.5 (Official Metric)':<32} | {m_tier1['macro_f05']*100:>14.2f}% | {m_tier2['macro_f05']*100:>16.2f}% | {m_comb['macro_f05']*100:>16.2f}%")
    log(f"{'Singleton Accuracy':<32} | {m_tier1['singleton_acc']:>14.2f}% | {m_tier2['singleton_acc']:>16.2f}% | {m_comb['singleton_acc']:>16.2f}%")
    log(f"{'True Positives (TP)':<32} | {m_tier1['tp']:>14,} | {m_tier2['tp']:>16,} | {m_comb['tp']:>16,}")
    log(f"{'False Positives (FP)':<32} | {m_tier1['fp']:>14,} | {m_tier2['fp']:>16,} | {m_comb['fp']:>16,}")
    log(f"{'False Negatives (FN)':<32} | {m_tier1['fn']:>14,} | {m_tier2['fn']:>16,} | {m_comb['fn']:>16,}")
    log("-" * 88)

    # Save report
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    log(f"\n[+] Full report saved to: {OUTPUT_REPORT}")
    log(f"[+] Total execution time: {time.time()-total_start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-anchors", type=int, default=10000, help="Number of S1 anchors for evaluation")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    args = parser.parse_args()

    evaluate(num_eval_anchors=args.num_anchors, batch_size=args.batch_size)
