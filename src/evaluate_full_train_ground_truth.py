#!/usr/bin/env python3
"""
src/evaluate_full_train_ground_truth.py

Rigorous End-to-End Evaluation on Train Ground Truth with State Veto & Multi-Signal Anchoring.
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

# Indian States & Metro Cities mapping to State Codes
STATES = {
    'maharashtra': 'MH', 'mh': 'MH', 'mumbai': 'MH', 'pune': 'MH', 'nagpur': 'MH', 'nashik': 'MH', 'thane': 'MH',
    'tamil nadu': 'TN', 'tamilnadu': 'TN', 'tn': 'TN', 'chennai': 'TN', 'coimbatore': 'TN', 'madurai': 'TN',
    'karnataka': 'KA', 'ka': 'KA', 'bangalore': 'KA', 'bengaluru': 'KA', 'mysore': 'KA', 'hubli': 'KA',
    'delhi': 'DL', 'new delhi': 'DL', 'dl': 'DL',
    'telangana': 'TS', 'ts': 'TS', 'tg': 'TS', 'hyderabad': 'TS', 'secunderabad': 'TS',
    'andhra pradesh': 'AP', 'andhra': 'AP', 'ap': 'AP', 'visakhapatnam': 'AP', 'vijayawada': 'AP',
    'gujarat': 'GJ', 'gj': 'GJ', 'ahmedabad': 'GJ', 'surat': 'GJ', 'vadodara': 'GJ', 'rajkot': 'GJ',
    'west bengal': 'WB', 'bengal': 'WB', 'wb': 'WB', 'kolkata': 'WB', 'howrah': 'WB',
    'uttar pradesh': 'UP', 'up': 'UP', 'lucknow': 'UP', 'kanpur': 'UP', 'noida': 'UP', 'varanasi': 'UP', 'agra': 'UP',
    'haryana': 'HR', 'hr': 'HR', 'gurgaon': 'HR', 'gurugram': 'HR', 'faridabad': 'HR',
    'rajasthan': 'RJ', 'rj': 'RJ', 'jaipur': 'RJ', 'jodhpur': 'RJ', 'kota': 'RJ',
    'kerala': 'KL', 'kl': 'KL', 'kochi': 'KL', 'thiruvananthapuram': 'KL', 'kozhikode': 'KL',
    'punjab': 'PB', 'pb': 'PB', 'ludhiana': 'PB', 'amritsar': 'PB', 'jalandhar': 'PB',
    'madhya pradesh': 'MP', 'mp': 'MP', 'indore': 'MP', 'bhopal': 'MP', 'gwalior': 'MP',
    'odisha': 'OR', 'or': 'OR', 'orissa': 'OR', 'bhubaneswar': 'OR', 'cuttack': 'OR',
    'bihar': 'BR', 'br': 'BR', 'patna': 'BR',
    'assam': 'AS', 'as': 'AS', 'guwahati': 'AS',
    'chandigarh': 'CH', 'ch': 'CH',
    'jharkhand': 'JH', 'jh': 'JH', 'ranchi': 'JH', 'dhanbad': 'JH',
    'chhattisgarh': 'CG', 'cg': 'CG', 'raipur': 'CG',
    'uttarakhand': 'UK', 'uk': 'UK', 'dehradun': 'UK',
    'goa': 'GA', 'ga': 'GA',
}

RE_INDIC = re.compile(r'[\u0900-\u0D7F]')
RE_WORD = re.compile(r'\w+')
RE_NUM = re.compile(r'\b\d+\b')
RE_PIN = re.compile(r'\b[1-9]\d{5}\b')
RE_CLEAN_WORD = re.compile(r'\b[a-zA-Z]{2,}\b')
STOPS = {'india', 'near', 'opp', 'opposite', 'road', 'rd', 'street', 'st', 'floor', 'flr', 'no', 'plot', 'ltd', 'pvt'}


def is_india(country_str):
    return bool(country_str and country_str.strip().lower() == "india")


def extract_address_tokens(addr):
    if not addr: return set()
    return {w for w in RE_WORD.findall(addr.lower()) if len(w) > 1 and w not in STOPS}


def extract_numbers(addr):
    if not addr: return set()
    return set(RE_NUM.findall(addr.lower()))


def extract_pincode(addr):
    if not addr: return None
    pins = RE_PIN.findall(addr)
    return pins[0] if pins else None


def extract_state(addr):
    if not addr: return None
    lower_addr = addr.lower()
    for mw in ['tamil nadu', 'andhra pradesh', 'west bengal', 'uttar pradesh', 'madhya pradesh', 'new delhi']:
        if mw in lower_addr:
            return STATES[mw]
    words = [w.lower() for w in RE_CLEAN_WORD.findall(addr)]
    for w in words:
        if w in STATES:
            return STATES[w]
    return None


def compute_f_beta(precision, recall, beta=0.5):
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return ((1 + beta_sq) * precision * recall) / denom


def evaluate(num_eval_anchors=10000, batch_size=512):
    total_start = time.time()
    os.makedirs("output", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    report_lines = []
    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=" * 85)
    log("🚀 STATE-AWARE PRECISION EVALUATION ON TRAIN DATASET & GROUND TRUTH")
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
    cand_states = []
    cand_pins = []
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
                    addr = row[2].strip() if len(row) > 2 else ""
                    cand_tokens.append(extract_address_tokens(addr))
                    cand_numbers.append(extract_numbers(addr))
                    cand_states.append(extract_state(addr))
                    cand_pins.append(extract_pincode(addr))
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
    gt_indic_matches = defaultdict(set)
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

    log(f"[+] Found {len(all_s1_with_indic_gt):,} S1 entities in Train with true Indic matches (in {time.time()-t0:.2f}s)")

    # -----------------------------------------------------------------------
    # Step 3: Select Evaluation Anchors (Balanced: Positives + Negatives)
    # -----------------------------------------------------------------------
    num_pos = num_eval_anchors // 2
    num_neg = num_eval_anchors - num_pos

    eval_s1_records = []
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
    # Step 4: Encode Candidate Embeddings on GPU
    # -----------------------------------------------------------------------
    model = SentenceTransformer(MODEL_NAME, device=device)
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
    log(f"[+] Candidate embeddings encoded in {time.time()-t0:.2f}s")
    del cand_names
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Step 5: Run Precision-Maximized Matching Pipeline
    # -----------------------------------------------------------------------
    log(f"\n[*] Step 5: Running State-Veto & Precision-Clamped Pipeline on {len(eval_s1_records):,} S1 anchors...")
    t0 = time.time()

    clamped_preds = defaultdict(set)
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

        for local_pos in range(len(batch)):
            s1_id = b_ids[local_pos]
            s1_addr = b_addrs[local_pos]
            s1_state = extract_state(s1_addr)
            s1_pin = extract_pincode(s1_addr)
            s1_toks = extract_address_tokens(s1_addr)
            s1_nums = extract_numbers(s1_addr)

            best_s2_score = 0.0
            best_s2_id = None
            best_s3_score = 0.0
            best_s3_id = None

            s_sims = sims[local_pos]

            # 1. High Confidence Tier 1 with State Compatibility Check
            top_scores, top_indices = torch.topk(s_sims, k=min(20, len(cand_ids)))
            top_scores_np = top_scores.cpu().numpy()
            top_indices_np = top_indices.cpu().numpy()

            for score, c_idx in zip(top_scores_np, top_indices_np):
                if score >= 0.92:
                    c_state = cand_states[c_idx]
                    # Veto if states explicitly contradict (e.g. MH vs TN)
                    if s1_state and c_state and s1_state != c_state:
                        continue
                    
                    cid = cand_ids[c_idx]
                    if cid.startswith("S2-") and score > best_s2_score:
                        best_s2_score = score
                        best_s2_id = cid
                    elif cid.startswith("S3-") and score > best_s3_score:
                        best_s3_score = score
                        best_s3_id = cid

            # 2. Tier 2: Address Bridging (Address Overlap & PIN code boost)
            if s1_toks:
                cand_hits = Counter()
                for t in s1_toks:
                    if t in inv_index:
                        for cid in inv_index[t]:
                            cand_hits[cid] += 1

                for cid, hits in cand_hits.items():
                    c_state = cand_states[cid]
                    # State Veto
                    if s1_state and c_state and s1_state != c_state:
                        continue

                    c_pin = cand_pins[cid]
                    has_pin_match = (s1_pin and c_pin and s1_pin == c_pin)
                    has_num_match = bool(s1_nums & cand_numbers[cid]) if (s1_nums and cand_numbers[cid]) else False

                    # Acceptance Conditions:
                    # Path A: PIN code match (gold standard) with Sim >= 0.40
                    # Path B: Number match with Overlap >= 4 and Sim >= 0.45
                    # Path C: Strong token overlap >= 6 and Sim >= 0.50
                    score = s_sims[cid].item()
                    is_valid = False
                    if has_pin_match and score >= 0.40:
                        is_valid = True
                    elif has_num_match and hits >= 4 and score >= 0.45:
                        is_valid = True
                    elif hits >= 6 and score >= 0.50:
                        is_valid = True

                    if is_valid:
                        target_id = cand_ids[cid]
                        if target_id.startswith("S2-") and score > best_s2_score:
                            best_s2_score = score
                            best_s2_id = target_id
                        elif target_id.startswith("S3-") and score > best_s3_score:
                            best_s3_score = score
                            best_s3_id = target_id

            if best_s2_id:
                clamped_preds[s1_id].add(best_s2_id)
            if best_s3_id:
                clamped_preds[s1_id].add(best_s3_id)

        del b_embs
        del sims

    # -----------------------------------------------------------------------
    # Step 6: Compute Exact Metrics
    # -----------------------------------------------------------------------
    log("\n" + "=" * 85)
    log("FINAL GROUND TRUTH EVALUATION METRICS (WITH STATE VETO & PRECISION CLAMPING)")
    log("=" * 85)

    tp_total = 0
    fp_total = 0
    fn_total = 0
    f05_macro_list = []
    singleton_correct = 0
    singleton_total = 0

    for s1_id, _, _ in eval_s1_records:
        gt_set = gt_indic_matches.get(s1_id, set())
        pred_set = clamped_preds.get(s1_id, set())

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

    log(f"Precision:                     {precision*100:>8.2f}%")
    log(f"Recall (Indic Ground Truth):   {recall*100:>8.2f}%")
    log(f"F1 Score (Harmonic Mean):      {f1*100:>8.2f}%")
    log(f"Micro F_0.5 Score:             {micro_f05*100:>8.2f}%")
    log(f"Macro F_0.5 (Official Metric): {macro_f05*100:>8.2f}%")
    log(f"Singleton Accuracy:            {singleton_acc:>8.2f}%")
    log(f"True Positives (TP):           {tp_total:>8,}")
    log(f"False Positives (FP):          {fp_total:>8,}")
    log(f"False Negatives (FN):          {fn_total:>8,}")
    log("-" * 85)
    log(f"[+] Total execution time: {time.time()-total_start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-anchors", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    evaluate(num_eval_anchors=args.num_anchors, batch_size=args.batch_size)
