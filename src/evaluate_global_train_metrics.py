#!/usr/bin/env python3
"""
src/evaluate_global_train_metrics.py

Computes exact Precision, Recall, Macro F0.5, and Accuracy of our Global Multi-Country Matcher
against Train Ground Truth (2,206,821 ground truth records across US and India).
"""

import os
import re
import csv
import sys
import time
import argparse
from collections import defaultdict, Counter

if "HF_HOME" not in os.environ:
    local_cache = os.path.abspath(".venv/hf_cache")
    if os.path.exists(".venv"):
        os.environ["HF_HOME"] = local_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Training Files
TRAIN_S1 = "student_resource/dataset/train/train_source1.tsv"
TRAIN_S2 = "student_resource/dataset/train/train_source2.tsv"
TRAIN_S3 = "student_resource/dataset/train/train_source3.tsv"
TRAIN_GT = "student_resource/dataset/train/train_ground_truth.tsv"
FINE_TUNED_MODEL = "output/fine_tuned_multilingual_entity_model"
BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

LEGAL_SUFFIXES_PATTERN = re.compile(
    r'\b(inc|incorporated|llc|llp|ltd|limited|pvt|private|corp|corporation|co|company|enterprises|enterprise|group|services|center|sarl|sas|eurl|sa|gmbh)\b',
    re.IGNORECASE
)
PUNCT_PATTERN = re.compile(r'[^\w\s]', re.UNICODE)
RE_WORD = re.compile(r'\w+')
RE_NUM = re.compile(r'\b\d+\b')
RE_ZIP = re.compile(r'\b\d{4,6}\b')
STOPS = {'india', 'usa', 'us', 'france', 'near', 'opp', 'opposite', 'road', 'rd', 'street', 'st', 'floor', 'flr', 'no', 'plot', 'ltd', 'pvt'}

def normalize_name(name: str) -> str:
    if not name: return ""
    text = PUNCT_PATTERN.sub(' ', name.lower())
    text = LEGAL_SUFFIXES_PATTERN.sub(' ', text)
    return " ".join(text.split())

def extract_address_tokens(addr: str):
    if not addr: return set()
    return {w for w in RE_WORD.findall(addr.lower()) if len(w) > 1 and w not in STOPS}

def extract_postal_code(addr: str):
    if not addr: return None
    codes = RE_ZIP.findall(addr)
    return codes[-1] if codes else None

def extract_numbers(addr: str):
    if not addr: return set()
    return set(RE_NUM.findall(addr.lower()))

def evaluate_on_train(sample_size=100_000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = FINE_TUNED_MODEL if os.path.exists(FINE_TUNED_MODEL) else BASE_MODEL
    print("=" * 70)
    print(f"📊 EVALUATING GLOBAL MULTI-COUNTRY METRICS ON TRAIN GROUND TRUTH")
    print(f"[*] Evaluation Sample Size: {sample_size:,} rows | Device: {device}")
    print(f"[*] Model: {model_path}")
    print("=" * 70)

    # 1. Load Ground Truth
    print(f"[*] Loading ground truth from {TRAIN_GT}...")
    ground_truth = {}
    with open(TRAIN_GT, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for i, row in enumerate(reader):
            if i >= sample_size:
                break
            s1_id = row[0].strip()
            matches = set(m.strip() for m in row[1].split(",") if m.strip()) if len(row) > 1 else set()
            ground_truth[s1_id] = matches
    print(f"[+] Loaded ground truth for {len(ground_truth):,} entities.")

    # 2. Build S2 and S3 Index on Train
    print("[*] Building candidate index on Train Source 2 & Source 3...")
    t0 = time.time()
    exact_index = defaultdict(list)
    postal_index = defaultdict(list)
    cand_ids = []
    cand_names = []
    cand_addrs = []
    cand_tokens = []
    cand_numbers = []

    for src_path in [TRAIN_S2, TRAIN_S3]:
        with open(src_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) < 4: continue
                eid, b_name, b_addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                cid = len(cand_ids)
                cand_ids.append(eid)
                cand_names.append(b_name)
                cand_addrs.append(b_addr)
                
                norm_n = normalize_name(b_name)
                if norm_n and country:
                    exact_index[(country.lower(), norm_n)].append(cid)
                
                p_code = extract_postal_code(b_addr)
                if p_code and country:
                    postal_index[(country.lower(), p_code)].append(cid)

                cand_tokens.append(extract_address_tokens(b_addr))
                cand_numbers.append(extract_numbers(b_addr))

    print(f"[+] Indexed {len(cand_ids):,} train candidates in {time.time()-t0:.2f}s")

    # 3. Match S1 Sample
    print(f"[*] Matching {len(ground_truth):,} Train S1 queries...")
    t_match = time.time()
    
    tp, fp, fn = 0, 0, 0
    singleton_correct = 0
    singleton_total = 0
    exact_entity_matches = 0

    with open(TRAIN_S1, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        
        for i, row in enumerate(reader):
            if i >= sample_size:
                break
            if len(row) < 4: continue
            s1_id, name, addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
            c_lower = country.lower()
            norm_n = normalize_name(name)
            s1_toks = extract_address_tokens(addr)
            s1_nums = extract_numbers(addr)

            gt_matches = ground_truth.get(s1_id, set())

            # Prediction
            pred_matches = set()
            exact_cids = exact_index.get((c_lower, norm_n), [])
            if exact_cids:
                best_s2, best_s3 = None, None
                for cid in exact_cids:
                    cid_target = cand_ids[cid]
                    is_s2 = cid_target.startswith(("S2-", "S2_"))
                    is_s3 = cid_target.startswith(("S3-", "S3_"))
                    t_overlap = len(s1_toks.intersection(cand_tokens[cid]))
                    num_overlap = len(s1_nums.intersection(cand_numbers[cid]))
                    score = t_overlap * 2 + num_overlap * 3

                # Require minimum address/number/geo compatibility: score >= 2 (or score >= 1 if bucket size <= 2)
                bucket_len = len(exact_cids)
                min_req = 1 if bucket_len <= 2 else 2

                if best_s2 and best_s2[1] >= min_req: 
                    pred_matches.add(best_s2[0])
                if best_s3 and best_s3[1] >= min_req: 
                    pred_matches.add(best_s3[0])

            # Evaluate
            if not gt_matches:
                singleton_total += 1
                if not pred_matches:
                    singleton_correct += 1
                else:
                    fp += len(pred_matches)
            else:
                cur_tp = len(pred_matches.intersection(gt_matches))
                cur_fp = len(pred_matches - gt_matches)
                cur_fn = len(gt_matches - pred_matches)

                tp += cur_tp
                fp += cur_fp
                fn += cur_fn

                if pred_matches == gt_matches:
                    exact_entity_matches += 1

    # Metrics computation
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta = 0.5
    f05 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if ((beta**2 * precision) + recall) > 0 else 0.0
    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0
    overall_exact_acc = exact_entity_matches / len(ground_truth)

    print("\n" + "=" * 70)
    print("📈 FINAL BENCHMARK METRICS SUMMARY")
    print("=" * 70)
    print(f"[*] True Positives (TP):     {tp:,}")
    print(f"[*] False Positives (FP):    {fp:,}")
    print(f"[*] False Negatives (FN):    {fn:,}")
    print("-" * 70)
    print(f"🎯 Precision:                {precision * 100:.2f}%")
    print(f"🎯 Recall:                   {recall * 100:.2f}%")
    print(f"🏆 Macro F0.5 Score:         {f05 * 100:.2f}%")
    print(f"🎯 Singleton Accuracy:       {singleton_acc * 100:.2f}%")
    print(f"🎯 Exact Entity Accuracy:    {overall_exact_acc * 100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=100_000)
    args = parser.parse_args()
    evaluate_on_train(sample_size=args.sample_size)
