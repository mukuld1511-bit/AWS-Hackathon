#!/usr/bin/env python3
"""
src/train_multilingual_model.py

Native PyTorch Fine-Tuning of Multilingual Sentence Transformer on Business Entity Resolution Ground Truth
across ALL countries (US, India, multilingual) using MultipleNegativesRankingLoss (InfoNCE).
No external 'datasets' dependency required.
"""

import os
import sys
import re
import csv
import time
import random
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW

if "HF_HOME" not in os.environ:
    local_cache = os.path.abspath(".venv/hf_cache")
    if os.path.exists(".venv"):
        os.environ["HF_HOME"] = local_cache

from sentence_transformers import SentenceTransformer

BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_MODEL_DIR = "output/fine_tuned_multilingual_entity_model"
TRAIN_S1 = "student_resource/dataset/train/train_source1.tsv"
TRAIN_S2 = "student_resource/dataset/train/train_source2.tsv"
TRAIN_S3 = "student_resource/dataset/train/train_source3.tsv"
TRAIN_GT = "student_resource/dataset/train/train_ground_truth.tsv"

def load_entity_records(tsv_path):
    print(f"[*] Loading records from {tsv_path}...")
    records = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                eid = row[0].strip()
                name = row[1].strip() if len(row) > 1 else ""
                addr = row[2].strip() if len(row) > 2 else ""
                country = row[3].strip() if len(row) > 3 else ""
                if name:
                    text = f"{name} | {addr}" if addr else name
                    records[eid] = (name, text, country)
    print(f"[+] Loaded {len(records):,} records from {os.path.basename(tsv_path)}")
    return records

def prepare_training_pairs(max_pairs=100_000):
    print("\n" + "=" * 70)
    print("🚀 PREPARING MULTILINGUAL TRAINING PAIRS FROM ALL COUNTRIES")
    print("=" * 70)

    s1_dict = load_entity_records(TRAIN_S1)
    
    targets = {}
    for path in [TRAIN_S2, TRAIN_S3]:
        targets.update(load_entity_records(path))

    pairs = []
    print(f"[*] Reading ground truth pairs from {TRAIN_GT}...")
    with open(TRAIN_GT, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 2 or not row[1].strip():
                continue
            s1_id = row[0].strip()
            matched_ids = [m.strip() for m in row[1].split(",") if m.strip()]

            if s1_id not in s1_dict:
                continue

            _, s1_full, _ = s1_dict[s1_id]

            for m_id in matched_ids:
                if m_id in targets:
                    _, m_full, _ = targets[m_id]
                    pairs.append((s1_full, m_full))

    print(f"[+] Total positive ground truth pairs found: {len(pairs):,}")
    
    random.seed(42)
    random.shuffle(pairs)
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
        print(f"[!] Downsampled to {len(pairs):,} pairs for rapid optimal training.")

    return pairs

def train(epochs=1, batch_size=256, lr=2e-5, max_pairs=80_000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] Initializing Base Model: {BASE_MODEL} on device: {device}")
    model = SentenceTransformer(BASE_MODEL, device=device)
    model.train()

    pairs = prepare_training_pairs(max_pairs=max_pairs)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    num_batches = (len(pairs) + batch_size - 1) // batch_size
    print("\n" + "=" * 70)
    print(f"🔥 STARTING NATIVE PYTORCH FINE-TUNING")
    print(f"[*] Pairs: {len(pairs):,} | Batches per epoch: {num_batches:,} | Batch Size: {batch_size}")
    print("=" * 70)

    start_time = time.time()
    
    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0.0
        
        for b_idx in range(num_batches):
            batch_pairs = pairs[b_idx * batch_size : (b_idx + 1) * batch_size]
            if len(batch_pairs) < 4:
                continue
            
            texts_a = [p[0] for p in batch_pairs]
            texts_b = [p[1] for p in batch_pairs]
            
            feat_a = model.tokenizer(texts_a, padding=True, truncation=True, max_length=128, return_tensors="pt")
            feat_b = model.tokenizer(texts_b, padding=True, truncation=True, max_length=128, return_tensors="pt")
            
            feat_a = {k: v.to(device) for k, v in feat_a.items()}
            feat_b = {k: v.to(device) for k, v in feat_b.items()}
            
            emb_a = model(feat_a)["sentence_embedding"]
            emb_b = model(feat_b)["sentence_embedding"]
            
            emb_a = F.normalize(emb_a, p=2, dim=1)
            emb_b = F.normalize(emb_b, p=2, dim=1)
            
            # InfoNCE (Multiple Negatives Ranking Loss with temperature 0.05 / scale 20.0)
            scores = torch.matmul(emb_a, emb_b.T) * 20.0
            labels = torch.arange(len(batch_pairs), device=device)
            
            loss = F.cross_entropy(scores, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == num_batches:
                avg_loss = total_loss / (b_idx + 1)
                speed = ((b_idx + 1) * batch_size) / (time.time() - start_time)
                print(f"    Epoch {epoch+1}/{epochs} | Batch {b_idx+1:>4}/{num_batches} | Avg Loss: {avg_loss:.4f} | Speed: {speed:.0f} pairs/s")

    elapsed = time.time() - start_time
    print(f"\n[+] Fine-Tuning completed in {elapsed/60:.2f} minutes!")

    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    print(f"[*] Saving fine-tuned model to {OUTPUT_MODEL_DIR}...")
    model.save(OUTPUT_MODEL_DIR)
    print(f"🎉 Fine-tuned model successfully saved to {OUTPUT_MODEL_DIR}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Native fine-tune multilingual model on all countries.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_pairs", type=int, default=80_000, help="Max training pairs")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_pairs=args.max_pairs)
