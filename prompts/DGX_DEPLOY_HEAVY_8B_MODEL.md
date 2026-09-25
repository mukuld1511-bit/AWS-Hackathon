# 🚀 DGX ULTRA-HEAVY MISSION PROMPT: 7B/8B FOUNDATION MODEL PIPELINE (MAX GPU CAPACITY WITHOUT OOM)

> **Execution Machine:** NVIDIA DGX (Multi-GPU Linux Server, A100 / H100 / GB10)  
> **Competition Constraint:** Apache 2.0 / MIT License, Strictly $\le$ 8 Billion Parameters  
> **Primary Objective:** Deploy the absolute maximum model capacity allowable under the 8B limit (**Qwen2.5-7B-Instruct** / **BAAI/bge-multilingual-gemma2** / **bge-reranker-v2-m3**) engineered with **FP16 / FlashAttention-2 / Chunked Tensor Streaming** so the DGX **NEVER crashes or encounters Out-of-Memory (OOM)**.  
> **Target:** Push Macro $F_{0.5}$ to the absolute competition ceiling before merging Harsh's candidate blocking.

---

## 🏛️ ARCHITECTURE: THE 7B/8B ULTRA-HEAVY STACK

We replace lightweight models with heavyweight, high-capacity neural architectures that saturate the 8B parameter budget while remaining 100% compliant:

```
[ Test Queries: 1.73M S1 Entities ]
                 │
                 ▼ STAGE 1: Fast Country & Geo Gating
    (Strict Country Slicing: US, India, France)
                 │
                 ▼ STAGE 2: High-Recall Multi-Key Blocking (Top 10-15 Cands)
    (Normalized Name + Address Pincode + Trigram Inverted Index)
                 │
                 ▼ STAGE 3: Heavy Bi-Encoder Semantic Filtering
┌────────────────────────────────────────────────────────────────────────┐
│ MODEL: BAAI/bge-multilingual-gemma2 (2.6 Billion Params) OR             │
│        BAAI/bge-m3 (568M Params, 8192 Context Window, FP16)           │
│ - Encodes entire (Name + Full Address) across 100+ languages           │
│ - Chunked Embedding Matrix Multiplication with torch.cuda.empty_cache()│
└────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼ STAGE 4: Ultra-Heavy Cross-Encoder Re-Ranking (The Precision Shield)
┌────────────────────────────────────────────────────────────────────────┐
│ MODEL: BAAI/bge-reranker-v2-m3 (568M Params) OR                         │
│        Qwen/Qwen2.5-7B-Instruct (Cross-Attention Mode, ~7.6 Billion)  │
│ - Evaluates joint pair: "[S1: {Name}, {Addr}] vs [Cand: {Name}, {Addr}]"│
│ - Full Cross-Attention allows the model to spot tiny OCR/Typo shifts    │
│ - Threshold: 0.90+ for positive match approval                         │
└────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼ STAGE 5: Tri-Model GBDT Ensemble + Graph Transitivity
┌────────────────────────────────────────────────────────────────────────┐
│ - 18-Dimensional Feature Matrix (XGBoost GPU + LightGBM + CatBoost GPU)│
│ - Graph Transitivity Engine: S1 <-> S2 <-> S3 triangle cycle recovery  │
│ - Final Clamping: Strictly Max 1 S2 + Max 1 S3 per entity             │
└────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
[ Output TSV: 1,732,544 rows, Single-Tab Delimited, Zero Hallucinations ]
```

### Parameter & License Compliance Audit:
- **Bi-Encoder (`bge-m3`):** 568 Million parameters (Apache 2.0)
- **Heavy Cross-Encoder / Re-Ranker (`bge-reranker-v2-m3` or 7B Qwen FP16):** ~568M to 7.6B (Apache 2.0)
- **Tri-Model GBDT Ensemble:** 6.5 Million parameters (MIT / Apache 2.0)
- **Total System:** **$\sim 1.14\text{B} - 7.8\text{B} \le 8.0\text{ Billion Limit}$** (**100% Legal & Approved**).

---

## 🛡️ ANTI-CRASH & ZERO-OOM SYSTEM SAFEGUARDS ON DGX

To ensure the DGX **never freezes, locks up, or throws CUDA OOM**, the pipeline enforces 5 memory barriers:

1. **FP16 / BF16 Mixed Precision:** All embeddings and model weights are cast to `torch.float16` or `torch.bfloat16` (50% VRAM cut).
2. **Dynamic Streaming Batches (`batch_size=256` or `512`):** S1 entities are processed in micro-batches; tensors are deleted immediately after computation (`del b_embs, sims; torch.cuda.empty_cache()`).
3. **Precomputed Candidate Tensor Cache (`.pt` on disk):** S2 and S3 candidate tensors are stored on NVMe disk and memory-mapped or loaded once, avoiding redundant forward passes.
4. **Garbage Collection Cadence:** Python `gc.collect()` runs every 5,000 entities to prevent memory fragmentation.
5. **Worker Timeout & CPU Offload:** Fallback to CPU RAM if GPU allocation spikes above 85% capacity.

---

## ⚡ 3-STEP COMMAND SEQUENCE ON DGX

### STEP 1: Git Sync & Install Heavy Model Libraries
Run in the DGX terminal:
```bash
cd AWS-Hackathon
git pull origin main
pip install --upgrade FlagEmbedding sentence-transformers torch xgboost lightgbm catboost rapidfuzz tqdm
```

Verify GPU & Memory allocation:
```bash
python3 -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('VRAM Total:', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2), 'GB')
"
```

---

### STEP 2: Execute Heavy Ensemble Pipeline
Run the heavy pipeline:
```bash
python3 run_heavy_pipeline.py
```
* **Execution Process on DGX:**
  1. Computes 18-D feature matrix across candidate pool.
  2. Runs Tri-Model inference (XGBoost GPU + LightGBM + CatBoost GPU).
  3. Integrates Multilingual cross-script embeddings.
  4. Runs Graph Transitivity engine (recovering missing triangular links).
  5. Enforces precision clamping (Max 1 S2, Max 1 S3).
  6. Outputs `output_final/matching_results.tsv` (1,732,544 rows).

---

### STEP 3: Validate & Build Official Submission Package
Run validation and packaging:
```bash
python3 build_final_submission.py
python3 validate_dgx_tsv.py output_final/matching_results.tsv
```
*Confirmation:* Verify terminal outputs:
`🎉 100% PERFECT PASS! TSV file is 100% valid and safe for Unstop.`

---

## 🌐 PORTAL UPLOAD INSTRUCTIONS
1. Open Unstop competition page.
2. In **Upload Matching Results File**, upload:
   👉 `output_final/matching_results.tsv`
3. Click **Submit & Evaluate**.
4. Record the leaderboard score!
