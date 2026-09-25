# 🚀 DGX ULTRA-HEAVY MISSION PROMPT: DEDICATED RETRIEVAL & RE-RANKING STACK (ZERO GENERATIVE LLMs, 100% DISCRIMINATIVE, ZERO-OOM)

> **Execution Machine:** NVIDIA DGX (Multi-GPU Linux Server, A100 / H100 / GB10)  
> **Model Policy:** **STRICTLY NO QWEN / NO GENERATIVE LLMs**. Exclusively deploy high-throughput, deterministic embedding and cross-encoder ranking architectures (**BAAI/bge-m3** + **bge-reranker-v2-m3** + **18-D Tri-Model GBDT Ensemble**).  
> **Competition Constraint:** Apache 2.0 / MIT License, Strictly $\le$ 8 Billion Parameters  
> **Primary Objective:** Maximize Macro $F_{0.5}$ using dedicated Discriminative Transformers and GPU Trees with **FP16 / Chunked Tensor Streaming** so the DGX **NEVER crashes or encounters Out-of-Memory (OOM)**.

---

## 🏛️ ARCHITECTURE: PURE DISCRIMINATIVE 100% DETERMINISTIC STACK

We avoid generative text LLMs (which are slow and can hallucinate) in favor of **State-of-the-Art Deep Embedding & Cross-Encoder Architectures**:

```
[ Test Queries: 1.73M S1 Entities ]
                 │
                 ▼ STAGE 1: Fast Country & Geo Gating
    (Strict Country Slicing: US, India, France — 0% Cross-Country Pairs)
                 │
                 ▼ STAGE 2: High-Recall Multi-Key Blocking (Top 10-15 Cands)
    (Normalized Name + Address Pincode + Trigram Inverted Index)
                 │
                 ▼ STAGE 3: Heavy Multi-Lingual Bi-Encoder Semantic Retrieval
┌────────────────────────────────────────────────────────────────────────┐
│ MODEL: BAAI/bge-m3 (568M Params, 8192 Token Context, Apache 2.0)       │
│ - Encodes entire (Cleaned Name + Full Address) across 100+ languages   │
│ - Unifies Dense Vector, Sparse (BM25-style), and ColBERT Multi-Vector  │
│ - Runs in native torch.float16 with torch.cuda.empty_cache()           │
└────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼ STAGE 4: Dedicated Cross-Encoder Re-Ranking (The Precision Shield)
┌────────────────────────────────────────────────────────────────────────┐
│ MODEL: BAAI/bge-reranker-v2-m3 (568M Params, Apache 2.0)               │
│ - Pure Discriminative Cross-Encoder (NO generative text generation)    │
│ - Computes full cross-attention over "[S1 Entity] <-> [Candidate]"    │
│ - Extremely sensitive to subtle differences in address / name typos    │
│ - Decision Threshold: >= 0.70 for candidate acceptance                 │
└────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼ STAGE 5: Tri-Model GBDT Ensemble + Graph Transitivity
┌────────────────────────────────────────────────────────────────────────┐
│ - 18-Dimensional Feature Matrix:                                       │
│     * XGBoost GPU (Depth-wise hist)                                    │
│     * LightGBM (Leaf-wise fast trees)                                  │
│     * CatBoost GPU (Symmetric oblivious trees)                         │
│ - Graph Transitivity Engine: S1 <-> S2 <-> S3 triangle cycle recovery  │
│ - Final Clamping: Strictly Max 1 S2 + Max 1 S3 per entity             │
└────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
[ Output TSV: 1,732,544 rows, Single-Tab Delimited, Zero Hallucinations ]
```

### Parameter & License Compliance Audit:
- **`bge-m3` Bi-Encoder:** 568 Million parameters (Apache 2.0)
- **`bge-reranker-v2-m3` Cross-Encoder:** 568 Million parameters (Apache 2.0)
- **Tri-Model GBDT Ensemble (XGB + LGB + CAT):** 6.5 Million parameters (MIT / Apache 2.0)
- **Total System:** **$\sim 1.14 \text{ Billion Parameters} \ll 8.0 \text{ Billion Limit}$** (**100% Apache 2.0 / MIT, Fully Compliant**).

---

## 🛡️ ANTI-CRASH & ZERO-OOM SYSTEM SAFEGUARDS ON DGX

To ensure the DGX **never freezes, locks up, or throws CUDA OOM**, the pipeline enforces 5 memory barriers:

1. **FP16 Mixed Precision:** All embeddings and model weights are cast to `torch.float16` (50% VRAM cut).
2. **Dynamic Streaming Batches (`batch_size=256` or `512`):** S1 entities are processed in micro-batches; tensors are deleted immediately after computation (`del b_embs, sims; torch.cuda.empty_cache()`).
3. **Precomputed Candidate Tensor Cache (`.pt` on disk):** S2 and S3 candidate tensors are stored on NVMe disk and memory-mapped, avoiding redundant forward passes.
4. **Garbage Collection Cadence:** Python `gc.collect()` runs every 5,000 entities to prevent memory fragmentation.
5. **VRAM Safety Cap:** 85% VRAM threshold guard.

---

## ⚡ 3-STEP COMMAND SEQUENCE ON DGX

### STEP 1: Git Sync & Install Libraries
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
