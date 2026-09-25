# 🎯 PRATEEK TASK PROMPT v2: Multilingual Indic Matcher — Production Finalization

> **Assigned Engineer:** Prateek  
> **Git Branch:** `prateek`  
> **Target Module:** `src/multilingual_matcher.py` & `src/validate_multilingual_rules.py`  
> **Target Deliverable:** Complete, Bug-Free, High-Precision Multilingual Model Ready for Mainline Integration  
> **Evaluation Metric:** Macro $F_{0.5}$ (Precision is weighted 2× over Recall. False Positives are heavily penalized).

---

## 🚨 CRITICAL POST-MORTEM & RULES (WHY PREVIOUS VERSION FAILED / WAS RISKY)

Prateek, your multilingual transliteration idea using `paraphrase-multilingual-MiniLM-L12-v2` is **conceptually brilliant**, but the initial implementation had **two fatal flaws** that must be fixed in this iteration before we can merge to `main`:

### ❌ Flaw 1: `MAX_MATCHES_PER_ENTITY = 15` (Precision Killer)
- In the ground truth, a Source 1 entity has **at most ONE true match from Source 2** and **at most ONE true match from Source 3** (Total $\le 2$ matches).
- Allowing up to 15 matches causes massive False Merges. Under Macro $F_{0.5}$, if you predict 10 matches for 1 true match, Precision drops to $1/10 = 0.10$, and the score drops from **0.415 down to ~0.15**!
- **Strict Rule:** For each Source 1 entity, you must **NEVER output more than 1 ID starting with `S2-` and 1 ID starting with `S3-`**.

### ❌ Flaw 2: Output File Missing Non-India Entities (Unstop Rejection)
- Unstop requires **exactly 1,732,544 rows** (every single Source 1 test entity must appear in `matching_results.tsv`).
- Streaming only India entities leaves out US and France (~900k rows missing), causing the Unstop evaluation server to immediately mark the submission as **"FAILED"**.
- **Strict Rule:** Your module must produce an internal output dictionary or candidate map where every Source 1 entity is accounted for (non-India or unmatched entities simply get empty matches `""`).

---

## 🏗️ YOUR ARCHITECTURAL BLUEPRINT (VERSION 2)

```
Test S1 (1,732,544 rows)
       │
       ▼ Country Split
┌───────────────────────────────┐
│ Is Country == 'India'?        │
└───────────────┬───────────────┘
                │
        YES ────┴──── NO ────────────────────────┐
        │                                        │
        ▼                                        ▼
[Indic Multilingual Engine]             [Pass Through / Empty]
  1. Dense Cosine Matrix                  (Leaves space for GBDT /
  2. Two-Tier Decision Rule:               Other models to fill US/FR)
     - Tier 1: Cosine >= 0.92
     - Tier 2: Cosine >= 0.45 AND
               Address Overlap >= 6
  3. Strict Selection:
     - Best S2 candidate (max score >= thresh)
     - Best S3 candidate (max score >= thresh)
        │                                        │
        └───────────────────┬────────────────────┘
                            │
                            ▼
           Full 1,732,544 Rows Output TSV
           (Strict Format: S1-ID \t S2-xxx,S3-yyy)
```

---

## 🛠️ EXACT SPECIFICATIONS & CODE REQUIREMENTS FOR `src/multilingual_matcher.py`

### 1. Two-Tier Decision Rule (Calibrated for Precision):
```python
# Constants to use:
TIER1_COSINE_THRESHOLD = 0.92      # High confidence pure name transliteration
TIER2_COSINE_THRESHOLD = 0.45      # Name similarity with strong address backing
MIN_ADDRESS_TOKEN_OVERLAP = 6      # Minimum overlapping address tokens
RELAXED_ADDR_TOKEN_OVERLAP = 4     # If house number / pincode matches
```

### 2. Output Formatting Rule:
For every S1 record in `test_source1.tsv`:
- If matches found: `f"{s1_id}\t{','.join(selected_matches)}\n"` where `selected_matches` has **at most one S2 ID and at most one S3 ID**.
- If no match found or non-India: `f"{s1_id}\t\n"` (singleton).
- Total rows output must be **exactly 1,732,544**.

### 3. Checkpointing & Memory Safety:
- Retain candidate embeddings caching (`output/indic_candidates_cache.pt`).
- Keep chunk size at `25,000` with `torch.cuda.empty_cache()` and `gc.collect()`.
- Save progress in `output/matcher_checkpoint.json`.

---

## 🧪 STEP-BY-STEP WORKFLOW FOR PRATEEK

### Step 1: Work on your branch
```bash
git checkout prateek
git pull origin prateek
```

### Step 2: Implement & Validate Locally
Refactor `src/multilingual_matcher.py` according to the blueprint above.
Run validation on your output:
```bash
python3 src/multilingual_matcher.py --batch-size 512 --chunk-size 25000
python3 validate_dgx_tsv.py output/multilingual_matches.tsv
```
*Verification Checklist:*
- [ ] Total lines is exactly 1,732,545 (1 header + 1,732,544 data rows).
- [ ] No S1 entity has more than 1 S2 match or more than 1 S3 match.
- [ ] Zero duplicate IDs in any row.
- [ ] `validate_dgx_tsv.py` outputs `🎉 100% PERFECT PASS`.

### Step 3: Push to GitHub & Hand Off to Mukul
Once validation passes:
```bash
git add src/multilingual_matcher.py src/validate_multilingual_rules.py
git commit -m "feat(prateek): finalize production-ready Indic multilingual matcher with precision clamping"
git push origin prateek
```
Notify Mukul that branch `prateek` is ready for review and mainline merge!

---

## 🔄 THE COLLABORATION CYCLE

```
[Prateek on 'prateek' branch]
   │ 1. Finalize matcher with 1-S2 + 1-S3 clamp
   │ 2. Validate row count == 1,732,544
   │ 3. Push to origin/prateek
   ▼
[Mukul on 'main' branch]
   │ 1. Pull & Review diff
   │ 2. Merge prateek into main
   │ 3. Fuse with GBDT (0.415 model) via build_sub5_ensemble.py
   │ 4. Run DGX evaluation & upload to Unstop
   ▼
[Leaderboard Score Update]
   │ Score jumps (Target 0.65+ / 0.80+)
   ▼
[Next Iteration]
   │ Prateek adds fine-tuned transliteration / French addresses
```
