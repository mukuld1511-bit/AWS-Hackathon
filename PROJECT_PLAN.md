# 📌 Candidate Generation & Blocking Engine: Project Plan

> **Role & Responsibility:** Harsh (Lead: Candidate Generation & Multi-Key Blocking Engine)  
> **Workspace Branch:** `harsh`  
> **Target Output:** `output/candidate_pairs.tsv`  

---

## 1. Executive Summary & Core Objectives

### **Problem Context**
In large-scale entity resolution across 3 independent sources ($S_1$, $S_2$, $S_3$), a brute-force pairwise comparison between Test $S_1$ (1.73M records) and the combined Test target pool $S_2 + S_3$ (~10M records) requires **~17.3 Trillion pairwise comparisons**. This is computationally impossible and will crash system memory.

### **Assigned Responsibility**
My sole responsibility is to design, implement, and benchmark the **Candidate Generation / Blocking Engine**. 

The engine must prune the 10 Million search space down to a small, high-quality candidate shortlist of **approximately 15–25 candidate entity IDs** per $S_1$ record, while ensuring maximum **Candidate Recall**.

> [!CRITICAL]
> **Golden Rule of Candidate Generation:**
> **High Candidate Recall is mandatory.** If a true matching entity is pruned during candidate blocking, no downstream ML model (LightGBM, DeBERTa, Cross-Encoder) can ever recover it.

---

## 2. Environment & Dataset Inspection Audit

An empirical audit of the workspace environment and datasets was conducted:

### **A. Hardware Environment Audit**
* **Machine:** NVIDIA DGX Spark
* **CPU Cores:** 20 Physical Cores
* **System Memory:** 121.69 GB Unified RAM
* **Python Runtime:** Python 3.12.3

### **B. Dataset Location & File Schemas**
* **Relative Path:** `6ab10eb3b23ba_student_resource/student_resource/dataset/`
* **Train Files:**
  - `train_source1.tsv`: 2,206,821 records | Header: `entity_id\tbusiness_name\tbusiness_address\tcountry`
  - `train_source2.tsv`: 5,034,616 records | Header: `entity_id\tbusiness_name\tbusiness_address\tcountry`
  - `train_source3.tsv`: 5,285,603 records | Header: `entity_id\tbusiness_name\tbusiness_address\tcountry`
  - `train_ground_truth.tsv`: 2,206,821 rows | Header: `source1_entity_id\tmatched_entity_ids`
* **Test Files:**
  - `test_source1.tsv`: 1,732,544 records | Header: `entity_id\tbusiness_name\tbusiness_address\tcountry`
  - `test_source2.tsv`: 4,887,273 records | Header: `entity_id\tbusiness_name\tbusiness_address\tcountry`
  - `test_source3.tsv`: 5,082,316 records | Header: `entity_id\tbusiness_name\tbusiness_address\tcountry`

---

## 3. Core Architectural Rules & Constraints

### **Rule 1: Hard Country Partitioning (Open-Set Country Support)**
* Training data proves **0.000% cross-country matches**.
* **Constraint:** $S_1$ entities from a country $C$ only query candidates from $S_2/S_3$ records belonging to country $C$.
* **Open-Set Requirement:** Do NOT hardcode country lists to `{"US", "India"}`. Test data contains **France** (259k $S_1$ entities, ~1.4M target entities). Country must be dynamically handled as an open set of string labels.

### **Rule 2: Multiple Independent Blocking Keys**
To handle name typos, legal suffix variations, transliterations, and address noise, the engine implements **3 complementary blocking keys**:

#### **Key A: Exact Normalized Name Key**
* **Normalization Logic:**
  1. Lowercase text.
  2. Strip punctuation/symbols using regex `[^\w\s]`.
  3. Strip common legal suffixes (`Inc`, `LLC`, `LLP`, `Ltd`, `Private`, `Pvt`, `Corp`, `Sarl`, `Co`, `Group`, `Services`, `Center`, `Enterprises`, `Solutions`, `Associates`).
  4. Collapse extra whitespaces.
* **Index Token:** `(country, normalized_name)`

#### **Key B: Configurable Name Prefix + Address Numbers Key**
* **Logic:** Takes first $N$ characters of normalized name (default $N=4$, fully configurable) + digit sequences extracted from address.
* **Index Token:** `(country, prefix_N, address_numbers)`
* **Purpose:** Catches minor name typos (e.g. `Maure Wilblims` vs `Maure Williams`) when street numbers match.

#### **Key C: Physical Address Anchor Key**
* **Logic:** Extracts the first numeric token + first alphabetic word (>2 chars) from address (e.g., `85_wayne` for `85 Wayne Avenue`).
* **Index Token:** `(country, address_anchor)`
* **Purpose:** Catches different brand/trade names or website domain names residing at the same physical location.

---

### **Rule 3: Candidate Union & Evidence-Based Priority Ranking**
When querying Key A, Key B, and Key C for an $S_1$ entity:
1. Retrieve candidate postings from all 3 inverted index buckets.
2. Accumulate **blocking evidence weights**:
   - Candidate matching Key A: +3.0 points (Strongest)
   - Candidate matching Key B: +2.0 points (Medium)
   - Candidate matching Key C: +1.5 points (Medium)
3. **Deduplicate & Rank:** Candidates are sorted deterministically in descending order of accumulated evidence score.
4. **Deterministic Truncation:** Never truncate an unordered set arbitrarily. Keep the top 15–25 candidates based on evidence score.

---

### **Rule 4: Streaming Memory Management on DGX Spark**
* **No `pandas.read_csv()`**: Streaming line-by-line reading via `csv.reader(delimiter="\t")` to avoid loading raw multi-gigabyte DataFrames into RAM.
* **Memory Monitoring:** Measure and report:
  - Total records indexed
  - Unique keys per inverted index
  - Total postings / candidate IDs
  - Process Resident Set Size (RSS) RAM footprint
  - Candidate count distribution metrics (P50, P90, P95, P99, Max, Singletons)

---

## 4. Proposed Modular Code Architecture

To ensure clean engineering and maintainability, code will be structured into modular Python components:

```
src/
├── normalization.py       # Name & address string normalization & legal suffix regex
├── blocking_keys.py       # Key generation functions (Key A, Key B, Key C)
├── blocking_index.py      # Inverted index datastructures & lookup query engine
├── ranking.py             # Evidence scoring, deduplication & deterministic capping
├── evaluation.py          # Candidate Recall evaluator on training ground truth
├── utils.py               # Memory profiling, timer, header validator & logging
scripts/
├── generate_candidates.py # Main runnable script for test dataset candidate generation
└── evaluate_blocking.py   # Benchmark & candidate recall evaluation script
```

---

## 5. 14-Stage Development Roadmap

| Stage | Goal | Action Items |
| :---: | :--- | :--- |
| **Stage 1** | Normalization Utilities | Build `src/normalization.py` with comprehensive legal suffix regex & string cleaners. |
| **Stage 2** | Key A Implementation | Implement `generate_key_a(name, country)`. |
| **Stage 3** | Key B Implementation | Implement `generate_key_b(name, address, country, prefix_len=4)`. |
| **Stage 4** | Key C Implementation | Implement `generate_key_c(address, country)`. |
| **Stage 5** | Country Partitioning | Enforce strict country key prefix `(country, key_val)` across all index lookups. |
| **Stage 6** | Inverted Index Engine | Build `src/blocking_index.py` using streaming `csv.reader` over $S_2$ and $S_3$. |
| **Stage 7** | Query Processing | Stream $S_1$ entities and query index buckets. |
| **Stage 8** | Candidate Union & Ranking | Accumulate evidence scores, sort candidates deterministically, cap at 25. |
| **Stage 9** | TSV Output Writer | Write `output/candidate_pairs.tsv` incrementally. |
| **Stage 10** | Sample Benchmarking | Test pipeline on a 50k record sample; check RAM, runtime, and throughput. |
| **Stage 11** | Full Index Benchmarking | Benchmark on full train dataset; log index size and memory RSS. |
| **Stage 12** | Recall Evaluation | Calculate Candidate Recall on `train_ground_truth.tsv`. |
| **Stage 13** | Performance Optimization | Fine-tune prefix length, evidence weights, and memory structures. |
| **Stage 14** | Final Test Execution | Run on full 1.73M test $S_1$ entities and generate official `candidate_pairs.tsv`. |

---

## 6. Validation & Metric Benchmarks

### **Primary Accuracy Metric: Candidate Recall**

$$\text{Candidate Recall} = \frac{\sum_{i \in S_1} |\text{Generated Candidates}_i \cap \text{True Matches}_i|}{\sum_{i \in S_1} |\text{True Matches}_i|}$$

* **Target Recall Ceiling:** **> 98.5%** on training ground truth.

### **Efficiency & Distribution Metrics to Log:**
1. Indexing Runtime (seconds) & Throughput (records/sec)
2. Query Runtime (seconds) & Total Processing Time
3. Memory RSS (GB) & Index Posting counts
4. Candidate Distribution Percentiles:
   - Median (P50), P90, P95, P99, Max Candidates per entity
   - Count of $S_1$ entities with 0 candidates (Singletons)
   - Count of $S_1$ entities with 1–5 candidates
   - Count of $S_1$ entities with 6–10 candidates
   - Count of $S_1$ entities with 11–25 candidates
   - Count of $S_1$ entities with >25 candidates

---

## 7. Deliverable Verification

Before handing over `output/candidate_pairs.tsv` to Mukul:
1. Run `python3 utils/validate_submission.py --candidate output/candidate_pairs.tsv --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test`.
2. Confirm exactly **1,732,544 rows** (plus header).
3. Confirm zero duplicate IDs per candidate string.
4. Confirm no cross-country or $S_1$ ID leaks.

---
