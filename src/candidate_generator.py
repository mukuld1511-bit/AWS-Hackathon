"""
Production Candidate Generation Pipeline (Stage 5).

Generates top-K candidate target entities for Source 1 records using the frozen
Stage 4.8.5 blocking and ranking configuration:
- Blocking Keys: A + B + C + D(250) + F(500) + G(500) + I(500)
- Country Isolation: Hard Same-Country Partitioning (0 cross-country pairs)
- Ranking Config (R3.3): A=3.0, B=2.0, C=1.5, D=1.0, F=1.5, G=2.0, I=2.0, Bonus=0.5
- Capping: Top K=25 candidates
- Tie-Breaking: Deterministic (score DESC, target_entity_id ASC)
- Format: TSV with header (source1_entity_id \\t candidate_entity_ids)
"""

import os
import sys
import csv
import time
import psutil
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# Ensure root workspace directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.targeted_keys import Stage47TargetedIndex
from src.ranking import RankingConfig, rank_candidates


@dataclass
class CandidateGenerationReport:
    """Detailed metadata and metrics report for candidate generation runs."""
    total_s1_records: int = 0
    total_candidate_pairs: int = 0
    singletons_count: int = 0
    non_singletons_count: int = 0
    zero_candidate_s1: int = 0
    fewer_than_25_cands: int = 0
    exactly_25_cands: int = 0
    
    # Candidate Volume Percentiles
    mean_candidates: float = 0.0
    p50_candidates: int = 0
    p95_candidates: int = 0
    p99_candidates: int = 0
    max_candidates: int = 0

    # Source breakdown
    s2_candidates_count: int = 0
    s3_candidates_count: int = 0

    # Invariant checks
    duplicate_pairs_count: int = 0
    invalid_target_ids_count: int = 0
    cross_country_violations: int = 0
    dropped_s1_records: int = 0
    is_deterministic: bool = True

    # Performance
    indexing_time_sec: float = 0.0
    query_eval_time_sec: float = 0.0
    total_runtime_sec: float = 0.0
    peak_ram_gb: float = 0.0
    output_file_size_mb: float = 0.0

    # Validation (Training Preflight only)
    total_true_matches: int = 0
    uncapped_recovered_matches: int = 0
    uncapped_blocking_recall: float = 0.0
    recov_k25_matches: int = 0
    recall_at_25: float = 0.0


def get_frozen_r3_3_ranking_config() -> RankingConfig:
    """Return the frozen Stage 4.8.5 validated R3.3 ranking configuration."""
    return RankingConfig(
        name_weight=3.0,
        prefix_weight=2.0,
        address_weight=1.5,
        key_d_weight=1.0,
        key_f_weight=1.5,
        key_g_weight=2.0,
        key_i_weight=2.0,
        multi_key_bonus=0.5,
        max_candidates=25
    )


def generate_candidate_pairs(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_path: str,
    ranking_cfg: Optional[RankingConfig] = None,
    gt_map: Optional[Dict[str, Set[str]]] = None
) -> CandidateGenerationReport:
    """Generate production candidate_pairs.tsv for S1 records against target S2/S3 index.

    Args:
        s1_path: Path to S1 TSV file.
        s2_path: Path to S2 TSV file.
        s3_path: Path to S3 TSV file.
        output_path: Destination TSV path (e.g. output/candidate_pairs.tsv).
        ranking_cfg: RankingConfig instance (defaults to frozen R3.3).
        gt_map: Optional ground truth mapping for training preflight recall verification.

    Returns:
        CandidateGenerationReport containing full execution metrics and invariant checks.
    """
    t0_start = time.time()
    process = psutil.Process(os.getpid())

    if ranking_cfg is None:
        ranking_cfg = get_frozen_r3_3_ranking_config()

    report = CandidateGenerationReport()
    valid_target_ids: Set[str] = set()

    # 1. Build Inverted Index across S2 and S3
    print(f"[*] Step 1/3: Building Country-Partitioned Inverted Index from S2 & S3...", flush=True)
    t0_idx = time.time()
    index = Stage47TargetedIndex(
        prefix_len=4,
        enable_exp_d=True,
        enable_exp_f=True,
        enable_exp_g=True,
        enable_exp_h=False,
        enable_exp_i=True,
        d_max_posting_threshold=250,
        f_max_posting_threshold=500,
        g_max_posting_threshold=500,
        i_max_posting_threshold=500
    )

    total_target_records = 0
    for target_path in [s2_path, s3_path]:
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target dataset file not found: {target_path}")
        
        with open(target_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) >= 4:
                    entity_id = row[0].strip()
                    name = row[1]
                    addr = row[2]
                    country = row[3].strip()
                    if entity_id and country:
                        index.add_record(entity_id, name, addr, country)
                        valid_target_ids.add(entity_id)
                        total_target_records += 1

    report.indexing_time_sec = round(time.time() - t0_idx, 2)
    ram_idx = round(process.memory_info().rss / (1024 * 1024 * 1024), 2)
    print(f"    Indexed {total_target_records:,} target records in {report.indexing_time_sec}s | Peak RAM: {ram_idx} GB", flush=True)

    # Ensure output parent directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    # 2. Stream S1 Records, Generate & Rank Candidates, Write TSV
    print(f"[*] Step 2/3: Processing S1 Records & Writing Candidate Pairs to '{output_path}'...", flush=True)
    t0_query = time.time()

    retained_candidate_counts = []
    uncapped_found_count = 0
    gt_total_true = sum(len(v) for v in gt_map.values()) if gt_map else 0

    with open(s1_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8", newline="") as fout:
        
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t")
        
        # Header: source1_entity_id \t candidate_entity_ids
        next(reader, None)
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for row in reader:
            if len(row) < 4:
                report.dropped_s1_records += 1
                continue
            
            s1_id = row[0].strip()
            s1_name = row[1]
            s1_addr = row[2]
            s1_ctry = row[3].strip()
            report.total_s1_records += 1

            true_targets = gt_map.get(s1_id, set()) if gt_map else set()
            if gt_map:
                if len(true_targets) > 0:
                    report.non_singletons_count += 1
                else:
                    report.singletons_count += 1

            # Candidate Query & Country Partitioning
            ev_map = index.query_record(s1_name, s1_addr, s1_ctry)
            uncapped_cands = set(ev_map.keys())

            if gt_map:
                uncapped_found = true_targets & uncapped_cands
                uncapped_found_count += len(uncapped_found)

            # Deterministic R3.3 Scoring & Ranking
            ranked = rank_candidates(ev_map, ranking_cfg)
            
            # Capping at top 25 candidates
            top_k_candidates = ranked[:25]
            retained_ids = [c.entity_id for c in top_k_candidates]
            num_retained = len(retained_ids)
            retained_candidate_counts.append(num_retained)
            report.total_candidate_pairs += num_retained

            # Metrics & Invariant Checks
            if num_retained == 0:
                report.zero_candidate_s1 += 1
            elif num_retained < 25:
                report.fewer_than_25_cands += 1
            else:
                report.exactly_25_cands += 1

            # Candidate ID Validity & Source Breakdown
            seen_for_s1 = set()
            for cand_id in retained_ids:
                if cand_id in seen_for_s1:
                    report.duplicate_pairs_count += 1
                seen_for_s1.add(cand_id)

                if cand_id not in valid_target_ids:
                    report.invalid_target_ids_count += 1

                if cand_id.startswith("S2-"):
                    report.s2_candidates_count += 1
                elif cand_id.startswith("S3-"):
                    report.s3_candidates_count += 1

            if gt_map:
                found_25 = set(retained_ids) & true_targets
                report.recov_k25_matches += len(found_25)

            # Output Formatting: comma-separated candidate string
            cand_str = ",".join(retained_ids)
            writer.writerow([s1_id, cand_str])

            if report.total_s1_records % 500000 == 0:
                cur_ram = round(process.memory_info().rss / (1024 * 1024 * 1024), 2)
                print(f"    Processed {report.total_s1_records:,} S1 records | RAM: {cur_ram} GB", flush=True)

    report.query_eval_time_sec = round(time.time() - t0_query, 2)

    # 3. Compute Distribution Statistics
    cand_arr = np.array(retained_candidate_counts, dtype=np.int32)
    report.mean_candidates = round(float(np.mean(cand_arr)), 2)
    report.p50_candidates = int(np.percentile(cand_arr, 50))
    report.p95_candidates = int(np.percentile(cand_arr, 95))
    report.p99_candidates = int(np.percentile(cand_arr, 99))
    report.max_candidates = int(np.max(cand_arr))

    report.total_runtime_sec = round(time.time() - t0_start, 2)
    report.peak_ram_gb = round(process.memory_info().rss / (1024 * 1024 * 1024), 2)
    report.output_file_size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)

    if gt_map and gt_total_true > 0:
        report.total_true_matches = gt_total_true
        report.uncapped_recovered_matches = uncapped_found_count
        report.uncapped_blocking_recall = round(uncapped_found_count / gt_total_true * 100, 4)
        report.recall_at_25 = round(report.recov_k25_matches / gt_total_true * 100, 4)

    return report
