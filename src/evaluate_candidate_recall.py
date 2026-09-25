"""
Ground-Truth Candidate Recall Evaluation Module.

Evaluates Candidate Recall@K, Full-Recall Rate@K, Blocking Recall (before capping),
and Blocking Key Evidence Breakdown against training ground-truth labels.
"""

import csv
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from src.blocking_index import BlockingIndex, BlockingEvidence
from src.ranking import RankingConfig, rank_candidates, get_candidate_ids


@dataclass
class EvidenceBreakdown:
    """Tracks which blocking key combinations recovered true matches."""
    key_a_only: int = 0
    key_b_only: int = 0
    key_c_only: int = 0
    key_ab: int = 0
    key_ac: int = 0
    key_bc: int = 0
    key_abc: int = 0
    not_found: int = 0

    @property
    def total_found(self) -> int:
        """Total true matches recovered across any key."""
        return (
            self.key_a_only + self.key_b_only + self.key_c_only +
            self.key_ab + self.key_ac + self.key_bc + self.key_abc
        )


@dataclass
class RecallMetricsAtK:
    """Metrics evaluated at a specific candidate cap K."""
    k: int
    recovered_true_matches: int = 0
    micro_recall: float = 0.0
    full_recall_count: int = 0
    full_recall_rate: float = 0.0
    total_candidates_generated: int = 0
    avg_candidates_per_s1: float = 0.0
    max_candidates_per_s1: int = 0
    zero_candidate_s1_count: int = 0
    non_singleton_zero_candidate_count: int = 0


@dataclass
class EvaluationSummary:
    """Summary metrics of a candidate recall evaluation run."""
    total_s1_eval: int = 0
    total_singleton_s1: int = 0
    total_non_singleton_s1: int = 0
    total_true_matches: int = 0
    blocking_recovered_matches: int = 0
    blocking_recall: float = 0.0
    evidence_breakdown: EvidenceBreakdown = field(default_factory=EvidenceBreakdown)
    metrics_by_k: Dict[int, RecallMetricsAtK] = field(default_factory=dict)
    total_indexing_time_sec: float = 0.0
    total_eval_time_sec: float = 0.0


def load_ground_truth(ground_truth_path: str) -> Dict[str, Set[str]]:
    """Stream and parse ground truth TSV into a dictionary mapping S1 ID -> Set[matched IDs].

    Args:
        ground_truth_path: Path to train_ground_truth.tsv.

    Returns:
        Dict mapping source1_entity_id -> set of true matching S2/S3 entity IDs (empty set for singletons).
    """
    gt_map: Dict[str, Set[str]] = {}
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # Header: source1_entity_id, matched_entity_ids
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            if not s1_id:
                continue
            matched_str = row[1].strip() if len(row) > 1 else ""
            if matched_str:
                matched_ids = {m.strip() for m in matched_str.split(",") if m.strip()}
                gt_map[s1_id] = matched_ids
            else:
                gt_map[s1_id] = set()
    return gt_map


def categorize_true_match_evidence(
    evidence: BlockingEvidence,
    breakdown: EvidenceBreakdown
) -> None:
    """Categorize evidence combination for a recovered true match into Breakdown counts."""
    a, b, c = evidence.matched_name_key, evidence.matched_prefix_key, evidence.matched_address_key
    if a and b and c:
        breakdown.key_abc += 1
    elif a and b:
        breakdown.key_ab += 1
    elif a and c:
        breakdown.key_ac += 1
    elif b and c:
        breakdown.key_bc += 1
    elif a:
        breakdown.key_a_only += 1
    elif b:
        breakdown.key_b_only += 1
    elif c:
        breakdown.key_c_only += 1


def evaluate_ground_truth_recall(
    index: BlockingIndex,
    s1_source_path: str,
    ground_truth_map: Dict[str, Set[str]],
    k_values: Tuple[int, ...] = (10, 15, 20, 25, 30),
    ranking_config: Optional[RankingConfig] = None,
    max_s1_records: Optional[int] = None
) -> EvaluationSummary:
    """Stream S1 entities, query index, rank candidates, and compute Recall metrics across K values.

    Args:
        index: Populated BlockingIndex built from target S2 and S3 sources.
        s1_source_path: Path to train_source1.tsv.
        ground_truth_map: Dictionary mapping s1_id -> set of true matched IDs.
        k_values: Tuple of K caps to evaluate (default: 10, 15, 20, 25, 30).
        ranking_config: RankingConfig instance.
        max_s1_records: Optional cap on S1 records to evaluate (useful for small samples/testing).

    Returns:
        EvaluationSummary containing full empirical metrics and diagnostics.
    """
    eval_start_time = time.time()
    cfg = ranking_config or RankingConfig()

    summary = EvaluationSummary()
    breakdown = summary.evidence_breakdown

    # Initialize metrics for each K
    metrics_by_k: Dict[int, RecallMetricsAtK] = {k: RecallMetricsAtK(k=k) for k in k_values}
    summary.metrics_by_k = metrics_by_k

    with open(s1_source_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # Header: entity_id, business_name, business_address, country

        for row in reader:
            if max_s1_records and summary.total_s1_eval >= max_s1_records:
                break
            if len(row) < 4:
                continue

            s1_id = row[0].strip()
            name = row[1]
            addr = row[2]
            country = row[3].strip()

            summary.total_s1_eval += 1
            true_matches = ground_truth_map.get(s1_id, set())
            is_singleton = (len(true_matches) == 0)

            if is_singleton:
                summary.total_singleton_s1 += 1
            else:
                summary.total_non_singleton_s1 += 1
                summary.total_true_matches += len(true_matches)

            # Query Blocking Index (Uncapped Blocking Candidates)
            evidence_map = index.query_record(name, addr, country)
            uncapped_cand_ids = set(evidence_map.keys())

            # 1. Uncapped Blocking Recall & Evidence Diagnostics
            for true_id in true_matches:
                if true_id in uncapped_cand_ids:
                    summary.blocking_recovered_matches += 1
                    categorize_true_match_evidence(evidence_map[true_id], breakdown)
                else:
                    breakdown.not_found += 1

            # 2. Rank Candidates once for max(k_values)
            max_k = max(k_values)
            cfg.max_candidates = max_k
            ranked_max = rank_candidates(evidence_map, cfg)
            ranked_max_ids = get_candidate_ids(ranked_max)

            # 3. Evaluate candidate list across each K cap
            for k in k_values:
                m_k = metrics_by_k[k]
                cands_k = ranked_max_ids[:k]
                cands_k_set = set(cands_k)

                num_cands = len(cands_k)
                m_k.total_candidates_generated += num_cands
                m_k.max_candidates_per_s1 = max(m_k.max_candidates_per_s1, num_cands)

                if num_cands == 0:
                    m_k.zero_candidate_s1_count += 1
                    if not is_singleton:
                        m_k.non_singleton_zero_candidate_count += 1

                if not is_singleton:
                    recovered = len(true_matches & cands_k_set)
                    m_k.recovered_true_matches += recovered
                    if true_matches.issubset(cands_k_set):
                        m_k.full_recall_count += 1

    summary.total_eval_time_sec = time.time() - eval_start_time

    # Finalize global metrics
    if summary.total_true_matches > 0:
        summary.blocking_recall = summary.blocking_recovered_matches / summary.total_true_matches

    for k, m_k in metrics_by_k.items():
        if summary.total_true_matches > 0:
            m_k.micro_recall = m_k.recovered_true_matches / summary.total_true_matches
        if summary.total_non_singleton_s1 > 0:
            m_k.full_recall_rate = m_k.full_recall_count / summary.total_non_singleton_s1
        if summary.total_s1_eval > 0:
            m_k.avg_candidates_per_s1 = m_k.total_candidates_generated / summary.total_s1_eval

    return summary


def generate_recall_report_table(summary: EvaluationSummary) -> str:
    """Format evaluation summary into a clean markdown report table."""
    lines = []
    lines.append(f"### Ground-Truth Candidate Recall Report")
    lines.append(f"- **Total S1 Evaluated:** {summary.total_s1_eval:,}")
    lines.append(f"- **Singleton S1 Count (0 True Matches):** {summary.total_singleton_s1:,}")
    lines.append(f"- **Non-Singleton S1 Count:** {summary.total_non_singleton_s1:,}")
    lines.append(f"- **Total True S2/S3 Matches:** {summary.total_true_matches:,}")
    lines.append(f"- **Blocking Recall (Uncapped):** {summary.blocking_recall * 100:.2f}% ({summary.blocking_recovered_matches:,} / {summary.total_true_matches:,})\n")

    lines.append("| K | Micro Recall@K | Full Recall Rate@K | Avg Cands/S1 | Max Cands | Zero-Cand S1 |")
    lines.append("| --: | -------------: | ----------------: | -----------: | --------: | -----------: |")

    for k in sorted(summary.metrics_by_k.keys()):
        m = summary.metrics_by_k[k]
        lines.append(
            f"| {k} | {m.micro_recall * 100:13.2f}% | {m.full_recall_rate * 100:16.2f}% | {m.avg_candidates_per_s1:11.2f} | {m.max_candidates_per_s1:8d} | {m.zero_candidate_s1_count:11d} |"
        )

    lines.append("\n#### True Match Evidence Breakdown (Uncapped Blocking):")
    eb = summary.evidence_breakdown
    total_true = summary.total_true_matches
    def pct(cnt):
        return f"{cnt:,} ({cnt/total_true*100:.2f}%)" if total_true > 0 else f"{cnt:,}"

    lines.append(f"- **Key A Only (Exact Name):** {pct(eb.key_a_only)}")
    lines.append(f"- **Key B Only (Prefix + Addr Nums):** {pct(eb.key_b_only)}")
    lines.append(f"- **Key C Only (Address Anchor):** {pct(eb.key_c_only)}")
    lines.append(f"- **Key A + B:** {pct(eb.key_ab)}")
    lines.append(f"- **Key A + C:** {pct(eb.key_ac)}")
    lines.append(f"- **Key B + C:** {pct(eb.key_bc)}")
    lines.append(f"- **Key A + B + C (All 3 Keys):** {pct(eb.key_abc)}")
    lines.append(f"- **Not Found by Blocking (Recall Failure):** {pct(eb.not_found)}")

    return "\n".join(lines)
