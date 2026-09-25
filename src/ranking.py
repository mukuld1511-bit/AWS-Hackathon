"""
Candidate Evidence Ranking & Deterministic Capping Module.

Converts structured BlockingEvidence into deterministic candidate rankings
with configurable evidence weights, stable multi-key tie-breaking, and top-K capping.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from src.blocking_index import BlockingEvidence


@dataclass
class RankingConfig:
    """Configuration structure for evidence scoring weights and top-K capping."""
    name_weight: float = 3.0       # Weight for Key A (Exact Normalized Name)
    prefix_weight: float = 2.0     # Weight for Key B (Name Prefix + Address Numbers)
    address_weight: float = 1.5    # Weight for Key C (Physical Address Anchor)
    key_d_weight: float = 0.0      # Weight for Key D (First Significant Name Token)
    key_f_weight: float = 0.0      # Weight for Key F (Street Token Pair)
    key_g_weight: float = 0.0      # Weight for Key G (Name Token Pair Anchor)
    key_i_weight: float = 0.0      # Weight for Key I (Address Token Pair)
    multi_key_bonus: float = 0.0   # Multi-key match bonus per extra key (>1)
    max_candidates: int = 25       # Baseline top-K candidate cap


@dataclass
class RankedCandidate:
    """Transparent representation of a scored and ranked candidate entity."""
    entity_id: str
    score: float
    evidence_count: int
    matched_name_key: bool
    matched_prefix_key: bool
    matched_address_key: bool
    matched_key_d: bool = False
    matched_key_f: bool = False
    matched_key_g: bool = False
    matched_key_i: bool = False


def score_candidate(
    candidate_id: str,
    evidence: BlockingEvidence,
    config: RankingConfig
) -> RankedCandidate:
    """Calculate the total blocking evidence score for a candidate.

    Args:
        candidate_id: Raw candidate entity ID string (e.g., S2-100).
        evidence: BlockingEvidence instance for this candidate.
        config: RankingConfig instance specifying weights.

    Returns:
        RankedCandidate object containing detailed score breakdown.
    """
    score = (
        (int(evidence.matched_name_key) * config.name_weight) +
        (int(evidence.matched_prefix_key) * config.prefix_weight) +
        (int(evidence.matched_address_key) * config.address_weight) +
        (int(evidence.matched_key_d) * config.key_d_weight) +
        (int(evidence.matched_key_f) * config.key_f_weight) +
        (int(evidence.matched_key_g) * config.key_g_weight) +
        (int(evidence.matched_key_i) * config.key_i_weight)
    )

    if config.multi_key_bonus > 0.0 and evidence.evidence_count > 1:
        score += (evidence.evidence_count - 1) * config.multi_key_bonus

    return RankedCandidate(
        entity_id=candidate_id,
        score=score,
        evidence_count=evidence.evidence_count,
        matched_name_key=evidence.matched_name_key,
        matched_prefix_key=evidence.matched_prefix_key,
        matched_address_key=evidence.matched_address_key,
        matched_key_d=evidence.matched_key_d,
        matched_key_f=evidence.matched_key_f,
        matched_key_g=evidence.matched_key_g,
        matched_key_i=evidence.matched_key_i,
    )


def rank_candidates(
    evidence_map: Dict[str, BlockingEvidence],
    config: Optional[RankingConfig] = None
) -> List[RankedCandidate]:
    """Rank candidates deterministically based on blocking evidence score.

    Tie-breaking Order:
    1. Descending evidence score (-score)
    2. Descending total matched keys (-evidence_count)
    3. Ascending candidate_id (lexicographical string order for exact stability)

    Args:
        evidence_map: Map of candidate_id -> BlockingEvidence from BlockingIndex.
        config: Optional RankingConfig (uses default config if None).

    Returns:
        List of RankedCandidate objects capped at config.max_candidates.
    """
    cfg = config or RankingConfig()

    if not evidence_map:
        return []

    # Score all candidates
    scored_list: List[RankedCandidate] = [
        score_candidate(cand_id, ev, cfg)
        for cand_id, ev in evidence_map.items()
    ]

    # Deterministic sorting using multi-level key tuple
    scored_list.sort(key=lambda c: (-c.score, -c.evidence_count, c.entity_id))

    # Apply top-K capping
    if cfg.max_candidates > 0:
        return scored_list[:cfg.max_candidates]
    return scored_list


def get_candidate_ids(ranked_list: List[RankedCandidate]) -> List[str]:
    """Helper utility extracting ordered entity IDs from RankedCandidate list."""
    return [c.entity_id for c in ranked_list]
