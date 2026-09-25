"""
Inverted Index Data Structures for Candidate Blocking Engine.

Implements memory-efficient inverted indexes for Key A, Key B, and Key C with
country isolation, exact entity ID preservation, and evidence tracking.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from src.blocking_keys import (
    KEY_NAME,
    KEY_PREFIX,
    KEY_ADDRESS,
    generate_key_a,
    generate_key_b,
    generate_key_c,
)


@dataclass
class BlockingEvidence:
    """Represents candidate match evidence across all blocking keys."""
    matched_name_key: bool = False   # Key A (Exact Normalized Name)
    matched_prefix_key: bool = False # Key B (Name Prefix + Address Numbers)
    matched_address_key: bool = False# Key C (Physical Address Anchor)
    matched_key_d: bool = False      # Key D (First Significant Name Token)
    matched_key_f: bool = False      # Key F (Street Token Pair)
    matched_key_g: bool = False      # Key G (Name Token Pair Anchor)
    matched_key_i: bool = False      # Key I (Address Token Pair)

    @property
    def evidence_count(self) -> int:
        """Total number of blocking keys matched by this candidate."""
        return (
            int(self.matched_name_key) +
            int(self.matched_prefix_key) +
            int(self.matched_address_key) +
            int(self.matched_key_d) +
            int(self.matched_key_f) +
            int(self.matched_key_g) +
            int(self.matched_key_i)
        )


class BlockingIndex:
    """Inverted index container for candidate retrieval across S2 and S3 pool."""

    def __init__(self, prefix_len: int = 4):
        self.prefix_len = prefix_len
        
        # Inverted Index Storage: Key -> List of Entity IDs
        self.name_index: Dict[Tuple[str, str], List[str]] = {}
        self.prefix_index: Dict[Tuple[str, str, str], List[str]] = {}
        self.address_index: Dict[Tuple[str, str], List[str]] = {}

    def add_record(
        self,
        entity_id: str,
        business_name: str,
        business_address: str,
        country: str
    ) -> None:
        """Insert blocking keys derived from an S2/S3 candidate record into inverted indexes."""
        if not entity_id or not country:
            return

        c_norm = country.strip()

        # Key A
        ka = generate_key_a(c_norm, business_name)
        if ka:
            if ka not in self.name_index:
                self.name_index[ka] = []
            posting = self.name_index[ka]
            if not posting or posting[-1] != entity_id:
                posting.append(entity_id)

        # Key B
        kb = generate_key_b(c_norm, business_name, business_address, prefix_len=self.prefix_len)
        if kb:
            if kb not in self.prefix_index:
                self.prefix_index[kb] = []
            posting = self.prefix_index[kb]
            if not posting or posting[-1] != entity_id:
                posting.append(entity_id)

        # Key C
        kc = generate_key_c(c_norm, business_address)
        if kc:
            if kc not in self.address_index:
                self.address_index[kc] = []
            posting = self.address_index[kc]
            if not posting or posting[-1] != entity_id:
                posting.append(entity_id)

    def query_record(
        self,
        business_name: str,
        business_address: str,
        country: str
    ) -> Dict[str, BlockingEvidence]:
        """Query inverted indexes for an S1 record and aggregate candidate match evidence."""
        candidates: Dict[str, BlockingEvidence] = {}

        if not country:
            return candidates

        c_norm = country.strip()

        # Query Key A
        ka = generate_key_a(c_norm, business_name)
        if ka and ka in self.name_index:
            for cand_id in self.name_index[ka]:
                if cand_id not in candidates:
                    candidates[cand_id] = BlockingEvidence()
                candidates[cand_id].matched_name_key = True

        # Query Key B
        kb = generate_key_b(c_norm, business_name, business_address, prefix_len=self.prefix_len)
        if kb and kb in self.prefix_index:
            for cand_id in self.prefix_index[kb]:
                if cand_id not in candidates:
                    candidates[cand_id] = BlockingEvidence()
                candidates[cand_id].matched_prefix_key = True

        # Query Key C
        kc = generate_key_c(c_norm, business_address)
        if kc and kc in self.address_index:
            for cand_id in self.address_index[kc]:
                if cand_id not in candidates:
                    candidates[cand_id] = BlockingEvidence()
                candidates[cand_id].matched_address_key = True

        return candidates

    def get_statistics(self) -> Dict[str, int]:
        """Return current entity index statistics."""
        total_postings = (
            sum(len(p) for p in self.name_index.values()) +
            sum(len(p) for p in self.prefix_index.values()) +
            sum(len(p) for p in self.address_index.values())
        )
        all_records = set()
        for p in self.name_index.values():
            all_records.update(p)
        for p in self.prefix_index.values():
            all_records.update(p)
        for p in self.address_index.values():
            all_records.update(p)

        return {
            "total_records": len(all_records),
            "unique_name_keys": len(self.name_index),
            "unique_prefix_keys": len(self.prefix_index),
            "unique_address_keys": len(self.address_index),
            "total_postings": total_postings
        }
