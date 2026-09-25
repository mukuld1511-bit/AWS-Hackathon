"""
Stage 4.7 Targeted Blocking Keys & Extended Index (Keys G, H, I).

Contains targeted experimental keys designed from failure analysis:
- Key G: Name Token Pair Anchor (country, sorted_token_1, sorted_token_2)
- Key H: Second/Rare Significant Name Token Anchor (country, second_sig_token)
- Key I: Address Token Pair Anchor (country, sorted_addr_token_1, sorted_addr_token_2)
"""

from typing import Dict, List, Optional, Set, Tuple
from src.normalization import normalize_name, clean_punctuation
from src.blocking_index import BlockingEvidence
from src.experimental_blocking import (
    extract_significant_tokens,
    COMMON_BUSINESS_STOP_WORDS,
    Stage46ExperimentalIndex
)

ADDRESS_STOP_WORDS: Set[str] = {
    "road", "street", "drive", "avenue", "lane", "court", "suite", "floor",
    "unit", "north", "south", "east", "west", "building", "block", "phase",
    "near", "opp", "opposite", "behind", "beside", "facing", "main", "cross",
    "plot", "nagar", "colony", "area", "dist", "district", "city", "state"
}


def generate_key_g_name_pair(country: str, business_name: str) -> Optional[Tuple[str, str, str]]:
    """Key G: Name Token Pair (country, sorted_token_1, sorted_token_2)."""
    if not country:
        return None
    tokens = extract_significant_tokens(business_name)
    if len(tokens) >= 2:
        sorted_pair = sorted([tokens[0], tokens[1]])
        return (country.strip(), sorted_pair[0], sorted_pair[1])
    return None


def generate_key_h_second_name_token(country: str, business_name: str) -> Optional[Tuple[str, str]]:
    """Key H: Second Significant Name Token (country, second_sig_token)."""
    if not country:
        return None
    tokens = extract_significant_tokens(business_name)
    if len(tokens) >= 2:
        return (country.strip(), tokens[1])
    return None


def generate_key_i_address_pair(country: str, business_address: str) -> Optional[Tuple[str, str, str]]:
    """Key I: Address Token Pair (country, sorted_addr_token_1, sorted_addr_token_2)."""
    if not country or not business_address:
        return None
    clean_addr = clean_punctuation(business_address.lower())
    tokens = [
        t for t in clean_addr.split()
        if len(t) >= 4 and t.isalpha() and t not in ADDRESS_STOP_WORDS
    ]
    if len(tokens) >= 2:
        sorted_tokens = sorted(tokens, key=lambda t: (-len(t), t))[:2]
        sorted_pair = sorted(sorted_tokens)
        return (country.strip(), sorted_pair[0], sorted_pair[1])
    return None


class Stage47TargetedIndex(Stage46ExperimentalIndex):
    """Stage 4.7 Extended BlockingIndex supporting candidate keys G, H, I with posting caps."""

    def __init__(
        self,
        prefix_len: int = 4,
        enable_exp_d: bool = True,
        enable_exp_f: bool = True,
        enable_exp_g: bool = False,
        enable_exp_h: bool = False,
        enable_exp_i: bool = False,
        d_max_posting_threshold: int = 250,
        f_max_posting_threshold: int = 500,
        g_max_posting_threshold: int = 500,
        h_max_posting_threshold: int = 250,
        i_max_posting_threshold: int = 500,
    ):
        super().__init__(
            prefix_len=prefix_len,
            enable_exp_d=enable_exp_d,
            enable_exp_e=False,
            enable_exp_f=enable_exp_f,
            d_token_strategy="first",
            d_max_posting_threshold=d_max_posting_threshold,
            f_max_posting_threshold=f_max_posting_threshold
        )
        self.enable_exp_g = enable_exp_g
        self.enable_exp_h = enable_exp_h
        self.enable_exp_i = enable_exp_i

        self.g_max_posting_threshold = g_max_posting_threshold
        self.h_max_posting_threshold = h_max_posting_threshold
        self.i_max_posting_threshold = i_max_posting_threshold

        self.exp_g_index: Dict[Tuple[str, str, str], List[str]] = {}
        self.exp_h_index: Dict[Tuple[str, str], List[str]] = {}
        self.exp_i_index: Dict[Tuple[str, str, str], List[str]] = {}

    def add_record(
        self,
        entity_id: str,
        business_name: str,
        business_address: str,
        country: str
    ) -> None:
        super().add_record(entity_id, business_name, business_address, country)

        if not entity_id or not country:
            return

        c_norm = country.strip()

        # Key G
        if self.enable_exp_g:
            key_g = generate_key_g_name_pair(c_norm, business_name)
            if key_g:
                if key_g not in self.exp_g_index:
                    self.exp_g_index[key_g] = []
                posting = self.exp_g_index[key_g]
                if len(posting) < self.g_max_posting_threshold:
                    if not posting or posting[-1] != entity_id:
                        posting.append(entity_id)

        # Key H
        if self.enable_exp_h:
            key_h = generate_key_h_second_name_token(c_norm, business_name)
            if key_h:
                if key_h not in self.exp_h_index:
                    self.exp_h_index[key_h] = []
                posting = self.exp_h_index[key_h]
                if len(posting) < self.h_max_posting_threshold:
                    if not posting or posting[-1] != entity_id:
                        posting.append(entity_id)

        # Key I
        if self.enable_exp_i:
            key_i = generate_key_i_address_pair(c_norm, business_address)
            if key_i:
                if key_i not in self.exp_i_index:
                    self.exp_i_index[key_i] = []
                posting = self.exp_i_index[key_i]
                if len(posting) < self.i_max_posting_threshold:
                    if not posting or posting[-1] != entity_id:
                        posting.append(entity_id)

    def query_record(
        self,
        business_name: str,
        business_address: str,
        country: str
    ) -> Dict[str, BlockingEvidence]:
        evidence_map = super().query_record(business_name, business_address, country)

        if not country:
            return evidence_map

        c_norm = country.strip()

        # Key G
        if self.enable_exp_g:
            key_g = generate_key_g_name_pair(c_norm, business_name)
            if key_g and key_g in self.exp_g_index:
                for cand_id in self.exp_g_index[key_g]:
                    if cand_id not in evidence_map:
                        evidence_map[cand_id] = BlockingEvidence()
                    evidence_map[cand_id].matched_key_g = True

        # Key H
        if self.enable_exp_h:
            key_h = generate_key_h_second_name_token(c_norm, business_name)
            if key_h and key_h in self.exp_h_index:
                for cand_id in self.exp_h_index[key_h]:
                    if cand_id not in evidence_map:
                        evidence_map[cand_id] = BlockingEvidence()

        # Key I
        if self.enable_exp_i:
            key_i = generate_key_i_address_pair(c_norm, business_address)
            if key_i and key_i in self.exp_i_index:
                for cand_id in self.exp_i_index[key_i]:
                    if cand_id not in evidence_map:
                        evidence_map[cand_id] = BlockingEvidence()
                    evidence_map[cand_id].matched_key_i = True

        return evidence_map
