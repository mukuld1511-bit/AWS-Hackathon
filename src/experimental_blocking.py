"""
Experimental Blocking Keys & Index Extensions (Stage 4.6).

Contains isolated implementations of experimental candidate generation strategies:
- Experiment D: Significant Name Token Anchor (first vs longest, with frequency caps 100/250/500)
- Experiment E: Extended Prefix (5-char) + Address Digits
- Experiment F: Address Street Pair Anchor (for digitless addresses)

Includes posting-list analysis utilities and marginal recall evaluation hooks.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from src.normalization import normalize_name, extract_address_numbers, clean_punctuation
from src.blocking_keys import generate_key_a, generate_key_b, generate_key_c
from src.blocking_index import BlockingIndex, BlockingEvidence

# Common generic business stop words to reject from token indexing (to prevent candidate explosion)
COMMON_BUSINESS_STOP_WORDS: Set[str] = {
    "store", "shop", "center", "centre", "services", "service", "group", "traders",
    "enterprise", "enterprises", "solutions", "tech", "technologies", "technology",
    "mart", "retail", "agency", "global", "international", "national", "pvt", "ltd",
    "inc", "llc", "llp", "corp", "corporation", "company", "co", "associates", "consulting",
    "consultants", "india", "american", "first", "new", "city", "state", "the", "bank",
    "market", "marketing", "trading", "finance", "holdings", "capital", "management"
}


def extract_significant_tokens(name: str) -> List[str]:
    """Extract valid significant tokens of normalized name (len >= 4, non-stopword, alphabetic)."""
    norm_n = normalize_name(name)
    if not norm_n:
        return []
    
    tokens = norm_n.split()
    valid = [t for t in tokens if len(t) >= 4 and t not in COMMON_BUSINESS_STOP_WORDS and t.isalpha()]
    return valid


def extract_first_significant_token(name: str) -> Optional[str]:
    """Extract the first significant token."""
    tokens = extract_significant_tokens(name)
    return tokens[0] if tokens else None


def extract_longest_significant_token(name: str) -> Optional[str]:
    """Extract the longest significant token."""
    tokens = extract_significant_tokens(name)
    if not tokens:
        return None
    # Sort by length descending, then alphabetical for deterministic tie-breaker
    tokens.sort(key=lambda t: (-len(t), t))
    return tokens[0]


def generate_key_d_token_anchor(
    country: str,
    business_name: str,
    token_strategy: str = "first"
) -> Optional[Tuple[str, str]]:
    """Experiment D: (country, significant_name_token)."""
    if not country:
        return None
    
    if token_strategy == "longest":
        token = extract_longest_significant_token(business_name)
    else:
        token = extract_first_significant_token(business_name)
        
    if not token:
        return None
    return (country.strip(), token)


def generate_key_e_prefix5(
    country: str,
    business_name: str,
    business_address: str
) -> Optional[Tuple[str, str, str]]:
    """Experiment E: (country, prefix_5chars, address_numbers)."""
    return generate_key_b(country, business_name, business_address, prefix_len=5)


def extract_street_tokens(address: str) -> Optional[Tuple[str, str]]:
    """Extract first 2 alphabetic street/location tokens (>3 chars) from address."""
    if not address:
        return None
    clean_addr = clean_punctuation(address.lower())
    tokens = [
        t for t in clean_addr.split()
        if len(t) >= 4 and t.isalpha() and t not in {
            "road", "street", "drive", "avenue", "lane", "court", "suite", "floor",
            "unit", "north", "south", "east", "west", "building", "block", "phase"
        }
    ]
    if len(tokens) >= 2:
        return (tokens[0], tokens[1])
    return None


def generate_key_f_street_anchor(country: str, business_address: str) -> Optional[Tuple[str, str, str]]:
    """Experiment F: (country, street_token_1, street_token_2)."""
    if not country:
        return None
    street_pair = extract_street_tokens(business_address)
    if not street_pair:
        return None
    return (country.strip(), street_pair[0], street_pair[1])


class Stage46ExperimentalIndex(BlockingIndex):
    """Stage 4.6 Extension of BlockingIndex supporting combinations of D, E, F with configurable caps."""

    def __init__(
        self,
        prefix_len: int = 4,
        enable_exp_d: bool = False,
        enable_exp_e: bool = False,
        enable_exp_f: bool = False,
        d_token_strategy: str = "first",
        d_max_posting_threshold: int = 500,
        f_max_posting_threshold: int = 500
    ):
        super().__init__(prefix_len=prefix_len)
        self.enable_exp_d = enable_exp_d
        self.enable_exp_e = enable_exp_e
        self.enable_exp_f = enable_exp_f
        self.d_token_strategy = d_token_strategy
        self.d_max_posting_threshold = d_max_posting_threshold
        self.f_max_posting_threshold = f_max_posting_threshold

        self.exp_d_index: Dict[Tuple[str, str], List[str]] = {}
        self.exp_e_index: Dict[Tuple[str, str, str], List[str]] = {}
        self.exp_f_index: Dict[Tuple[str, str, str], List[str]] = {}

    def add_record(
        self,
        entity_id: str,
        business_name: str,
        business_address: str,
        country: str
    ) -> None:
        # Add baseline Keys A, B, C
        super().add_record(entity_id, business_name, business_address, country)

        if not entity_id or not country:
            return

        c_norm = country.strip()

        # Experiment D: Token Anchor
        if self.enable_exp_d:
            kd = generate_key_d_token_anchor(c_norm, business_name, token_strategy=self.d_token_strategy)
            if kd:
                if kd not in self.exp_d_index:
                    self.exp_d_index[kd] = []
                posting = self.exp_d_index[kd]
                if len(posting) < self.d_max_posting_threshold:
                    if not posting or posting[-1] != entity_id:
                        posting.append(entity_id)

        # Experiment E: 5-char Prefix
        if self.enable_exp_e:
            ke = generate_key_e_prefix5(c_norm, business_name, business_address)
            if ke:
                if ke not in self.exp_e_index:
                    self.exp_e_index[ke] = []
                posting = self.exp_e_index[ke]
                if not posting or posting[-1] != entity_id:
                    posting.append(entity_id)

        # Experiment F: Street Anchor
        if self.enable_exp_f:
            kf = generate_key_f_street_anchor(c_norm, business_address)
            if kf:
                if kf not in self.exp_f_index:
                    self.exp_f_index[kf] = []
                posting = self.exp_f_index[kf]
                if len(posting) < self.f_max_posting_threshold:
                    if not posting or posting[-1] != entity_id:
                        posting.append(entity_id)

    def query_record(
        self,
        business_name: str,
        business_address: str,
        country: str
    ) -> Dict[str, BlockingEvidence]:
        # Query baseline keys A, B, C
        candidates = super().query_record(business_name, business_address, country)
        if not country:
            return candidates

        c_norm = country.strip()

        # Query Exp D
        if self.enable_exp_d:
            kd = generate_key_d_token_anchor(c_norm, business_name, token_strategy=self.d_token_strategy)
            if kd and kd in self.exp_d_index:
                for cand_id in self.exp_d_index[kd]:
                    if cand_id not in candidates:
                        candidates[cand_id] = BlockingEvidence()
                    candidates[cand_id].matched_key_d = True

        # Query Exp E
        if self.enable_exp_e:
            ke = generate_key_e_prefix5(c_norm, business_name, business_address)
            if ke and ke in self.exp_e_index:
                for cand_id in self.exp_e_index[ke]:
                    if cand_id not in candidates:
                        candidates[cand_id] = BlockingEvidence()
                    candidates[cand_id].matched_prefix_key = True

        # Query Exp F
        if self.enable_exp_f:
            kf = generate_key_f_street_anchor(c_norm, business_address)
            if kf and kf in self.exp_f_index:
                for cand_id in self.exp_f_index[kf]:
                    if cand_id not in candidates:
                        candidates[cand_id] = BlockingEvidence()
                    candidates[cand_id].matched_key_f = True

        return candidates

    def get_posting_list_stats(self, index_dict: Dict) -> Dict[str, float]:
        """Compute detailed percentile and threshold distribution statistics for an index."""
        if not index_dict:
            return {
                "unique_keys": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0,
                "gt_25": 0, "gt_50": 0, "gt_100": 0, "gt_250": 0, "gt_500": 0, "gt_1000": 0
            }

        lengths = sorted([len(p) for p in index_dict.values()])
        n = len(lengths)

        def pct(p):
            idx = int(p * (n - 1))
            return lengths[idx]

        return {
            "unique_keys": n,
            "avg": round(sum(lengths) / n, 2),
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": lengths[-1],
            "gt_25": sum(1 for l in lengths if l > 25),
            "gt_50": sum(1 for l in lengths if l > 50),
            "gt_100": sum(1 for l in lengths if l > 100),
            "gt_250": sum(1 for l in lengths if l > 250),
            "gt_500": sum(1 for l in lengths if l > 500),
            "gt_1000": sum(1 for l in lengths if l > 1000),
        }
