"""
Unit tests for Stage 4.6 extensions in src/experimental_blocking.py
Uses standard library unittest framework.
"""

import unittest
from src.experimental_blocking import (
    extract_first_significant_token,
    extract_longest_significant_token,
    generate_key_d_token_anchor,
    generate_key_e_prefix5,
    generate_key_f_street_anchor,
    Stage46ExperimentalIndex,
    COMMON_BUSINESS_STOP_WORDS,
)

class TestStage46ExperimentalBlocking(unittest.TestCase):

    def test_longest_token_extraction(self):
        """Test longest significant token extraction."""
        # Tokens: ['orelee', 'barbershop'] -> longest is 'barbershop' (len 10 vs len 6)
        self.assertEqual(extract_longest_significant_token("Orelee's Barbershop Inc"), "barbershop")
        # First significant token was 'orelee'
        self.assertEqual(extract_first_significant_token("Orelee's Barbershop Inc"), "orelee")

    def test_token_strategy_key_generation(self):
        """Test token strategy options in Key D."""
        key_first = generate_key_d_token_anchor("US", "Orelee's Barbershop", token_strategy="first")
        key_longest = generate_key_d_token_anchor("US", "Orelee's Barbershop", token_strategy="longest")
        
        self.assertEqual(key_first, ("US", "orelee"))
        self.assertEqual(key_longest, ("US", "barbershop"))

    def test_threshold_capping(self):
        """Test configurable threshold capping in Stage46ExperimentalIndex."""
        idx = Stage46ExperimentalIndex(
            enable_exp_d=True,
            d_max_posting_threshold=2
        )
        idx.add_record("S2-1", "Orelee's Barbershop", "100 Main St", "US")
        idx.add_record("S2-2", "Orelee's Salon", "200 Main St", "US")
        idx.add_record("S2-3", "Orelee's Spa", "300 Main St", "US")  # Should be capped

        kd = ("US", "orelee")
        self.assertEqual(len(idx.exp_d_index[kd]), 2)

    def test_posting_list_stats_calculation(self):
        """Test posting list statistics calculation."""
        idx = Stage46ExperimentalIndex(enable_exp_d=True, d_max_posting_threshold=100)
        for i in range(30):
            idx.add_record(f"S2-{i}", "Orelee Barbershop", "100 Main St", "US")
        
        stats = idx.get_posting_list_stats(idx.exp_d_index)
        self.assertEqual(stats["unique_keys"], 1)
        self.assertEqual(stats["p50"], 30)
        self.assertEqual(stats["gt_25"], 1)
        self.assertEqual(stats["gt_50"], 0)


if __name__ == "__main__":
    unittest.main()
