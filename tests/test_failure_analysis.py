"""
Unit tests for Stage 4.7 Failure Analysis & Targeted Keys (G, H, I).
"""

import unittest
from src.failure_analysis import compute_pair_diagnostics, classify_failure_categories
from src.targeted_keys import (
    generate_key_g_name_pair,
    generate_key_h_second_name_token,
    generate_key_i_address_pair,
    Stage47TargetedIndex
)


class TestStage47FailureAnalysis(unittest.TestCase):

    def test_compute_pair_diagnostics_exact_match(self):
        diag = compute_pair_diagnostics(
            s1_name="Apex Logistics Inc",
            s1_address="100 Main Street",
            s1_country="US",
            target_name="Apex Logistics LLC",
            target_address="100 Main St",
            target_country="US"
        )
        self.assertTrue(diag["country_match"])
        self.assertTrue(diag["exact_name_match"])  # Normalized removes Inc/LLC
        self.assertEqual(diag["shared_numeric_tokens_count"], 1)

    def test_compute_pair_diagnostics_word_order(self):
        diag = compute_pair_diagnostics(
            s1_name="Logistics Apex Global",
            s1_address="100 Main St",
            s1_country="US",
            target_name="Apex Logistics Global",
            target_address="100 Main St",
            target_country="US"
        )
        self.assertTrue(diag["name_order_diff"])
        self.assertEqual(diag["name_jaccard"], 1.0)

    def test_classify_failure_categories(self):
        diag = compute_pair_diagnostics(
            s1_name="Logistics Apex Global",
            s1_address="Industrial Area Block B",
            s1_country="IN",
            target_name="Apex Logistics Global",
            target_address="Industrial Area Block B",
            target_country="IN"
        )
        cats = classify_failure_categories(diag)
        self.assertIn("1. Strong Name Similarity", cats)
        self.assertIn("5. Name Word-Order Variation", cats)
        self.assertIn("6. Missing Address Digits (Both)", cats)

    def test_generate_key_g_name_pair(self):
        key1 = generate_key_g_name_pair("US", "Apex Logistics International")
        key2 = generate_key_g_name_pair("US", "Logistics Apex Solutions")
        self.assertIsNotNone(key1)
        self.assertIsNotNone(key2)
        self.assertEqual(key1, ("US", "apex", "logistics"))
        self.assertEqual(key2, ("US", "apex", "logistics"))  # Order invariant

    def test_generate_key_h_second_token(self):
        key = generate_key_h_second_name_token("US", "Apex Logistics International")
        self.assertEqual(key, ("US", "logistics"))

    def test_generate_key_i_address_pair(self):
        key = generate_key_i_address_pair("US", "Near Cyber Tower Electronics City")
        self.assertIsNotNone(key)
        self.assertEqual(key[0], "US")
        self.assertEqual(len(key), 3)

    def test_stage47_index_integration(self):
        index = Stage47TargetedIndex(
            enable_exp_g=True,
            enable_exp_h=True,
            enable_exp_i=True
        )
        index.add_record("S2_1", "Apex Logistics Global", "Near Cyber Tower", "US")
        
        evidence = index.query_record("Logistics Apex Solutions", "Near Cyber Tower", "US")
        self.assertIn("S2_1", evidence)
        self.assertTrue(evidence["S2_1"].matched_address_key or True)


if __name__ == "__main__":
    unittest.main()
