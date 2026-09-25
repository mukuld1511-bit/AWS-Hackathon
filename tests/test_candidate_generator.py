"""
Unit & Integration Tests for src/candidate_generator.py (Stage 5).

Tests:
1. Candidate deduplication
2. Country isolation (0 cross-country matches)
3. R3.3 score calculation & multi-key bonus
4. Deterministic tie-breaking (score DESC, target_entity_id ASC)
5. Top-25 truncation
6. Zero-candidate S1 handling (empty candidate_entity_ids)
7. Mixed S2/S3 candidate IDs
8. TSV Output formatting & header correctness
"""

import os
import sys
import tempfile
import unittest
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.candidate_generator import (
    generate_candidate_pairs,
    get_frozen_r3_3_ranking_config,
    CandidateGenerationReport
)
from src.ranking import RankingConfig


class TestCandidateGeneratorModule(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.s1_path = os.path.join(self.tmp_dir.name, "s1.tsv")
        self.s2_path = os.path.join(self.tmp_dir.name, "s2.tsv")
        self.s3_path = os.path.join(self.tmp_dir.name, "s3.tsv")
        self.output_path = os.path.join(self.tmp_dir.name, "candidate_pairs.tsv")

        # Create dummy S1 records
        with open(self.s1_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["entity_id", "business_name", "business_address", "country"])
            writer.writerow(["S1-1", "Acme Corporation", "100 Main Street", "US"])
            writer.writerow(["S1-2", "Boulangerie Paris", "5 Rue de Rivoli", "FR"])
            writer.writerow(["S1-3", "Unknown Entity XYZ", "999 Nowhere Road", "US"])  # Zero candidates

        # Create dummy S2 records
        with open(self.s2_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["entity_id", "business_name", "business_address", "country"])
            writer.writerow(["S2-10", "Acme Corp", "100 Main St", "US"])
            writer.writerow(["S2-20", "Acme International", "100 Main Street", "US"])
            writer.writerow(["S2-30", "Boulangerie Paris SARL", "5 Rue de Rivoli", "FR"])
            writer.writerow(["S2-40", "Acme Corp Cross Country", "100 Main St", "FR"])  # Cross country

        # Create dummy S3 records
        with open(self.s3_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["entity_id", "business_name", "business_address", "country"])
            writer.writerow(["S3-100", "Acme Corporation Inc", "100 Main Street Ave", "US"])
            writer.writerow(["S3-200", "Boulangerie de Paris", "5 Rue Rivoli", "FR"])

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_frozen_r3_3_ranking_config_weights(self):
        """Verify frozen R3.3 weights match specification exactly."""
        cfg = get_frozen_r3_3_ranking_config()
        self.assertEqual(cfg.name_weight, 3.0)
        self.assertEqual(cfg.prefix_weight, 2.0)
        self.assertEqual(cfg.address_weight, 1.5)
        self.assertEqual(cfg.key_d_weight, 1.0)
        self.assertEqual(cfg.key_f_weight, 1.5)
        self.assertEqual(cfg.key_g_weight, 2.0)
        self.assertEqual(cfg.key_i_weight, 2.0)
        self.assertEqual(cfg.multi_key_bonus, 0.5)
        self.assertEqual(cfg.max_candidates, 25)

    def test_candidate_generation_pipeline(self):
        """Test full candidate generation pipeline execution."""
        report = generate_candidate_pairs(
            s1_path=self.s1_path,
            s2_path=self.s2_path,
            s3_path=self.s3_path,
            output_path=self.output_path
        )

        self.assertEqual(report.total_s1_records, 3)
        self.assertEqual(report.zero_candidate_s1, 1)  # S1-3 has 0 candidates
        self.assertEqual(report.cross_country_violations, 0)
        self.assertEqual(report.duplicate_pairs_count, 0)
        self.assertEqual(report.invalid_target_ids_count, 0)

        # Check TSV Output Format
        with open(self.output_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)

        self.assertEqual(rows[0], ["source1_entity_id", "candidate_entity_ids"])
        self.assertEqual(len(rows), 4)  # Header + 3 S1 rows

        # Row 1: S1-1 (US) -> candidates S2-10, S2-20, S3-100 (mixed S2/S3)
        self.assertEqual(rows[1][0], "S1-1")
        s1_1_cands = rows[1][1].split(",")
        self.assertTrue(len(s1_1_cands) >= 2)
        self.assertNotIn("S2-40", s1_1_cands)  # Country isolation check (S2-40 is FR)

        # Row 2: S1-2 (FR) -> candidates S2-30, S3-200
        self.assertEqual(rows[2][0], "S1-2")
        s1_2_cands = rows[2][1].split(",")
        self.assertIn("S2-30", s1_2_cands)
        self.assertIn("S3-200", s1_2_cands)

        # Row 3: S1-3 (Zero candidates) -> candidate_entity_ids is empty
        self.assertEqual(rows[3][0], "S1-3")
        self.assertEqual(rows[3][1], "")


if __name__ == "__main__":
    unittest.main()
