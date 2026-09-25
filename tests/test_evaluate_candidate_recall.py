"""
Unit tests for src/evaluate_candidate_recall.py
Uses standard library unittest framework.
"""

import unittest
from src.blocking_index import BlockingIndex, BlockingEvidence
from src.ranking import RankingConfig
from src.evaluate_candidate_recall import (
    EvidenceBreakdown,
    RecallMetricsAtK,
    EvaluationSummary,
    categorize_true_match_evidence,
    evaluate_ground_truth_recall,
)

class TestCandidateRecallEvaluation(unittest.TestCase):

    def setUp(self):
        # Create a small synthetic index
        self.index = BlockingIndex(prefix_len=4)
        
        # S2-10: US | "Acme Corp" | "100 Main St" (Matches Key A, Key B, Key C for Acme)
        self.index.add_record("S2-10", "Acme Corp", "100 Main St", "US")
        # S2-20: US | "Acme Services" | "100 Main St" (Matches Key B, Key C)
        self.index.add_record("S2-20", "Acme Services", "100 Main St", "US")
        # S3-30: US | "Acme Corporation" | "200 Main St" (Matches Key A)
        self.index.add_record("S3-30", "Acme Corporation", "200 Main St", "US")
        # S2-40: US | "Unrelated Store" | "50 Oak St"
        self.index.add_record("S2-40", "Unrelated Store", "50 Oak St", "US")

    def test_evidence_categorization(self):
        """Verify evidence breakdown categorization."""
        eb = EvidenceBreakdown()
        
        # Key A+B+C
        categorize_true_match_evidence(BlockingEvidence(True, True, True), eb)
        self.assertEqual(eb.key_abc, 1)

        # Key A only
        categorize_true_match_evidence(BlockingEvidence(True, False, False), eb)
        self.assertEqual(eb.key_a_only, 1)

        # Key B+C
        categorize_true_match_evidence(BlockingEvidence(False, True, True), eb)
        self.assertEqual(eb.key_bc, 1)

        self.assertEqual(eb.total_found, 3)

    def test_synthetic_ground_truth_evaluation(self):
        """Test ground truth evaluation on synthetic S1 queries."""
        # Create synthetic ground truth mapping
        # S1-1: Non-singleton, true matches = {S2-10, S3-30}
        # S1-2: Singleton, true matches = set()
        # S1-3: Non-singleton, true matches = {S2-999} (Not in index -> Blocking failure)
        gt_map = {
            "S1-1": {"S2-10", "S3-30"},
            "S1-2": set(),
            "S1-3": {"S2-999"}
        }

        # Create temporary synthetic TSV for S1
        import tempfile
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-1\tAcme Inc\t100 Main St\tUS\n")
            f.write("S1-2\tUnique Lone Business\t999 Remote Rd\tUS\n")
            f.write("S1-3\tMissing Match Entity\t77 Secret Ave\tUS\n")
            temp_path = f.name

        summary = evaluate_ground_truth_recall(
            index=self.index,
            s1_source_path=temp_path,
            ground_truth_map=gt_map,
            k_values=(1, 2, 5),
            ranking_config=RankingConfig(name_weight=3.0, prefix_weight=2.0, address_weight=1.5)
        )

        # Verify summary statistics
        self.assertEqual(summary.total_s1_eval, 3)
        self.assertEqual(summary.total_singleton_s1, 1)
        self.assertEqual(summary.total_non_singleton_s1, 2)
        self.assertEqual(summary.total_true_matches, 3)  # 2 for S1-1 + 1 for S1-3

        # Blocking recall: S2-10 and S3-30 found for S1-1 (2 matches). S2-999 missed (1 match).
        self.assertEqual(summary.blocking_recovered_matches, 2)
        self.assertAlmostEqual(summary.blocking_recall, 2 / 3, places=4)

        # Check evidence breakdown: S2-999 not found = 1
        self.assertEqual(summary.evidence_breakdown.not_found, 1)

        # Check K=1 metrics (only top 1 candidate retained per S1)
        m_k1 = summary.metrics_by_k[1]
        self.assertEqual(m_k1.recovered_true_matches, 1)  # Only 1 of {S2-10, S3-30} retained in top 1
        self.assertEqual(m_k1.full_recall_count, 0)        # Neither non-singleton got ALL matches

        # Check K=5 metrics (all candidates retained)
        m_k5 = summary.metrics_by_k[5]
        self.assertEqual(m_k5.recovered_true_matches, 2)  # S2-10 & S3-30 both recovered
        self.assertEqual(m_k5.full_recall_count, 1)        # S1-1 recovered ALL true matches

        import os
        os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
