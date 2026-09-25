"""
Unit tests for src/ranking.py
Uses standard library unittest framework.
"""

import unittest
from src.blocking_index import BlockingEvidence
from src.ranking import (
    RankingConfig,
    RankedCandidate,
    score_candidate,
    rank_candidates,
)


def get_candidate_ids(ranked_list):
    return [c.entity_id for c in ranked_list]


class TestRankingModule(unittest.TestCase):

    def setUp(self):
        self.config = RankingConfig(name_weight=3.0, prefix_weight=2.0, address_weight=1.5, max_candidates=25)

    def test_score_calculation_individual_and_combinations(self):
        """Test score calculation for baseline key combinations."""
        # A only
        ev_a = BlockingEvidence(matched_name_key=True)
        self.assertEqual(score_candidate("S2-1", ev_a, self.config).score, 3.0)

        # B only
        ev_b = BlockingEvidence(matched_prefix_key=True)
        self.assertEqual(score_candidate("S2-2", ev_b, self.config).score, 2.0)

        # C only
        ev_c = BlockingEvidence(matched_address_key=True)
        self.assertEqual(score_candidate("S2-3", ev_c, self.config).score, 1.5)

        # A + B
        ev_ab = BlockingEvidence(matched_name_key=True, matched_prefix_key=True)
        self.assertEqual(score_candidate("S2-4", ev_ab, self.config).score, 5.0)

        # A + B + C
        ev_abc = BlockingEvidence(matched_name_key=True, matched_prefix_key=True, matched_address_key=True)
        self.assertEqual(score_candidate("S2-7", ev_abc, self.config).score, 6.5)

    def test_experimental_key_weights_and_bonus(self):
        """Test scoring with experimental keys D, F, G, I and multi-key bonus."""
        exp_cfg = RankingConfig(
            name_weight=3.0, prefix_weight=2.0, address_weight=1.5,
            key_d_weight=1.0, key_f_weight=1.5, key_g_weight=2.0, key_i_weight=2.0,
            multi_key_bonus=0.5
        )

        # Key D only
        ev_d = BlockingEvidence(matched_key_d=True)
        self.assertEqual(score_candidate("S2-D", ev_d, exp_cfg).score, 1.0)

        # Key G only
        ev_g = BlockingEvidence(matched_key_g=True)
        self.assertEqual(score_candidate("S2-G", ev_g, exp_cfg).score, 2.0)

        # Key I only
        ev_i = BlockingEvidence(matched_key_i=True)
        self.assertEqual(score_candidate("S2-I", ev_i, exp_cfg).score, 2.0)

        # Key D + G (2 keys: base 1.0 + 2.0 = 3.0, bonus +0.5 = 3.5)
        ev_dg = BlockingEvidence(matched_key_d=True, matched_key_g=True)
        rc_dg = score_candidate("S2-DG", ev_dg, exp_cfg)
        self.assertEqual(rc_dg.score, 3.5)
        self.assertEqual(rc_dg.evidence_count, 2)

    def test_evidence_count(self):
        """Verify score calculation and evidence_count."""
        ev = BlockingEvidence(matched_name_key=True, matched_address_key=True)
        rc = score_candidate("S2-100", ev, self.config)
        self.assertEqual(rc.evidence_count, 2)
        self.assertEqual(rc.score, 4.5)

    def test_descending_score_order(self):
        """Ensure candidates are sorted in descending score order."""
        evidence_map = {
            "S2-1": BlockingEvidence(matched_name_key=True),                                   # 3.0
            "S2-2": BlockingEvidence(matched_name_key=True, matched_prefix_key=True, matched_address_key=True), # 6.5
            "S2-3": BlockingEvidence(matched_prefix_key=True, matched_address_key=True),      # 3.5
        }
        ranked = rank_candidates(evidence_map, self.config)
        ids = get_candidate_ids(ranked)
        self.assertEqual(ids, ["S2-2", "S2-3", "S2-1"])

    def test_deterministic_tie_breaking(self):
        """Equal scores sorted by evidence count, then candidate_id ascending."""
        evidence_map = {
            "S2-200": BlockingEvidence(matched_name_key=True),  # 3.0, count 1
            "S2-100": BlockingEvidence(matched_name_key=True),  # 3.0, count 1
        }
        ranked = rank_candidates(evidence_map, self.config)
        ids = get_candidate_ids(ranked)
        self.assertEqual(ids, ["S2-100", "S2-200"])

    def test_candidate_caps_k(self):
        """Test candidate caps across K=10, 15, 20, 25, 30."""
        evidence_map = {
            f"S2-{i:03d}": BlockingEvidence(matched_name_key=True)
            for i in range(40)
        }

        for k in [10, 15, 20, 25, 30]:
            cfg = RankingConfig(max_candidates=k)
            ranked = rank_candidates(evidence_map, cfg)
            self.assertEqual(len(ranked), k)

    def test_empty_evidence_map(self):
        """Test empty evidence input."""
        ranked = rank_candidates({}, self.config)
        self.assertEqual(ranked, [])


if __name__ == "__main__":
    unittest.main()
