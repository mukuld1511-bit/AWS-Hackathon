"""
Unit tests for src/blocking_index.py
Uses standard library unittest framework.
"""

import unittest
from src.blocking_index import BlockingIndex, BlockingEvidence

class TestBlockingIndex(unittest.TestCase):

    def setUp(self):
        self.index = BlockingIndex(prefix_len=4)

    def test_add_and_query_exact_name(self):
        """Test inserting and querying Key A."""
        self.index.add_record("S2-100", "ABC Corporation", "100 Main St", "US")
        candidates = self.index.query_record("ABC Inc", "200 Oak Ave", "US")
        
        self.assertIn("S2-100", candidates)
        self.assertTrue(candidates["S2-100"].matched_name_key)
        self.assertFalse(candidates["S2-100"].matched_prefix_key)
        self.assertFalse(candidates["S2-100"].matched_address_key)

    def test_country_isolation(self):
        """Ensure US queries never return India or France records."""
        self.index.add_record("S2-1", "ABC Corporation", "85 Wayne Avenue", "US")
        self.index.add_record("S3-1", "ABC Corporation", "85 Wayne Avenue", "India")
        self.index.add_record("S2-3", "ABC Corporation", "85 Wayne Avenue", "France")

        # Query US
        us_cands = self.index.query_record("ABC Corp", "85 Wayne Ave", "US")
        self.assertIn("S2-1", us_cands)
        self.assertNotIn("S3-1", us_cands)
        self.assertNotIn("S2-3", us_cands)

        # Query France
        fr_cands = self.index.query_record("ABC Corp", "85 Wayne Ave", "France")
        self.assertIn("S2-3", fr_cands)
        self.assertNotIn("S2-1", fr_cands)
        self.assertNotIn("S3-1", fr_cands)

    def test_s2_and_s3_ids_remain_distinct(self):
        """Ensure S2 and S3 IDs are stored as exact distinct raw strings."""
        self.index.add_record("S2-123", "Target Store", "50 Market St", "US")
        self.index.add_record("S3-123", "Target Store", "50 Market St", "US")

        cands = self.index.query_record("Target Store", "50 Market St", "US")
        self.assertIn("S2-123", cands)
        self.assertIn("S3-123", cands)
        self.assertEqual(len(cands), 2)

    def test_multiple_candidates_under_one_key(self):
        """Ensure multiple candidates are retrieved under a single key."""
        self.index.add_record("S2-1", "Alpha Inc", "10 First St", "US")
        self.index.add_record("S2-2", "Alpha Corp", "20 Second St", "US")
        self.index.add_record("S3-1", "Alpha Limited", "30 Third St", "US")

        cands = self.index.query_record("Alpha Co", "100 High St", "US")
        self.assertIn("S2-1", cands)
        self.assertIn("S2-2", cands)
        self.assertIn("S3-1", cands)

    def test_same_entity_matching_multiple_keys(self):
        """Ensure candidate matching multiple keys accumulates evidence flags correctly."""
        # S2-1 matches name ("alpha"), prefix ("alph", "85"), and address ("85_wayne")
        self.index.add_record("S2-1", "Alpha Corp", "85 Wayne Avenue", "US")

        cands = self.index.query_record("Alpha Inc", "85 Wayne Ave", "US")
        self.assertIn("S2-1", cands)
        ev = cands["S2-1"]
        self.assertTrue(ev.matched_name_key)
        self.assertTrue(ev.matched_prefix_key)
        self.assertTrue(ev.matched_address_key)
        self.assertEqual(ev.evidence_count, 3)

    def test_duplicate_insertion_prevention(self):
        """Ensure duplicate record insertions do not duplicate postings."""
        self.index.add_record("S2-100", "ABC Corp", "100 Main St", "US")
        self.index.add_record("S2-100", "ABC Corp", "100 Main St", "US")  # Repeated call

        key = ("US", "abc")
        posting = self.index.name_index[key]
        self.assertEqual(posting, ["S2-100"])

    def test_empty_and_missing_address_handling(self):
        """Ensure empty/missing addresses do not crash and generate only valid keys."""
        self.index.add_record("S2-50", "No Address Inc", "", "US")
        self.index.add_record("S3-50", "No Address Inc", None, "US")

        cands = self.index.query_record("No Address Corp", "", "US")
        self.assertIn("S2-50", cands)
        self.assertIn("S3-50", cands)
        self.assertTrue(cands["S2-50"].matched_name_key)
        self.assertFalse(cands["S2-50"].matched_prefix_key)
        self.assertFalse(cands["S2-50"].matched_address_key)

    def test_deterministic_repeated_indexing_and_querying(self):
        """Ensure indexing and querying are deterministic across runs."""
        rec = ("S2-99", "Delta Tech", "42 Wall St", "US")
        self.index.add_record(*rec)
        res1 = self.index.query_record("Delta Tech", "42 Wall St", "US")
        res2 = self.index.query_record("Delta Tech", "42 Wall St", "US")
        self.assertEqual(res1.keys(), res2.keys())
        self.assertEqual(res1["S2-99"].evidence_count, res2["S2-99"].evidence_count)

    def test_statistics(self):
        """Test summary statistics calculation."""
        self.index.add_record("S2-1", "ABC Inc", "85 Wayne Ave", "US")
        stats = self.index.get_statistics()
        self.assertEqual(stats["total_records"], 1)
        self.assertEqual(stats["unique_name_keys"], 1)
        self.assertEqual(stats["total_postings"], 3)


if __name__ == "__main__":
    unittest.main()
