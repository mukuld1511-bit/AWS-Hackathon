"""
Unit tests for src/blocking_keys.py
Uses standard library unittest framework.
"""

import unittest
from src.blocking_keys import (
    generate_key_a,
    generate_key_b,
    generate_key_c,
)

class TestBlockingKeys(unittest.TestCase):

    def test_key_a_exact_normalized_name(self):
        """Test Key A generation."""
        key = generate_key_a("US", "ABC Corporation")
        self.assertEqual(key, ("US", "abc"))

    def test_key_a_empty_name(self):
        """Key A returns None if business name is empty."""
        self.assertIsNone(generate_key_a("US", ""))
        self.assertIsNone(generate_key_a("US", "   "))
        self.assertIsNone(generate_key_a("US", None))

    def test_key_b_prefix_address_numbers(self):
        """Test Key B generation."""
        key = generate_key_b("US", "Maure Williams Colombier", "85 Wayne Avenue", prefix_len=4)
        self.assertEqual(key, ("US", "maur", "85"))

    def test_key_b_configurable_prefix_len(self):
        """Test configurable prefix length."""
        key3 = generate_key_b("US", "Maure Williams", "85 Wayne Ave", prefix_len=3)
        self.assertEqual(key3, ("US", "mau", "85"))
        key5 = generate_key_b("US", "Maure Williams", "85 Wayne Ave", prefix_len=5)
        self.assertEqual(key5, ("US", "maure", "85"))

    def test_key_b_missing_name_or_address_numbers(self):
        """Key B returns None if name or address numbers are missing."""
        self.assertIsNone(generate_key_b("US", "", "85 Wayne Avenue"))
        self.assertIsNone(generate_key_b("US", "Maure Williams", "Wayne Avenue"))  # No digits
        self.assertIsNone(generate_key_b("US", "Maure Williams", ""))

    def test_key_c_address_anchor(self):
        """Test Key C generation."""
        key = generate_key_c("US", "85 Wayne Avenue")
        self.assertEqual(key, ("US", "85_wayne"))

    def test_key_c_empty_or_missing_anchor(self):
        """Key C returns None if address anchor cannot be built."""
        self.assertIsNone(generate_key_c("US", ""))
        self.assertIsNone(generate_key_c("US", "Wayne Avenue"))  # No numeric token
        self.assertIsNone(generate_key_c("US", "85"))            # No word token

    def test_country_isolation_keys(self):
        """Keys for different countries must remain distinct."""
        key_us = generate_key_a("US", "ABC Inc")
        key_in = generate_key_a("India", "ABC Inc")
        key_fr = generate_key_a("France", "ABC Inc")
        
        self.assertEqual(key_us, ("US", "abc"))
        self.assertEqual(key_in, ("India", "abc"))
        self.assertEqual(key_fr, ("France", "abc"))
        self.assertNotEqual(key_us, key_in)
        self.assertNotEqual(key_us, key_fr)


if __name__ == "__main__":
    unittest.main()
