"""
Unit tests for src/normalization.py
Uses standard library unittest framework only.
"""

import unittest
from src.normalization import (
    normalize_name,
    extract_address_numbers,
    extract_address_anchor,
)

class TestNormalizationModule(unittest.TestCase):

    # -------------------------------------------------------------------------
    # Test normalize_name()
    # -------------------------------------------------------------------------

    def test_normalize_name_none_and_empty(self):
        """1. None, 2. Empty string, 3. Whitespace-only input."""
        self.assertEqual(normalize_name(None), "")
        self.assertEqual(normalize_name(""), "")
        self.assertEqual(normalize_name("   \t\n  "), "")

    def test_normalize_name_lowercasing(self):
        """4. Lowercasing."""
        self.assertEqual(normalize_name("ACME PRODUCTS"), "acme products")

    def test_normalize_name_punctuation_and_symbols(self):
        """5. Punctuation, 6. Apostrophes, 7. Hyphens."""
        self.assertEqual(normalize_name("O'Reilly Barbershop Inc."), "o reilly barbershop")
        self.assertEqual(normalize_name("Foo-Bar LLC"), "foo bar")
        self.assertEqual(normalize_name("B+ Retail & Services"), "b retail")

    def test_normalize_name_legal_suffixes(self):
        """8. Legal suffix removal, 9. Multi-token suffixes."""
        self.assertEqual(normalize_name("ABC Corporation"), "abc")
        self.assertEqual(normalize_name("XYZ Pvt Ltd"), "xyz")
        self.assertEqual(normalize_name("Global Tech Private Limited"), "global tech")
        self.assertEqual(normalize_name("Apex Solutions Inc"), "apex")
        self.assertEqual(normalize_name("Alpha Enterprises Group"), "alpha")

    def test_normalize_name_internal_word_preservation(self):
        """Ensure non-suffix internal words are preserved."""
        self.assertEqual(normalize_name("Corporation Bank"), "bank")
        self.assertEqual(normalize_name("Center Point Retail Inc"), "point retail")

    def test_normalize_name_repeated_whitespace(self):
        """10. Repeated whitespace collapsing."""
        self.assertEqual(normalize_name("  Mega   Store   LLC  "), "mega store")

    def test_normalize_name_unicode(self):
        """15. Unicode character preservation."""
        self.assertEqual(normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड"), "राम मार्केटिंग")
        self.assertEqual(normalize_name("SCI Ptit Àmicale Sarl"), "sci ptit àmicale")
        self.assertEqual(normalize_name("தமிழ் இன்வெஸ்ட்மெண்ட்ஸ் Ltd"), "தமிழ் இன்வெஸ்ட்மெண்ட்ஸ்")

    def test_normalize_name_determinism(self):
        """16. Deterministic repeated calls."""
        raw_name = "  Super-Tech Solutions, Pvt. Ltd.  "
        res1 = normalize_name(raw_name)
        res2 = normalize_name(raw_name)
        self.assertEqual(res1, res2)
        self.assertEqual(res1, "super tech")

    # -------------------------------------------------------------------------
    # Test extract_address_numbers()
    # -------------------------------------------------------------------------

    def test_extract_address_numbers_none_and_empty(self):
        """14. Missing address handling."""
        self.assertEqual(extract_address_numbers(None), "")
        self.assertEqual(extract_address_numbers(""), "")
        self.assertEqual(extract_address_numbers("   "), "")

    def test_extract_address_numbers_basic(self):
        """11. Address numbers extraction."""
        self.assertEqual(extract_address_numbers("85 Wayne Avenue"), "85")
        self.assertEqual(extract_address_numbers("85 Wayne Avenue, Floor 2, Apt 104"), "85_2_104")

    def test_extract_address_numbers_punctuation(self):
        """12. Address punctuation handling."""
        self.assertEqual(extract_address_numbers("85, Wayne Ave.#42"), "85_42")

    def test_extract_address_numbers_no_digits(self):
        """Address with no numeric tokens."""
        self.assertEqual(extract_address_numbers("Main Street, High Point"), "")

    # -------------------------------------------------------------------------
    # Test extract_address_anchor()
    # -------------------------------------------------------------------------

    def test_extract_address_anchor_none_and_empty(self):
        """Missing address handling."""
        self.assertEqual(extract_address_anchor(None), "")
        self.assertEqual(extract_address_anchor(""), "")
        self.assertEqual(extract_address_anchor("   "), "")

    def test_extract_address_anchor_standard(self):
        """13. Address anchor extraction."""
        self.assertEqual(extract_address_anchor("85 Wayne Avenue"), "85_wayne")
        self.assertEqual(extract_address_anchor("85, Wayne Ave."), "85_wayne")
        self.assertEqual(extract_address_anchor("85 Wayne Avenue, Floor 2"), "85_wayne")

    def test_extract_address_anchor_short_words(self):
        """Skip alphabetic words shorter than 3 characters."""
        self.assertEqual(extract_address_anchor("10 A St, Boston"), "10_boston")

    def test_extract_address_anchor_missing_components(self):
        """Return empty string if numeric or valid alphabetic token is missing."""
        self.assertEqual(extract_address_anchor("Wayne Avenue"), "")
        self.assertEqual(extract_address_anchor("12345"), "")


if __name__ == "__main__":
    unittest.main()
