import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import normalizer  # noqa: E402


class TestNormalizer(unittest.TestCase):
    def test_script_detection(self):
        self.assertEqual(normalizer.detect_script("Aata bagha"), "roman")
        self.assertEqual(normalizer.detect_script("पुण्यात पूर"), "devanagari")
        self.assertEqual(normalizer.detect_script("पुण्यात flood"), "mixed")

    def test_roman_to_devanagari(self):
        out = normalizer.normalize("Aata bagha kay zala")
        self.assertEqual(out["normalized"], "आता बघा काय झाला")

    def test_filler_removed_from_keywords(self):
        kws = normalizer.extract_keywords("आता बघा नेता काय बोलला")
        self.assertEqual(kws, ["नेता", "बोलला"])

    def test_bilingual_queries(self):
        qs = normalizer.process("Pune madhe flood aala")["queries"]
        self.assertTrue(any("पूर" in q for q in qs))      # English 'flood' -> Marathi
        self.assertTrue(any("पुणे" in q for q in qs))     # Roman 'Pune' -> Devanagari


if __name__ == "__main__":
    unittest.main()
