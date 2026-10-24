import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import verifier  # noqa: E402


class TestSimilarityGuard(unittest.TestCase):
    def test_unrelated_evidence_gives_not_enough(self):
        # Scores copied from a real run with the rewriting layer off (best 0.24, then 0.12).
        evidence = [
            {"chunk": "x", "url": "u1", "score": 0.24},
            {"chunk": "y", "url": "u2", "score": 0.12},
        ]
        out = verifier.verify("Pune madhe pur aala aahe", evidence)
        self.assertEqual(out["verdict"], verifier.NOT_ENOUGH)
        self.assertEqual(out["method"], "similarity-guard")

    def test_no_evidence(self):
        out = verifier.verify("anything", [])
        self.assertEqual(out["verdict"], verifier.NOT_ENOUGH)


if __name__ == "__main__":
    unittest.main()
