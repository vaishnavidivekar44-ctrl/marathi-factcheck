import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ranker  # noqa: E402

try:
    import retriever
except ImportError:  # requests / bs4 missing: skip those tests
    retriever = None


class TestRankerHelpers(unittest.TestCase):
    def test_latest_year(self):
        self.assertEqual(ranker.latest_year("Updated: August 8, 2019 19:30 IST"), 2019)
        self.assertEqual(ranker.latest_year("Aug 03, 2026 and Jul 6, 2025"), 2026)
        self.assertIsNone(ranker.latest_year("no year here, 12345 is not a year"))

    def test_one_per_url_keeps_best(self):
        chunks = [
            {"url": "a", "score": 0.9}, {"url": "a", "score": 0.8},
            {"url": "b", "score": 0.7},
        ]
        out = ranker.one_per_url(chunks)
        self.assertEqual([c["url"] for c in out], ["a", "b"])
        self.assertEqual(out[0]["score"], 0.9)


@unittest.skipIf(retriever is None, "retriever dependencies not installed")
class TestBlocklist(unittest.TestCase):
    def test_social_media_blocked(self):
        self.assertTrue(retriever.is_blocked("https://www.instagram.com/reel/abc/"))
        self.assertTrue(retriever.is_blocked("https://www.youtube.com/watch?v=1"))
        self.assertTrue(retriever.is_blocked("https://m.facebook.com/x"))

    def test_news_not_blocked(self):
        self.assertFalse(retriever.is_blocked("https://www.esakal.com/pune/story"))
        self.assertFalse(retriever.is_blocked("https://marathi.ndtv.com/pune"))


if __name__ == "__main__":
    unittest.main()
