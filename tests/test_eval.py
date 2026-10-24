import csv
import importlib.util
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("eval_mod", os.path.join(ROOT, "evaluation", "eval.py"))
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)


def fake_run(text, use_rewriting=True):
    """Pretend pipeline: rewriting finds the right page, baseline finds junk."""
    if use_rewriting:
        ev = [{"score": 0.62, "url": "https://example.com/amp/fact-check/old-video-factcheck-1"}]
        verdict = "Refuted"
    else:
        ev = [{"score": 0.20, "url": "https://example.com/other"}]
        verdict = "Not enough evidence"
    return {"steps": {"docs_found": 4}, "evidence": ev,
            "result": {"verdict": verdict, "confidence": "low"}}


class TestEval(unittest.TestCase):
    def test_slug_ignores_amp_www_query(self):
        a = eval_mod.slug("https://www.site.com/amp/fact-check/old-video-1/?utm=x")
        b = eval_mod.slug("http://site.com/fact-check/old-video-1")
        self.assertEqual(a, b)

    def test_main_end_to_end_with_fake_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "claims.csv")
            with open(src, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["claim_id", "form", "text", "label", "source_url"])
                w.writerow([1, "roman", "Pune madhe pur aala aahe", "Refuted",
                            "https://example.com/fact-check/old-video-factcheck-1"])
                w.writerow([1, "devanagari", "पुणे मध्ये पूर आला आहे", "Refuted",
                            "https://example.com/fact-check/old-video-factcheck-1"])
            with mock.patch.object(eval_mod.pipeline, "run", side_effect=fake_run), \
                    mock.patch.object(eval_mod, "PAUSE_SECONDS", 0):
                eval_mod.main(src)
            with open(os.path.join(d, "results_summary.csv"), encoding="utf-8-sig") as f:
                rows = {(r["system"], r["form"]): r for r in csv.DictReader(f)}
            on = rows[("with_rewriting", "roman")]
            off = rows[("baseline", "roman")]
            self.assertEqual(on["accuracy"], "1.0")
            self.assertEqual(on["source_found"], "1.0")
            self.assertEqual(off["accuracy"], "0.0")
            self.assertEqual(off["source_found"], "0.0")

    def test_failed_run_is_recorded_not_fatal(self):
        def boom(text, use_rewriting=True):
            raise RuntimeError("connection reset")
        with mock.patch.object(eval_mod.pipeline, "run", side_effect=boom):
            res = eval_mod.run_one({"claim_id": 1, "form": "roman", "text": "x", "label": "Refuted"}, True)
        self.assertIn("connection reset", res["error"])


if __name__ == "__main__":
    unittest.main()
