"""
Compare plain RAG (baseline) vs RAG + code-mixed rewriting, per input form.

CSV columns: claim_id, form (devanagari|roman|mixed), text, label, source_url (optional)
  label = Supported | Refuted | Not enough evidence   (rows labelled TODO are skipped for accuracy)
  source_url = the published fact-check page for the claim. If given, we also measure whether
               the system found that page (an objective retrieval measure).

Run:      python evaluation/eval.py evaluation/test_claims.csv
Resume:   python evaluation/eval.py evaluation/test_claims.csv --resume   (skips finished runs)

Outputs (next to the input CSV):
  results_detail.csv   one line per run, written as the run finishes (nothing is lost if it stops)
  results_summary.csv  the final table
"""
import csv
import os
import sys
import time
from collections import defaultdict
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402
import pipeline  # noqa: E402

DETAIL_FIELDS = ["claim_id", "form", "system", "text", "docs_found", "best_similarity",
                 "verdict", "confidence", "source_found", "correct", "error"]
PAUSE_SECONDS = 3  # gap between runs so the search engine does not rate-limit us


def slug(url: str) -> str:
    """Last path piece of a URL, so amp / www / query variants of one page compare equal."""
    path = urlparse(url.strip()).path.rstrip("/")
    return path.rsplit("/", 1)[-1].lower()


def run_one(row: dict, use_rewriting: bool) -> dict:
    base = {"claim_id": row["claim_id"], "form": row["form"],
            "system": "with_rewriting" if use_rewriting else "baseline", "text": row["text"],
            "docs_found": 0, "best_similarity": 0.0, "verdict": "", "confidence": "",
            "source_found": "", "correct": "", "error": ""}
    try:
        out = pipeline.run(row["text"], use_rewriting=use_rewriting)
    except Exception as exc:  # one bad run must not kill the whole evaluation
        base["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return base
    if "error" in out:
        base["error"] = out["error"]
        return base
    evidence = out.get("evidence", [])
    base["docs_found"] = out.get("steps", {}).get("docs_found", 0)
    base["best_similarity"] = round(max((e["score"] for e in evidence), default=0.0), 3)
    base["verdict"] = out["result"]["verdict"]
    base["confidence"] = out["result"]["confidence"]
    src = (row.get("source_url") or "").strip()
    if src:
        wanted = slug(src)
        base["source_found"] = int(any(slug(e["url"]) == wanted for e in evidence))
    label = row.get("label", "")
    if label and label != "TODO":
        base["correct"] = int(base["verdict"] == label)
    return base


def summarize(rows: list) -> list:
    """Group run rows by (system, form) and compute the table. Pure function, easy to test."""
    groups = defaultdict(list)
    for r in rows:
        if not r.get("error"):
            groups[(r["system"], r["form"])].append(r)
    out = []
    for (system, form), g in sorted(groups.items()):
        n = len(g)
        relevant = sum(1 for r in g if float(r["best_similarity"]) >= config.MIN_SIMILARITY)
        found = [r for r in g if r["source_found"] != ""]
        graded = [r for r in g if r["correct"] != ""]
        out.append({
            "system": system, "form": form, "n": n,
            "relevant_evidence": round(relevant / n, 2),
            "source_found": round(sum(int(r["source_found"]) for r in found) / len(found), 2) if found else "",
            "avg_best_similarity": round(sum(float(r["best_similarity"]) for r in g) / n, 2),
            "accuracy": round(sum(int(r["correct"]) for r in graded) / len(graded), 2) if graded else "",
        })
    return out


def print_table(table: list, errors: int):
    cols = ["system", "form", "n", "relevant_evidence", "source_found", "avg_best_similarity", "accuracy"]
    print()
    print(f"{'system':16}{'form':13}{'n':>3}{'relevant':>10}{'source':>8}{'avg sim':>9}{'accuracy':>10}")
    for t in table:
        print(f"{t['system']:16}{t['form']:13}{t['n']:>3}{str(t['relevant_evidence']):>10}"
              f"{str(t['source_found']):>8}{str(t['avg_best_similarity']):>9}{str(t['accuracy']):>10}")
    if errors:
        print(f"\n{errors} run(s) failed and are not counted. See the error column in results_detail.csv.")
    return cols


def main(path: str, resume: bool = False):
    folder = os.path.dirname(os.path.abspath(path))
    detail_path = os.path.join(folder, "results_detail.csv")
    summary_path = os.path.join(folder, "results_summary.csv")

    with open(path, encoding="utf-8") as f:
        claims = list(csv.DictReader(f))

    done_rows, done_keys = [], set()
    if resume and os.path.exists(detail_path):
        with open(detail_path, encoding="utf-8-sig") as f:
            done_rows = list(csv.DictReader(f))
        done_keys = {(r["claim_id"], r["form"], r["system"]) for r in done_rows if not r["error"]}
        done_rows = [r for r in done_rows if not r["error"]]

    total = len(claims) * 2
    all_rows = list(done_rows)
    mode = "a" if (resume and os.path.exists(detail_path)) else "w"
    errors = 0
    with open(detail_path, mode, newline="", encoding="utf-8-sig") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=DETAIL_FIELDS)
        if mode == "w":
            writer.writeheader()
        i = 0
        for row in claims:
            for use_rewriting in (False, True):
                i += 1
                system = "with_rewriting" if use_rewriting else "baseline"
                if (row["claim_id"], row["form"], system) in done_keys:
                    continue
                res = run_one(row, use_rewriting)
                writer.writerow(res)
                out_f.flush()
                if res["error"]:
                    errors += 1
                else:
                    all_rows.append(res)
                print(f"[{i}/{total}] claim {res['claim_id']} {res['form']:10} {system:15} "
                      f"docs={res['docs_found']} best_sim={res['best_similarity']} "
                      f"verdict={res['verdict'] or 'ERROR'}", flush=True)
                time.sleep(PAUSE_SECONDS)

    table = summarize(all_rows)
    cols = print_table(table, errors)
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    print(f"\nSaved: {detail_path}\n       {summary_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "evaluation/test_claims.csv", resume="--resume" in sys.argv)
