"""End-to-end pipeline: input -> normalize -> live retrieval -> rank -> verdict."""
import normalizer
import ranker
import retriever
import verifier


def run(text: str, use_rewriting: bool = True) -> dict:
    """use_rewriting=False is the plain-RAG baseline used in the evaluation."""
    text = text.strip()
    steps = {}

    if text.startswith(("http://", "https://")):
        page = retriever.fetch_page(text)
        claim = (page["title"] + ". " + page["text"][:300]).strip(". ")
        steps["link_extracted"] = claim
        if not claim:
            return {"error": "Could not read that link.", "steps": steps}
    else:
        claim = text

    if use_rewriting:
        proc = normalizer.process(claim)
        queries = proc["queries"]
        steps.update({"script": proc["script"], "normalized": proc["normalized"],
                      "keywords": proc["keywords"]})
        claim_for_match = proc["normalized"]
    else:
        queries = [claim]
        claim_for_match = claim
        steps["script"] = normalizer.detect_script(claim)

    steps["queries"] = queries
    docs = retriever.retrieve(queries)
    steps["docs_found"] = len(docs)
    evidence = ranker.rank(claim_for_match, docs)
    result = verifier.verify(claim_for_match, evidence)
    return {"claim": claim, "steps": steps, "evidence": evidence, "result": result}


if __name__ == "__main__":
    import json
    import sys
    out = run(" ".join(sys.argv[1:]) or "पुण्यात पूर आला आहे")
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
