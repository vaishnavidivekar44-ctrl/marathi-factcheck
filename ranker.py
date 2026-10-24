"""
Module 3: Evidence ranking. Chunk retrieved articles, embed with LaBSE, rank by
similarity to the claim. LaBSE is cross-lingual, so Marathi, English and mixed
text land in the same vector space.
"""
import re

import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def chunk_text(text: str, window: int = 3, stride: int = 2) -> list:
    sentences = [s.strip() for s in re.split(r"(?<=[।.!?])\s+|\n+", text) if len(s.strip()) > 20]
    if not sentences:
        return [text] if text.strip() else []
    return [" ".join(sentences[i:i + window]) for i in range(0, len(sentences), stride)]


def latest_year(text: str):
    """Newest 4-digit year (2000-2039) mentioned in the text, or None. A rough age hint only."""
    years = [int(y) for y in re.findall(r"(?<!\d)(20[0-3]\d)(?!\d)", text)]
    return max(years) if years else None


def one_per_url(chunks: list) -> list:
    """Keep only the best-scoring chunk of each page. Expects chunks sorted by score, best first."""
    seen, out = set(), []
    for c in chunks:
        if c["url"] in seen:
            continue
        seen.add(c["url"])
        out.append(c)
    return out


def rank(claim: str, docs: list, top_k: int = None) -> list:
    top_k = top_k or config.TOP_K_CHUNKS
    chunks = []
    for d in docs:
        for c in chunk_text(d["content"]):
            chunks.append({"chunk": c, "url": d["url"], "title": d["title"],
                           "trusted": d.get("trusted", False), "source": d["source"]})
    if not chunks:
        return []
    model = _get_model()
    claim_vec = model.encode([claim], normalize_embeddings=True)[0]
    chunk_vecs = model.encode([c["chunk"] for c in chunks], normalize_embeddings=True)
    scores = chunk_vecs @ claim_vec
    for c, s in zip(chunks, scores):
        c["score"] = float(s)
        c["year"] = latest_year(c["title"] + " " + c["chunk"])
    chunks.sort(key=lambda c: c["score"], reverse=True)
    return one_per_url(chunks)[:top_k]
