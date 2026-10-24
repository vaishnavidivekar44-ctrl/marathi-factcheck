"""
Module 2: Live retrieval. No static dataset: everything is fetched at query time.

Sources:
  1. Google Fact Check Tools API (needs a free API key)  - existing fact-checks
  2. Web search (DuckDuckGo via `ddgs`, no key needed)  - news articles
  3. Direct page fetch                                  - full article text
"""
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

import config

HEADERS = {"User-Agent": "Mozilla/5.0 (academic research project; Marathi fact-checking)"}


def is_trusted(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in config.TRUSTED_DOMAINS)


def is_blocked(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in config.BLOCKED_DOMAINS)


def search_factcheck_api(query: str, lang: str = "mr", n: int = 5) -> list:
    if not config.GOOGLE_FACTCHECK_API_KEY:
        return []
    try:
        r = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": query, "languageCode": lang, "pageSize": n,
                    "key": config.GOOGLE_FACTCHECK_API_KEY},
            timeout=config.REQUEST_TIMEOUT,
        )
        r.raise_for_status()
    except Exception:
        return []
    docs = []
    for claim in r.json().get("claims", []):
        for rev in claim.get("claimReview", []):
            docs.append({
                "title": rev.get("title") or claim.get("text", ""),
                "url": rev.get("url", ""),
                "content": f'Claim: {claim.get("text", "")}. Rating: {rev.get("textualRating", "")}.',
                "source": "factcheck_api",
                "publisher": rev.get("publisher", {}).get("name", ""),
            })
    return docs


def search_web(query: str, n: int = 6, attempts: int = 3) -> list:
    """Web search with retries: a dropped connection or rate limit waits and tries again."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return []
    results = []
    for attempt in range(attempts):
        try:
            with DDGS() as d:
                results = list(d.text(query, region="in-en", max_results=n))
            break
        except Exception:
            if attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
    return [{"title": r.get("title", ""), "url": r.get("href", ""),
             "content": r.get("body", ""), "source": "web"} for r in results]


def fetch_page(url: str, max_chars: int = 6000) -> dict:
    """Return {'title', 'text'} for a URL. Empty strings on failure."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=config.REQUEST_TIMEOUT)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or r.encoding
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        title = (soup.find("h1") or soup.title)
        title = title.get_text(strip=True) if title else ""
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(p for p in paras if len(p) > 40)
        return {"title": title, "text": text[:max_chars]}
    except Exception:
        return {"title": "", "text": ""}


def retrieve(queries: list, max_pages: int = None) -> list:
    """Run all queries, dedupe by URL, put trusted sources first, fetch full text."""
    max_pages = max_pages or config.MAX_PAGES
    seen, docs = set(), []
    for q in queries:
        for d in search_factcheck_api(q) + search_web(q):
            if d["url"] and d["url"] not in seen and not is_blocked(d["url"]):
                seen.add(d["url"])
                d["trusted"] = is_trusted(d["url"])
                docs.append(d)
    docs.sort(key=lambda d: (d["source"] != "factcheck_api", not d["trusted"]))
    docs = docs[:max_pages]
    for d in docs:
        if d["source"] == "web":
            page = fetch_page(d["url"])
            if len(page["text"]) > len(d["content"]):
                d["content"] = page["text"]
    return docs
