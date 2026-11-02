"""Central settings. Secrets come from environment variables, never hardcode them."""
import os

GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")

EMBED_MODEL = "sentence-transformers/LaBSE"
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# Sources ranked higher when found. Check these domains yourself and edit the list.
TRUSTED_DOMAINS = [
    "factcrescendo.com", "bbc.com", "loksatta.com", "lokmat.com",
    "maharashtratimes.com", "esakal.com", "boomlive.in", "factly.in",
    "vishvasnews.com", "newschecker.in", "altnews.in", "pib.gov.in",
]

# Social media and video pages are not news articles, so they are skipped as evidence.
BLOCKED_DOMAINS = [
    "instagram.com", "facebook.com", "fb.watch", "youtube.com", "youtu.be",
    "tiktok.com", "x.com", "twitter.com", "pinterest.com", "threads.net",
]

MIN_SIMILARITY = 0.30   # passages below this LaBSE score are dropped as garbage; the NLI model judges the rest
NLI_THRESHOLD = 0.60    # entailment / contradiction must beat this to give Supported / Refuted
# Both numbers are first guesses: tune them on your test claims.

# Fact-check cue rule (used when no LLM key is set). A page counts as a fact-check if its
# domain is in this list or its text says "fact check"; if such a page is close enough to the
# claim AND contains a debunking word ("false", "old video", "खोटा" ...), the verdict is Refuted.
FACTCHECK_DOMAINS = [
    "factcrescendo.com", "boomlive.in", "factly.in", "vishvasnews.com", "newschecker.in",
    "altnews.in", "thequint.com", "factchecker.in", "afwa.in", "digitaleye.in",
]
FACTCHECK_MIN_SIMILARITY = 0.40

MAX_PAGES = 6        # pages fetched per claim
TOP_K_CHUNKS = 5     # evidence chunks passed to the verifier
REQUEST_TIMEOUT = 8
