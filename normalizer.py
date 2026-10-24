"""
Module 1: Code-mixed Marathi normalization + query rewriting (your core contribution).

raw text -> detect script -> Roman->Devanagari (lexicon + IndicXlit)
         -> keywords -> bilingual search queries
"""
import re

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")

# Common Roman Marathi words. Extend this list as you test real social media text.
ROMAN_LEXICON = {
    "aata": "आता", "ata": "आता", "bagha": "बघा", "pahaa": "पहा", "paha": "पहा",
    "kay": "काय", "kai": "काय", "kaay": "काय", "zala": "झाला", "jhala": "झाला",
    "zali": "झाली", "jhali": "झाली", "aahe": "आहे", "ahe": "आहे", "aahet": "आहेत",
    "nahi": "नाही", "nahi.": "नाही", "he": "हे", "ha": "हा", "hi": "ही",
    "madhe": "मध्ये", "var": "वर", "kara": "करा", "chalu": "चालू",
    "bolla": "बोलला", "neta": "नेता", "batmi": "बातमी", "khar": "खरं",
    "khota": "खोटा", "khoti": "खोटी", "sarkar": "सरकार", "mumbai": "मुंबई",
    "pune": "पुणे", "nagpur": "नागपूर", "nashik": "नाशिक", "maharashtra": "महाराष्ट्र",
    "paus": "पाऊस", "pur": "पूर", "ani": "आणि", "pan": "पण", "kon": "कोण",
    "kuthe": "कुठे", "kadhi": "कधी", "mantri": "मंत्री", "nivadnuk": "निवडणूक",
    "police": "पोलीस", "aala": "आला", "aali": "आली", "gela": "गेला",
}

# English words common in Marathi posts -> Marathi search terms.
EN_TO_MR = {
    "political": "राजकीय", "politics": "राजकारण", "news": "बातमी",
    "flood": "पूर", "rain": "पाऊस", "election": "निवडणूक", "minister": "मंत्री",
    "police": "पोलीस", "government": "सरकार", "viral": "व्हायरल",
    "video": "व्हिडिओ", "photo": "फोटो", "accident": "अपघात", "fire": "आग",
    "arrest": "अटक", "scam": "घोटाळा", "bridge": "पूल", "school": "शाळा",
    "exam": "परीक्षा", "result": "निकाल", "price": "भाव", "petrol": "पेट्रोल",
}

MARATHI_STOPWORDS = {
    "आता", "बघा", "पहा", "काय", "झाला", "झाली", "झाले", "आहे", "आहेत", "नाही",
    "हे", "हा", "ही", "मध्ये", "वर", "करा", "चालू", "आणि", "पण", "की",
    "तो", "ती", "ते", "या", "त्या", "एक", "लिंक", "क्लिक", "शेअर",
    "खोटा", "खोटी", "खोटे", "खरं", "खरी",
}
ENGLISH_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "of",
    "to", "and", "or", "this", "that", "link", "click", "share", "fully",
    "fake", "true", "real",
}


def detect_script(text: str) -> str:
    """Return 'devanagari', 'roman', 'mixed' or 'other'."""
    has_dev = bool(DEVANAGARI.search(text))
    has_lat = bool(LATIN.search(text))
    if has_dev and has_lat:
        return "mixed"
    if has_dev:
        return "devanagari"
    if has_lat:
        return "roman"
    return "other"


_xlit_engine = None


def _get_xlit():
    """Load IndicXlit lazily. Returns None if it is not installed."""
    global _xlit_engine
    if _xlit_engine is None:
        try:
            from ai4bharat.transliteration import XlitEngine
            _xlit_engine = XlitEngine("mr", beam_width=4, src_script_type="en")
        except Exception:
            _xlit_engine = False
    return _xlit_engine or None


def transliterate_word(word: str) -> str:
    """Roman word -> Devanagari. Lexicon, then English map, then IndicXlit, else unchanged."""
    key = word.lower().strip(".,!?;:\"'()")
    if key in ROMAN_LEXICON:
        return ROMAN_LEXICON[key]
    if key in EN_TO_MR:
        return word  # keep English; the query builder adds the Marathi term
    engine = _get_xlit()
    if engine:
        try:
            return engine.translit_word(key, topk=1)["mr"][0]
        except Exception:
            pass
    return word


def normalize(text: str) -> dict:
    script = detect_script(text)
    tokens = re.findall(r"[ऀ-ॿA-Za-z0-9]+|[^\s\w]", text)
    out = []
    for tok in tokens:
        if tok[0].isalpha() and LATIN.match(tok[0]):
            out.append(transliterate_word(tok))
        else:
            out.append(tok)
    normalized = " ".join(out)
    normalized = re.sub(r"\s+([,.!?;:])", r"\1", normalized)
    return {"original": text, "script": script, "normalized": normalized}


def extract_keywords(normalized: str, max_words: int = 8) -> list:
    words = re.findall(r"[ऀ-ॿA-Za-z0-9]+", normalized)
    keep = []
    for w in words:
        if w in MARATHI_STOPWORDS or w.lower() in ENGLISH_STOPWORDS:
            continue
        if len(w) < 2 or w in keep:
            continue
        keep.append(w)
    return keep[:max_words]


def build_queries(normalized: str) -> list:
    """Bilingual queries: Marathi-only, Marathi + fact-check term, English-mixed, full claim."""
    kws = extract_keywords(normalized)
    if not kws:
        return [normalized]
    mr_words = list(dict.fromkeys(EN_TO_MR.get(w.lower(), w) for w in kws))
    mr = " ".join(mr_words)
    mixed = " ".join(kws)
    queries = [mr, f"{mr} फॅक्ट चेक", f"{mixed} fact check", normalized[:120]]
    return list(dict.fromkeys(q for q in queries if q.strip()))


def process(text: str) -> dict:
    result = normalize(text)
    result["keywords"] = extract_keywords(result["normalized"])
    result["queries"] = build_queries(result["normalized"])
    return result


if __name__ == "__main__":
    import json
    import sys
    sample = " ".join(sys.argv[1:]) or "He bagha neta kai bolla, fully fake news aahe"
    print(json.dumps(process(sample), ensure_ascii=False, indent=2))
