# Marathi Live Fact-Checker (work in progress)
**Status-** under active development. Results are from a small pilot and will change.

Code-mixed Marathi query rewriting + live retrieval-augmented fact-checking. No training dataset:
evidence is fetched live from the web for every claim.

```
input (text / link / image)
  -> normalizer.py   detect script, Roman -> Devanagari, keywords, bilingual queries
  -> retriever.py    Google Fact Check API + web search + page fetch (live)
  -> ranker.py       chunk + LaBSE similarity -> top evidence
  -> verifier.py     entailment: Supported / Refuted / Not enough evidence
  -> app.py          Streamlit UI showing every step
```

## Setup from scratch

1. Install Python 3.10 or newer from python.org (tick "Add to PATH" on Windows).
2. Open a terminal in this folder and create a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. Install packages (large: torch and the models; use good internet):
   `pip install -r requirements.txt`
4. Optional keys (see `.env.example`). Windows: `set ANTHROPIC_API_KEY=...`
   Mac/Linux: `export ANTHROPIC_API_KEY=...`
5. Test the normalizer (works with no keys, no internet):
   `python -m unittest discover tests`
   `python normalizer.py "Aata bagha kay zala political news madhe"`
6. Test the full pipeline from the terminal:
   `python pipeline.py "पुण्यात पूर आला आहे"`
7. Run the app: `streamlit run app.py`

The first full run downloads LaBSE (~1.8 GB) and the NLI model (~1 GB). This is normal.

## Evaluation (this is your paper's proof)

1. Put 100-200 real claims in `evaluation/test_claims.csv`, each written 3 ways
   (devanagari, roman, mixed) with a label taken from a real published fact-check.
2. `python evaluation/eval.py evaluation/test_claims.csv`
3. It prints retrieval hit rate and accuracy for baseline RAG vs RAG + your rewriting
   layer, per input form. Your result is the gap on roman and mixed input.

## Roadmap

- Better transliteration: `pip install ai4bharat-transliteration` (normalizer uses it automatically).
- Grow `ROMAN_LEXICON` and `EN_TO_MR` in `normalizer.py` from real social media posts.
- Add-on A: text-image inconsistency (image caption + entity comparison). OCR input already works.
- Add-on B: explainability. The UI already shows normalization steps and evidence sentences.
- Video input (Whisper speech-to-text) is deferred.

## Known limits

- Web search results vary by day, so run the evaluation several times and report the spread.
- Source domains in `config.TRUSTED_DOMAINS` are a starting list: verify and edit them.
- Scraping: check each site's terms before heavy use.
