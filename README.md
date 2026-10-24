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
1. Install Python 3.12 from python.org (tick "Add to PATH" on Windows).
     Python   3.14 may not work with the ML libraries yet.
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

The test set is 9 claims from published fact-checks, each written 3 ways
(Devanagari, Roman, mixed), so 27 inputs. All claims in this pilot are false claims.

Run it: python evaluation/eval.py evaluation/test_claims.csv

It compares plain retrieval (baseline) with retrieval + the code-mixed rewriting layer,
per input form. Results are in evaluation/results_summary.csv.

Pilot findings: the rewriting layer improved evidence retrieval for mixed-language input
and verdict accuracy for Roman input. The verdict step is still the weakest part: it can
mark a false claim "Supported" when a fact-check article quotes the claim. The sample is
small (n=9 per cell), so treat these numbers as indicative only.

## Roadmap

- Better transliteration: `pip install ai4bharat-transliteration` (normalizer uses it automatically).
- Grow `ROMAN_LEXICON` and `EN_TO_MR` in `normalizer.py` from real social media posts.
- Add-on A: text-image inconsistency (image caption + entity comparison). OCR input already works.
- Add-on B: explainability. The UI already shows normalization steps and evidence sentences.
- Video input (Whisper speech-to-text) is deferred.
- * Improve the verdict step (fact-check cue words or an LLM verifier).

## Known limits

- Web search results vary by day, so run the evaluation several times and report the spread.
- Source domains in `config.TRUSTED_DOMAINS` are a starting list: verify and edit them.
- Scraping: check each site's terms before heavy use.
