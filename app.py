"""Streamlit UI. Run with:  streamlit run app.py"""
from datetime import date

import streamlit as st

import pipeline

st.set_page_config(page_title="Marathi Live Fact-Checker", layout="wide")
st.title("Marathi Live Fact-Checker")
st.caption("Code-mixed Marathi query rewriting + live retrieval-augmented verification")

mode = st.sidebar.radio("Input", ["Text / Link", "Image (OCR)"])
use_rewriting = st.sidebar.checkbox("Use code-mixed rewriting layer", value=True)

text = ""
if mode == "Text / Link":
    text = st.text_area("Paste a claim, social media post, or news link "
                        "(Devanagari, Roman, or mixed):", height=150)
else:
    img = st.file_uploader("Upload a news screenshot or meme", type=["jpg", "jpeg", "png"])
    if img:
        st.image(img, use_container_width=True)
        try:
            import easyocr
            reader = easyocr.Reader(["mr", "en"], gpu=False)
            lines = reader.readtext(img.getvalue(), detail=0)
            text = st.text_area("Text found in the image (edit if needed):", " ".join(lines))
        except Exception as exc:
            st.error(f"OCR failed or easyocr is not installed: {exc}")

if st.button("Verify") and text.strip():
    with st.spinner("Searching live sources..."):
        out = pipeline.run(text, use_rewriting=use_rewriting)
    if "error" in out:
        st.error(out["error"])
        st.stop()

    res = out["result"]
    colour = {"Supported": "green", "Refuted": "red"}.get(res["verdict"], "orange")
    st.markdown(f"## Verdict: :{colour}[{res['verdict']}]  (confidence: {res['confidence']})")
    if res.get("explanation_marathi"):
        st.write(res["explanation_marathi"])
    if res.get("explanation_english"):
        st.write(res["explanation_english"])

    with st.expander("How the system understood your input", expanded=True):
        st.json(out["steps"])

    st.subheader("Evidence from live sources")
    for i, e in enumerate(out["evidence"]):
        star = " (used for verdict)" if i in res.get("key_evidence", []) else ""
        badge = "trusted" if e["trusted"] else "other"
        old = ""
        if e.get("year") and e["year"] < date.today().year - 1:
            old = f" · WARNING: mentions {e['year']}, may be old"
        st.markdown(f"**[{i}] [{e['title'] or e['url']}]({e['url']})** "
                    f"· {badge} · similarity {e['score']:.2f}{old}{star}")
        st.write(e["chunk"])
        st.markdown("---")
