from typing import Dict, Any

from keybert import KeyBERT
import spacy


_nlp = spacy.load("en_core_web_sm")
_kw_model = KeyBERT()


def highlight(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract keywords and highlight them in the summary."""
    summary = state.get("summary_ko")
    if not summary:
        state["error"] = "No summary available to highlight"
        return state

    try:
        keywords = [kw for kw, _ in _kw_model.extract_keywords(summary, top_n=5, use_mmr=True)]
        for kw in keywords:
            summary = summary.replace(kw, f"**{kw}**")
        state["summary_ko"] = summary
        state["keywords"] = keywords
    except Exception as exc:
        state["error"] = str(exc)
    return state
