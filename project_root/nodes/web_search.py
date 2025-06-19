from typing import Dict, Any, List

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper


_searcher = TavilySearchAPIWrapper()


def web_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """Perform web search for given keywords and store references."""
    if not state.get("ext_flag"):
        return state

    keywords = state.get("keywords") or []
    if not keywords:
        return state

    refs: List[dict] = []
    try:
        for kw in keywords[:3]:
            results = _searcher.results(kw, 3)
            for r in results:
                refs.append({"title": r["title"], "url": r["url"]})
        state["external_refs"] = refs
    except Exception as exc:
        state["error"] = str(exc)
    return state
