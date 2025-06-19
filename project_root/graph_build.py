from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langchain.schema import Document
from langgraph.graph import StateGraph

from nodes.load_pdf import load
from nodes.split import split
from nodes.embed_store import embed
from nodes.summarize import summarize
from nodes.keyword_highlight import highlight
from nodes.web_search import web_search
from nodes.retrieve_answer import retrieve, answer
from nodes.error_handler import handle


class PaperState(TypedDict, total=False):
    pdf_path: str
    metadata: Dict[str, Any]
    chunks_raw: List[Document]
    chunks: List[Document]
    vector: Any
    summary_type: str
    summary_ko: str
    summary_en: str
    user_query: str
    answer_ko: str
    answer_en: str
    external_refs: List[Dict[str, str]]
    keywords: List[str]
    retrieved_docs: List[Document]
    error: str


def route(state: Dict[str, Any]) -> str:
    """Decide next step based on state."""
    if state.get("error"):
        return "error"
    if state.get("user_query"):
        return "question"
    return "summary"


def build_graph() -> StateGraph:
    """Construct the processing StateGraph."""
    graph = StateGraph(PaperState)

    graph.add_node("LoadPDF", load)
    graph.add_node("Split", split)
    graph.add_node("EmbedStore", embed)
    graph.add_node("DecisionRouter", lambda x: x)
    graph.add_node("Summarize", summarize)
    graph.add_node("Highlight", highlight)
    graph.add_node("WebSearch", web_search)
    graph.add_node("Retrieve", retrieve)
    graph.add_node("Answer", answer)
    graph.add_node("ErrorHandler", handle)

    graph.set_entry_point("LoadPDF")
    graph.add_edge("LoadPDF", "Split")
    graph.add_edge("Split", "EmbedStore")
    graph.add_edge("EmbedStore", "DecisionRouter")

    graph.add_conditional_edges(
        "DecisionRouter",
        route,
        {
            "summary": "Summarize",
            "question": "Retrieve",
            "error": "ErrorHandler",
        },
    )

    graph.add_edge("Summarize", "Highlight")
    graph.add_edge("Highlight", "WebSearch")
    graph.add_edge("WebSearch", "ErrorHandler")

    graph.add_edge("Retrieve", "Answer")
    graph.add_edge("Answer", "ErrorHandler")

    graph.set_finish_point("ErrorHandler")

    return graph


def exec_graph(**kwargs: Any) -> PaperState:
    """Run the compiled graph with provided keyword state."""
    graph = build_graph().compile()
    state: PaperState = {}
    state.update(kwargs)
    return graph.invoke(state)
