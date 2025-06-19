from typing import Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter


def split(state: Dict[str, Any]) -> Dict[str, Any]:
    """Split loaded documents into chunks under 1000 tokens."""
    raw_docs = state.get("chunks_raw")
    if not raw_docs:
        state["error"] = "No documents to split"
        return state

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    try:
        chunks = splitter.split_documents(raw_docs)
        state["chunks"] = chunks
    except Exception as exc:
        state["error"] = str(exc)
    return state
