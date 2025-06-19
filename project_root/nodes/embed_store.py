from pathlib import Path
from typing import Dict, Any

from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)


def embed(state: Dict[str, Any]) -> Dict[str, Any]:
    """Embed chunks and store in a Chroma vector DB."""
    docs = state.get("chunks")
    if not docs:
        state["error"] = "No chunks to embed"
        return state

    embedding = OpenAIEmbeddings()
    try:
        vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=str(DATA_DIR))
        vectordb.persist()
        state["vector"] = vectordb
    except Exception as exc:
        state["error"] = str(exc)
    return state
