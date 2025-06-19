from pathlib import Path
from typing import Dict, Any

from langchain_community.document_loaders import PyPDFLoader


def load(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load a PDF file and store raw documents in state.

    Expects ``state['pdf_path']`` to contain a path to the PDF file.
    Stores the loaded ``Document`` list in ``state['chunks_raw']``.
    """
    pdf_path = Path(state.get("pdf_path", ""))
    if not pdf_path.is_file():
        state["error"] = f"PDF not found: {pdf_path}"
        return state

    loader = PyPDFLoader(str(pdf_path))
    try:
        docs = loader.load()
        state["chunks_raw"] = docs
    except Exception as exc:
        state["error"] = str(exc)
    return state
