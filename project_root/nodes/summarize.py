from typing import Dict, Any

from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_models import ChatOpenAI


LLM = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def summarize(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary in the requested style."""
    docs = state.get("chunks")
    if not docs:
        state["error"] = "No documents to summarize"
        return state

    summary_type = state.get("summary_type", "tldr")
    prompt_map = {
        "tldr": "한국어로 3줄 요약해 주세요 (250~300자).",
        "section": "각 섹션별로 한 문장씩 한국어 bullet을 생성하세요.",
        "deep": "한국어로 700~900자 심층 요약을 작성하세요.",
    }

    chain = load_summarize_chain(LLM, chain_type="map_reduce")
    try:
        summary = chain.run({"input_documents": docs, "question": prompt_map.get(summary_type, prompt_map["tldr"])})
        state["summary_ko"] = summary
    except Exception as exc:
        state["error"] = str(exc)
    return state
