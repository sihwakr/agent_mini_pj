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
    lang_option = state.get("lang_option", "한국어")
    prompt_map_ko = {
        "tldr": "한국어로 3줄 요약해 주세요 (250~300자).",
        "section": "각 섹션별로 한 문장씩 한국어 bullet을 생성하세요.",
        "deep": "한국어로 700~900자 심층 요약을 작성하세요.",
    }
    prompt_map_en = {
        "tldr": "Provide a three-sentence TL;DR in English.",
        "section": "Summarize each section in one English sentence bullet.",
        "deep": "Write a detailed English summary in 5-7 sentences.",
    }

    chain = load_summarize_chain(LLM, chain_type="map_reduce")
    try:
        summary_ko = chain.run({
            "input_documents": docs,
            "question": prompt_map_ko.get(summary_type, prompt_map_ko["tldr"]),
        })
        state["summary_ko"] = summary_ko

        if "영어" in lang_option:
            summary_en = chain.run({
                "input_documents": docs,
                "question": prompt_map_en.get(summary_type, prompt_map_en["tldr"]),
            })
            state["summary_en"] = summary_en
    except Exception as exc:
        state["error"] = str(exc)
    return state
