from typing import Dict, Any

from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI


LLM = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve relevant documents from vector store."""
    vectordb = state.get("vector")
    query = state.get("user_query")
    if not vectordb or not query:
        state["error"] = "Missing vector store or query"
        return state

    try:
        docs = vectordb.similarity_search(query, k=4)
        state["retrieved_docs"] = docs
    except Exception as exc:
        state["error"] = str(exc)
    return state


def answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Answer user question based on retrieved docs."""
    docs = state.get("retrieved_docs")
    query = state.get("user_query")
    vectordb = state.get("vector")
    if not docs or not query or not vectordb:
        state["error"] = "No documents or query for answering"
        return state

    try:
        qa_chain = RetrievalQA.from_chain_type(llm=LLM, retriever=vectordb.as_retriever())
        result = qa_chain.run(query)
        state["answer_ko"] = result
    except Exception as exc:
        state["error"] = str(exc)
    return state
