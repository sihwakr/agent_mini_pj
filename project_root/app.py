import streamlit as st
from pathlib import Path

from graph_build import exec_graph

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="논문 요약·Q&A", layout="wide")

st.title("시민 친화형 논문 요약 + Q&A 챗봇")

uploaded = st.file_uploader("논문 PDF 업로드", type=["pdf"])
sum_style = st.selectbox("요약 형식", ["TL;DR", "섹션별", "Deep-Dive"])

if uploaded:
    pdf_path = DATA_DIR / uploaded.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded.getbuffer())

    if st.button("요약 생성"):
        result = exec_graph(pdf_path=str(pdf_path), summary_type=sum_style.lower())
        st.markdown(result.get("summary_ko", "요약 실패"))

    st.header("질문하기")
    user_q = st.text_input("질문")
    if st.button("Ask") and user_q:
        ans_state = exec_graph(pdf_path=str(pdf_path), user_query=user_q)
        st.markdown(ans_state.get("answer_ko", "답변 실패"))
