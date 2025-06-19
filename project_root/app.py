import streamlit as st
from pathlib import Path

from graph_build import exec_graph

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="논문 요약·Q&A", layout="wide")

st.title("시민 친화형 논문 요약 + Q&A 챗봇")

uploaded = st.file_uploader("논문 PDF 업로드", type=["pdf"])
ext_flag = st.checkbox("외부 웹 정보도 포함")
lang_option = st.radio("언어", ["한국어", "영어", "한국어+영어"], index=0)
sum_style = st.selectbox("요약 형식", ["TL;DR", "섹션별", "Deep-Dive"])

if uploaded:
    pdf_path = DATA_DIR / uploaded.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded.getbuffer())

    if st.button("요약 생성"):
        result = exec_graph(
            pdf_path=str(pdf_path),
            summary_type=sum_style.lower(),
            ext_flag=ext_flag,
            lang_option=lang_option,
        )
        st.markdown(result.get("summary_ko", "요약 실패"))
        if "영어" in lang_option:
            st.markdown("---")
            st.markdown(result.get("summary_en", "English summary failed"))

    st.header("질문하기")
    user_q = st.text_input("질문")
    if st.button("Ask") and user_q:
        ans_state = exec_graph(
            pdf_path=str(pdf_path),
            user_query=user_q,
            ext_flag=ext_flag,
            lang_option=lang_option,
        )
        st.markdown(ans_state.get("answer_ko", "답변 실패"))
        if "영어" in lang_option:
            st.markdown("---")
            st.markdown(ans_state.get("answer_en", "English answer failed"))
