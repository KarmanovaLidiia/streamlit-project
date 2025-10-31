# app.py  (Streamlit UI для HF Spaces)
import streamlit as st
import pandas as pd
from io import BytesIO

from assessment_engine import run_inference_df  # твой фасад

st.set_page_config(page_title="Автооценка ответов", layout="centered")
st.title("Автоматическая оценка экзаменационных ответов")

uploaded = st.file_uploader("Загрузите CSV с вопросами/ответами", type=["csv"])
with_expl = st.checkbox("Добавлять объяснения", value=True)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Первые строки входных данных:")
    st.dataframe(df.head())

    if st.button("Оценить"):
        with st.spinner("Считаю..."):
            out = run_inference_df(df, with_explanations=with_expl)
        st.success("Готово!")
        st.dataframe(out.head())

        buf = BytesIO()
        out.to_csv(buf, index=False)
        st.download_button("Скачать файл с оценками", buf.getvalue(), file_name="predicted.csv", mime="text/csv")
