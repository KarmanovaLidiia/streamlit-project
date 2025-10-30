import streamlit as st
from pathlib import Path
import pandas as pd
import time
from src.predict import pipeline_infer

st.set_page_config(page_title="Автоматическая оценка ответа иностранного гражданина", layout="centered")

st.title("Автоматическая оценка ответа иностранного гражданина")
st.caption("Загрузите CSV как в задании. Получите файл с колонкой pred_score.")

uploaded = st.file_uploader("Загрузите CSV", type=["csv"])
run = st.button("Оценить")

if uploaded and run:
    tmp_in = Path("data/raw/tmp_input.csv")
    tmp_out = Path("data/processed/tmp_output.csv")

    content = uploaded.read()
    tmp_in.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_in, "wb") as f:
        f.write(content)

    start_time = time.time()
    st.info("🔹 Началась обработка файла...")

    # Визуальный прогресс-бар (фиксированный — для ориентира)
    progress = st.progress(0, text="⏳ Подготовка данных...")

    for percent_complete in range(1, 6):
        time.sleep(0.2)  # имитация активности (убрать при желании)
        progress.progress(percent_complete * 5, text=f"🔄 Подготовка ({percent_complete * 5}%)")

    pipeline_infer(tmp_in, tmp_out)

    progress.progress(100, text="✅ Обработка завершена")

    duration = time.time() - start_time
    if duration > 120:
        st.warning(f"⚠️ Предсказание заняло {int(duration)} секунд. Возможно, файл большой или система перегружена.")

    st.success("Готово!")
    df = pd.read_csv(tmp_out, encoding="utf-8-sig", sep=None, engine="python")
    st.dataframe(df.head(20))
    st.download_button("⬇️ Скачать результат", data=tmp_out.read_bytes(),
                       file_name="predicted.csv", mime="text/csv")
