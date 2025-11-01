# app.py
import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# быстрый режим по умолчанию
os.environ.setdefault("FAST_MODE", "1")

# импорт основного пайплайна
from src.predict import pipeline_infer

# --- Конфигурация страницы ---
st.set_page_config(page_title="Русский как иностранный – автооценка", layout="centered")

st.title("Автооценка устных ответов (RFL • CatBoost + ruSBERT)")
st.caption("Загрузите CSV входного формата и получите файл с колонками pred_score и pred_score_rounded.")

# --- Информация о формате ---
with st.expander("Формат входного CSV", expanded=False):
    st.markdown(
        """
        Обязательные столбцы:
        - **№ вопроса** (1..4)
        - **Текст вопроса**
        - **Транскрибация ответа**
        - *(опционально)* **Оценка экзаменатора** — если есть, её не трогаем, добавим предсказания рядом.  

        Разделитель — `;`, кодировка — UTF-8 (автоопределяется).
        """
    )

# --- Пример шаблона CSV ---
with st.expander("📄 Скачать шаблон CSV"):
    demo = pd.DataFrame({
        "№ вопроса": [1, 2],
        "Текст вопроса": ["<p>Добро пожаловать...</p>", "<p>Опишите свой день...</p>"],
        "Транскрибация ответа": ["Здравствуйте! Я приехал...", "Мой день начинается с..."],
        "Оценка экзаменатора": [None, None],
    })
    st.dataframe(demo)
    buf_tmpl = io.BytesIO()
    demo.to_csv(buf_tmpl, index=False, sep=";", encoding="utf-8-sig")
    st.download_button("⬇ Скачать шаблон CSV", buf_tmpl.getvalue(), "template.csv", "text/csv")

# --- Функция загрузки и нормализации ---
required = ["№ вопроса", "Текст вопроса", "Транскрибация ответа"]
aliases = {
    "номер вопроса": "№ вопроса",
    "вопрос": "Текст вопроса",
    "текст задания": "Текст вопроса",
    "транскрибация": "Транскрибация ответа",
    "транскрипт": "Транскрибация ответа",
    "ответ": "Транскрибация ответа",
}


def load_and_normalize_csv(raw_bytes: bytes) -> pd.DataFrame:
    import io
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, engine="python")

            # убрать возможные артефакты Git-конфликтов
            if not df.empty and str(df.columns[0]).startswith("<<<"):
                text = raw_bytes.decode("utf-8", errors="ignore")
                lines = [ln for ln in text.splitlines() if not ln.startswith(("<<<", "===", ">>>"))]
                df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, engine="python")

            # нормализация имён колонок
            rename_map = {}
            for c in list(df.columns):
                key = str(c).strip().lower()
                if key in aliases:
                    rename_map[c] = aliases[key]
            if rename_map:
                df = df.rename(columns=rename_map)

            return df
        except Exception:
            continue
    raise ValueError("Не удалось прочитать CSV. Проверьте разделитель (';' или ',') и кодировку UTF-8.")


# --- Основной интерфейс ---
uploaded = st.file_uploader("Загрузите CSV", type=["csv"])
slow = st.toggle("Медленный режим", value=False, help="Выключите для быстрой оценки (точность ≈ прежняя).")
run = st.button("Посчитать")

if uploaded and run:
    try:
        raw = uploaded.read()
        df_in = load_and_normalize_csv(raw)

        # проверка обязательных колонок
        missing = [c for c in required if c not in df_in.columns]
        if missing:
            st.error(f"❌ В файле нет обязательных колонок: {missing}. Проверь заголовки и разделитель ';'.")
            st.dataframe(df_in.head())
            st.stop()

        # сохраняем временно
        tmp_in = Path("data/api_tmp/tmp_input.csv")
        tmp_in.parent.mkdir(parents=True, exist_ok=True)
        df_in.to_csv(tmp_in, index=False, sep=";", encoding="utf-8-sig")

        # режим скорости
        os.environ["FAST_MODE"] = "0" if slow else "1"

        tmp_out = Path("data/api_tmp/tmp_output.csv")
        with st.spinner("Считаем..."):
            pipeline_infer(tmp_in, tmp_out)

        df_out = pd.read_csv(tmp_out, sep=";", encoding="utf-8-sig")
        st.success("✅ Готово!")
        st.dataframe(df_out.head(20), use_container_width=True)

        buf = io.BytesIO()
        df_out.to_csv(buf, index=False, sep=";", encoding="utf-8-sig")
        st.download_button("⬇ Скачать результат (CSV)", data=buf.getvalue(), file_name="predicted.csv", mime="text/csv")
    except Exception as e:
        st.exception(e)

# --- Подвал ---
st.markdown("---")
st.caption("Модель: CatBoost Q1..Q4 + ruSBERT. Быстрый режим = FAST_MODE=1.")
