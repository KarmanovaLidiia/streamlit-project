import streamlit as st
import subprocess
import sys


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    from transformers import pipeline
except ImportError:
    st.warning("Устанавливаем transformers...")
    install_package("transformers")
    from transformers import pipeline

st.title("Минимальное приложение с Hugging Face")


# Простая модель для теста
@st.cache_resource
def load_model():
    try:
        return pipeline("sentiment-analysis")
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None


model = load_model()

if model:
    text = st.text_input("Введите текст:", "I love this!")

    if st.button("Анализировать") and text:
        result = model(text)[0]
        st.write(f"Результат: {result['label']}")
        st.write(f"Уверенность: {result['score']:.4f}")
else:
    st.error("Не удалось загрузить модель")