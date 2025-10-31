# -*- coding: utf-8 -*-
import streamlit as st
from transformers import pipeline
import os

# Отключаем предупреждения
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Простая конфигурация
st.set_page_config(
    page_title="AI Model Demo",
    page_icon="🤖",
    layout="wide"
)

# Простой заголовок
st.title("🤖 Демо AI Моделей")
st.write("Тестирование моделей машинного обучения")

# Боковая панель
st.sidebar.header("Настройки")

# Выбор задачи
task = st.sidebar.selectbox(
    "Выберите задачу:",
    ["Анализ тональности", "Генерация текста", "Классификация"]
)

# Основной контент
if task == "Анализ тональности":
    st.header("📊 Анализ тональности текста")
    text = st.text_area("Введите текст:", "Я очень рад этому!")

    if st.button("Анализировать"):
        with st.spinner("Анализируем..."):
            try:
                classifier = pipeline("sentiment-analysis")
                result = classifier(text)[0]
                st.success(f"Результат: {result['label']}")
                st.info(f"Уверенность: {result['score']:.4f}")
            except Exception as e:
                st.error(f"Ошибка: {e}")

elif task == "Генерация текста":
    st.header("✍️ Генерация текста")
    prompt = st.text_area("Введите начало текста:", "Искусственный интеллект")

    if st.button("Сгенерировать"):
        with st.spinner("Генерируем..."):
            try:
                generator = pipeline("text-generation", model="gpt2")
                result = generator(prompt, max_length=100, num_return_sequences=1)
                st.write("**Результат:**")
                st.write(result[0]['generated_text'])
            except Exception as e:
                st.error(f"Ошибка: {e}")

elif task == "Классификация":
    st.header("🏷️ Классификация текста")
    text = st.text_area("Введите текст для классификации:", "Это потрясающий продукт!")

    if st.button("Классифицировать"):
        with st.spinner("Классифицируем..."):
            try:
                classifier = pipeline("text-classification")
                results = classifier(text)
                st.write("**Результаты:**")
                for result in results:
                    st.write(f"- {result['label']}: {result['score']:.4f}")
            except Exception as e:
                st.error(f"Ошибка: {e}")

# Информация внизу
st.sidebar.markdown("---")
st.sidebar.info("Простое демо для тестирования моделей")