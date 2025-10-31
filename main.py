import streamlit as st
import sys
import subprocess
import os

# Попытка импорта transformers с обработкой ошибок
try:
    from transformers import pipeline, AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    st.error(f"Ошибка импорта transformers: {e}")
    st.info("Попытка установки transformers...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "tokenizers"])
        from transformers import pipeline, AutoModel, AutoTokenizer

        TRANSFORMERS_AVAILABLE = True
        st.success("Transformers успешно установлен!")
    except:
        TRANSFORMERS_AVAILABLE = False
        st.error("Не удалось установить transformers. Пожалуйста, установите вручную: pip install transformers torch")

# Импорт других библиотек
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Настройка страницы
st.set_page_config(
    page_title="Hugging Face Model Demo",
    page_icon="🤗",
    layout="wide"
)

# Заголовок приложения
st.title("🤗 Hugging Face Model in Streamlit")
st.write("Демонстрация работы моделей из Hugging Face Hub")

# Проверка доступности основных библиотек
if not TRANSFORMERS_AVAILABLE:
    st.error("""
    **Библиотека transformers недоступна!**

    Пожалуйста, установите её выполнив в терминале:
    ```
    pip install transformers torch tokenizers
    ```
    """)
    st.stop()

if not TORCH_AVAILABLE:
    st.warning("PyTorch недоступен. Некоторые функции могут не работать.")

# Сайдбар для выбора модели
st.sidebar.header("Настройки модели")

# Выбор типа задачи
task_type = st.sidebar.selectbox(
    "Выберите тип задачи:",
    ["text-classification", "sentiment-analysis", "text-generation", "translation", "question-answering"]
)

# Популярные модели для разных задач (используем легкие модели)
model_options = {
    "text-classification": "distilbert-base-uncased-finetuned-sst-2-english",
    "text-generation": "gpt2",
    "sentiment-analysis": "distilbert-base-uncased-finetuned-sst-2-english",
    "translation": "t5-small",
    "question-answering": "distilbert-base-cased-distilled-squad"
}

# Информация о моделях
st.sidebar.markdown("### Информация о моделях")
for task, model in model_options.items():
    st.sidebar.write(f"**{task}**: `{model}`")


# Кэшируем загрузку модели
@st.cache_resource(show_spinner=False)
def load_model(task, model_name):
    try:
        st.info(f"Загружаем модель: {model_name}")
        if task == "translation":
            return pipeline(task, model=model_name, tokenizer=model_name)
        else:
            return pipeline(task, model=model_name)
    except Exception as e:
        st.error(f"Ошибка загрузки модели {model_name}: {e}")
        return None


# Основной интерфейс
if task_type == "text-classification":
    st.header("📊 Классификация текста")
    st.write("Классифицируйте текст на различные категории")

    text_input = st.text_area(
        "Введите текст для классификации:",
        "I love this product! It's amazing!",
        height=100
    )

    if st.button("🎯 Классифицировать", type="primary"):
        with st.spinner("Загружаем модель классификации..."):
            classifier = load_model("text-classification", model_options["text-classification"])

        if classifier:
            try:
                results = classifier(text_input)
                st.subheader("📈 Результаты классификации:")

                # Создаем красивый вывод результатов
                for i, result in enumerate(results):
                    score_percent = result['score'] * 100
                    st.write(f"**{i + 1}. {result['label']}**")
                    st.progress(float(result['score']))
                    st.write(f"Уверенность: **{score_percent:.2f}%**")
                    st.markdown("---")
            except Exception as e:
                st.error(f"Ошибка при классификации: {e}")

elif task_type == "sentiment-analysis":
    st.header("😊 Анализ тональности")
    st.write("Определите эмоциональную окраску текста")

    text_input = st.text_area(
        "Введите текст для анализа тональности:",
        "I'm so happy today! Everything is going well.",
        height=100
    )

    if st.button("🔍 Проанализировать", type="primary"):
        with st.spinner("Загружаем модель анализа тональности..."):
            analyzer = load_model("sentiment-analysis", model_options["sentiment-analysis"])

        if analyzer:
            try:
                results = analyzer(text_input)
                st.subheader("📊 Результаты анализа:")

                for result in results:
                    score = result['score']
                    label = result['label']

                    # Визуализируем результат
                    if "POSITIVE" in label.upper():
                        st.success(f"😊 Положительный оттенок")
                    elif "NEGATIVE" in label.upper():
                        st.error(f"😞 Отрицательный оттенок")
                    else:
                        st.info(f"😐 {label}")

                    st.metric("Уверенность", f"{score:.4f}")

            except Exception as e:
                st.error(f"Ошибка при анализе тональности: {e}")

elif task_type == "text-generation":
    st.header("✍️ Генерация текста")
    st.write("Сгенерируйте текст на основе вашего промпта")

    prompt = st.text_area(
        "Введите промпт:",
        "The future of artificial intelligence",
        height=100
    )

    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Максимальная длина текста", 50, 300, 100)
    with col2:
        temperature = st.slider("Температура", 0.1, 1.0, 0.7)

    if st.button("🚀 Сгенерировать текст", type="primary"):
        with st.spinner("Генерируем текст..."):
            generator = load_model("text-generation", model_options["text-generation"])

        if generator:
            try:
                results = generator(
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True
                )
                st.subheader("📝 Сгенерированный текст:")
                st.write(results[0]['generated_text'])

                st.info("💡 **Подсказка:** Поэкспериментируйте с температурой для разных результатов!")

            except Exception as e:
                st.error(f"Ошибка при генерации текста: {e}")

elif task_type == "translation":
    st.header("🌐 Перевод текста")
    st.write("Переведите текст между языками")

    text_to_translate = st.text_area(
        "Введите текст для перевода:",
        "Hello, how are you? I hope you have a great day!",
        height=100
    )

    if st.button("🔄 Перевести", type="primary"):
        with st.spinner("Загружаем модель перевода..."):
            translator = load_model("translation", model_options["translation"])

        if translator:
            try:
                # Для T5 модели нужно добавить префикс
                if "t5" in model_options["translation"]:
                    text_to_translate = "translate English to French: " + text_to_translate

                result = translator(text_to_translate)
                st.subheader("📖 Перевод:")
                st.success(result[0]['translation_text'])

            except Exception as e:
                st.error(f"Ошибка при переводе: {e}")

elif task_type == "question-answering":
    st.header("❓ Вопрос-ответ")
    st.write("Получите ответы на вопросы на основе предоставленного контекста")

    context = st.text_area(
        "Контекст:",
        "The Eiffel Tower is located in Paris, France. It was built in 1889 and is one of the most famous landmarks in the world. The tower is 330 meters tall and was designed by Gustave Eiffel.",
        height=150
    )
    question = st.text_input("Вопрос:", "Where is the Eiffel Tower located?")

    if st.button("🔎 Найти ответ", type="primary"):
        with st.spinner("Ищем ответ в тексте..."):
            qa_pipeline = load_model("question-answering", model_options["question-answering"])

        if qa_pipeline:
            try:
                result = qa_pipeline(question=question, context=context)
                st.subheader("💡 Ответ:")
                st.success(f"**{result['answer']}**")

                # Дополнительная информация
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Уверенность", f"{result['score']:.4f}")
                with col2:
                    start_pos = result['start']
                    end_pos = result['end']
                    st.metric("Позиция в тексте", f"{start_pos}-{end_pos}")

            except Exception as e:
                st.error(f"Ошибка при поиске ответа: {e}")

# Информация в сайдбаре
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **💡 О приложении:**

    Это приложение демонстрирует интеграцию моделей 
    Hugging Face с Streamlit.

    Модели загружаются автоматически при первом 
    использовании и кэшируются для быстрого доступа.

    **⚠️ Примечание:** Первая загрузка модели может 
    занять несколько минут.
    """
)

# Статус библиотек в футере
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Статус библиотек")
st.sidebar.write(f"🤗 Transformers: {'✅ Доступна' if TRANSFORMERS_AVAILABLE else '❌ Недоступна'}")
st.sidebar.write(f"🔥 PyTorch: {'✅ Доступна' if TORCH_AVAILABLE else '❌ Недоступна'}")
st.sidebar.write(f"🐼 Pandas: {'✅ Доступна' if PANDAS_AVAILABLE else '⚠️ Недоступна'}")

# Инструкция по устранению неполадок
if not TRANSFORMERS_AVAILABLE:
    st.sidebar.markdown("---")
    st.sidebar.error("""
    **Устранение неполадок:**

    Выполните в терминале:
    ```bash
    pip install transformers torch tokenizers
    ```
    """)