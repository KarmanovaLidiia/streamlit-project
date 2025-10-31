import streamlit as st
import sys
import subprocess

# Попытка импорта с обработкой ошибок
try:
    from transformers import pipeline

    st.success("✅ Transformers imported successfully!")
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.info("Installing transformers...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
        from transformers import pipeline

        st.success("✅ Transformers installed and imported!")
    except:
        st.error("Failed to install transformers")
        st.stop()

st.title("🧠 AI Models Working Demo")
st.write("This version should work reliably")

# Простой и надежный интерфейс
task = st.selectbox("Choose task:", ["Sentiment Analysis", "Text Generation"])

text = st.text_area("Enter text:", "I love artificial intelligence!")

if st.button("Run"):
    try:
        if task == "Sentiment Analysis":
            with st.spinner("Analyzing..."):
                # Используем явное указание модели
                classifier = pipeline("sentiment-analysis",
                                      model="distilbert-base-uncased-finetuned-sst-2-english")
                result = classifier(text)[0]
                st.success(f"Result: {result['label']}")
                st.info(f"Confidence: {result['score']:.4f}")

        elif task == "Text Generation":
            with st.spinner("Generating..."):
                generator = pipeline("text-generation",
                                     model="gpt2",
                                     max_length=100)
                result = generator(text, num_return_sequences=1)
                st.write("**Generated text:**")
                st.write(result[0]['generated_text'])

    except Exception as e:
        st.error(f"Error: {str(e)}")