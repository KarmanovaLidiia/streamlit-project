import streamlit as st
from transformers import pipeline


def main():
    st.title("Working AI Models")
    st.write("Based on your successful test")

    # Используем ту же модель, что работала в тесте
    text = st.text_area("Text to analyze:", "I love this product!")

    if st.button("Get Sentiment"):
        try:
            classifier = pipeline("sentiment-analysis")
            result = classifier(text)[0]
            st.write(f"**Label:** {result['label']}")
            st.write(f"**Score:** {result['score']:.4f}")
            st.balloons()
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()