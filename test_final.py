try:
    from transformers import pipeline

    print("✅ SUCCESS: Pipeline imported correctly!")

    # Тестируем несколько моделей
    print("Testing sentiment analysis...")
    classifier = pipeline("sentiment-analysis")
    result = classifier("I love this!")[0]
    print(f"Sentiment test: {result}")

    print("Testing text generation...")
    generator = pipeline("text-generation", model="gpt2", max_length=50)
    result = generator("The future of AI")[0]
    print(f"Generation test: {result['generated_text'][:100]}...")

    print("🎉 ALL TESTS PASSED!")

except Exception as e:
    print(f"❌ ERROR: {e}")