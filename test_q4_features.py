import pandas as pd
import sys

sys.path.append('src')

try:
    from features_q4 import enhanced_q4_features

    print("✅ Модуль enhanced_q4_features загружен")

    # Тестовые данные
    test_data = pd.DataFrame({
        'question_number': [4, 4, 4],
        'question_text': ["Опишите картинку..."] * 3,
        'answer_text': [
            "На картинке я вижу семью на кухне. Мама готовит, папа моет посуду. У меня тоже есть семья - двое детей. Мы любим играть вместе.",
            "Не знаю что сказать...",
            "Лето. Парк. Дети играют. У меня трое детей. Мы гуляем в парке."
        ]
    })

    result = enhanced_q4_features(test_data)
    print("\n🔍 РЕЗУЛЬТАТ ТЕСТА:")
    print(result.columns.tolist())  # Какие колонки появились
    print("\n📊 ЗНАЧЕНИЯ ФИЧ:")
    q4_cols = [c for c in result.columns if c.startswith('q4_')]
    print(result[q4_cols].head(3))

except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback

    traceback.print_exc()