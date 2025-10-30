import pandas as pd
import sys
sys.path.append('src')

# Пробуем разные варианты импорта
try:
    from features_q4 import enhanced_q4_features
    print("✅ Импорт enhanced_q4_features - УСПЕХ")
    func = enhanced_q4_features
except:
    try:
        from features_q4 import q4_slot_features
        print("✅ Импорт q4_slot_features - УСПЕХ")
        func = q4_slot_features
    except Exception as e:
        print(f"❌ Ошибка импорта: {e}")
        exit()

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

try:
    result = func(test_data)
    print(f"✅ Функция выполнена успешно!")
    print(f"📊 Колонки: {[c for c in result.columns if 'q4' in c]}")
    print(f"🔍 Пример данных:")
    print(result[['question_number', 'answer_text']].join(
        result[[c for c in result.columns if 'q4' in c]].head(2)
    ))
except Exception as e:
    print(f"❌ Ошибка выполнения: {e}")
    import traceback
    traceback.print_exc()