# test_simple.py
import pandas as pd
from feature_extractor import RussianFeatureExtractor


def simple_test():
    """Простой тест исправленного экстрактора"""
    print("🧪 ПРОСТОЙ ТЕСТ ИСПРАВЛЕННОЙ ВЕРСИИ")
    print("=" * 50)

    # Создаем экстрактор
    extractor = RussianFeatureExtractor(use_heavy_models=False)

    # Тестовые данные с разным качеством ответов
    test_cases = [
        {
            'name': 'Хороший ответ',
            'question': 'Расскажите о вашем городе',
            'answer': 'Привет! Я живу в Санкт-Петербурге. Это культурная столица России с красивыми дворцами, каналами и богатой историей.',
            'type': 1
        },
        {
            'name': 'Средний ответ',
            'question': 'Что вы любите делать?',
            'answer': 'Люблю читать книги и гулять. Иногда хожу в кино.',
            'type': 2
        },
        {
            'name': 'Плохой ответ',
            'question': 'Опишите картинку',
            'answer': 'Не знаю...',
            'type': 4
        }
    ]

    print("\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print("-" * 60)

    for case in test_cases:
        test_df = pd.DataFrame([{
            'Текст вопроса': case['question'],
            'Транскрибация ответа': case['answer'],
            '№ вопроса': case['type']
        }])

        features = extractor.extract_all_features(test_df.iloc[0])

        print(f"\n{case['name']}:")
        print(f"  Ответ: '{case['answer'][:40]}...'")
        print(f"  composite_quality_score: {features['composite_quality_score']:.3f}")
        print(f"  grammar_quality: {features['grammar_quality']:.3f}")
        print(f"  style_score: {features['style_score']:.3f}")
        print(f"  word_count: {features['word_count']}")
        print(f"  lexical_diversity: {features['lexical_diversity']:.3f}")


if __name__ == "__main__":
    simple_test()