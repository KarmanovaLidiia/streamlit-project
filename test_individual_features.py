from feature_extractor import RussianFeatureExtractor


def test_specific_features():
    """Тестирование конкретных функций экстрактора"""
    print("🔍 ТЕСТИРОВАНИЕ КОНКРЕТНЫХ ФУНКЦИЙ")
    print("=" * 40)

    extractor = RussianFeatureExtractor(use_heavy_models=False)

    # Тест разных типов текстов
    test_cases = [
        {
            'name': 'Хороший ответ',
            'text': 'Здравствуйте! Я живу в большом городе. Здесь много парков и музеев.',
            'question': 'Расскажите о вашем городе.',
            'type': 1
        },
        {
            'name': 'Короткий ответ',
            'text': 'Не знаю.',
            'question': 'Что вы думаете?',
            'type': 2
        },
        {
            'name': 'Ответ с ошибками',
            'text': 'Я живу там где много деревьев и красиво но иногда шумно.',
            'question': 'Где вы живете?',
            'type': 1
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}:")
        print(f"   Текст: '{case['text']}'")

        # Тестируем отдельные функции
        text_features = extractor.extract_enhanced_text_features(case['text'])
        grammar_features = extractor.extract_grammar_features(case['text'])
        discourse_features = extractor.extract_discourse_features(case['text'])

        print(f"   📝 Текстовые: сл={text_features['word_count']}, р={text_features['lexical_diversity']:.2f}")
        print(
            f"   📚 Грамматика: ош={grammar_features['grammar_error_ratio']:.2f}, полн={grammar_features['sentence_completeness']:.2f}")
        print(f"   💬 Дискурс: прив={discourse_features['has_greeting']}, вопр={discourse_features['has_questions']}")


# Запуск всех тестов
if __name__ == "__main__":
    test_specific_features()