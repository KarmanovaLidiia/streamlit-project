import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_extracted_features():
    """Анализ извлеченных признаков без сложных зависимостей"""

    try:
        # Загружаем извлеченные признаки
        features_df = pd.read_csv('real_data_features.csv', index_col=0)
    except FileNotFoundError:
        print("❌ Файл real_data_features.csv не найден!")
        print("💡 Сначала запустите test_real_data.py")
        return

    print("📊 ДЕТАЛЬНЫЙ АНАЛИЗ ИЗВЛЕЧЕННЫХ ПРИЗНАКОВ")
    print("=" * 50)

    print(f"Всего признаков: {len(features_df.columns)}")
    print(f"Обработано строк: {len(features_df)}")

    # Анализ заполненности
    null_analysis = features_df.isnull().sum()
    null_features = null_analysis[null_analysis > 0]

    if len(null_features) > 0:
        print(f"\n❌ Признаки с пропусками:")
        for feature, null_count in null_features.items():
            print(f"   {feature}: {null_count} пропусков ({null_count / len(features_df):.1%})")
    else:
        print(f"\n✅ Все признаки полностью заполнены!")

    # Статистика по числовым признакам
    numeric_features = features_df.select_dtypes(include=[np.number])

    print(f"\n📈 СТАТИСТИКА ПРИЗНАКОВ:")
    stats_summary = numeric_features.agg(['mean', 'std', 'min', 'max']).T
    stats_summary['cv'] = stats_summary['std'] / stats_summary['mean']  # Коэффициент вариации

    # Показываем топ-10 самых информативных признаков
    informative_features = stats_summary[stats_summary['std'] > 0].sort_values('cv', ascending=False)

    print(f"\n🎯 ТОП-15 самых информативных признаков (по вариативности):")
    for feature, row in informative_features.head(15).iterrows():
        print(f"   {feature:25} mean={row['mean']:6.2f} std={row['std']:6.2f} cv={row['cv']:.2f}")

    # Визуализация распределения ключевых признаков
    key_features = ['text_length', 'word_count', 'lexical_diversity', 'composite_quality_score']
    available_features = [f for f in key_features if f in numeric_features.columns]

    if available_features:
        plt.figure(figsize=(15, 10))

        for i, feature in enumerate(available_features, 1):
            plt.subplot(2, 2, i)
            plt.hist(numeric_features[feature].dropna(), bins=20, alpha=0.7, edgecolor='black')
            plt.title(f'Распределение {feature}')
            plt.xlabel(feature)
            plt.ylabel('Частота')

        plt.tight_layout()
        plt.savefig('features_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\n📊 Визуализация сохранена в features_distribution.png")

    # Анализ качества композитного показателя
    if 'composite_quality_score' in numeric_features.columns:
        print(f"\n🎯 АНАЛИЗ КОМПОЗИТНОГО ПОКАЗАТЕЛЯ КАЧЕСТВА:")
        quality_scores = numeric_features['composite_quality_score']
        print(f"   Среднее: {quality_scores.mean():.3f}")
        print(f"   Стандартное отклонение: {quality_scores.std():.3f}")
        print(f"   Диапазон: [{quality_scores.min():.3f}, {quality_scores.max():.3f}]")

        # Распределение по квантилям
        quantiles = quality_scores.quantile([0.25, 0.5, 0.75])
        print(f"   Квантили: 25%={quantiles[0.25]:.3f}, 50%={quantiles[0.5]:.3f}, 75%={quantiles[0.75]:.3f}")

        # Анализ что влияет на качество
        print(f"\n🔍 КОРРЕЛЯЦИЯ С КОМПОЗИТНЫМ ПОКАЗАТЕЛЕМ:")
        correlations = numeric_features.corr()['composite_quality_score'].abs().sort_values(ascending=False)

        for feature, corr in correlations.head(10).items():
            if feature != 'composite_quality_score':
                actual_corr = numeric_features.corr()['composite_quality_score'][feature]
                direction = "↑" if actual_corr > 0 else "↓"
                print(f"   {direction} {feature:25} {actual_corr:+.3f}")


def check_feature_correlations_with_target():
    """Проверка корреляции признаков с целевой переменной (если есть оценки)"""

    try:
        features_df = pd.read_csv('real_data_features.csv', index_col=0)
    except FileNotFoundError:
        return

    # Ищем колонку с оценками в исходных данных
    score_columns = [col for col in features_df.columns if
                     'score' in col.lower() or 'оценк' in col.lower() or 'балл' in col.lower()]

    if score_columns:
        target_col = score_columns[0]
        print(f"\n🎯 КОРРЕЛЯЦИЯ ПРИЗНАКОВ С {target_col}:")
        print("-" * 50)

        correlations = features_df.corr()[target_col].abs().sort_values(ascending=False)

        # Показываем топ-10 наиболее коррелирующих признаков
        top_correlated = correlations.head(11)  # +1 потому что target сам с собой

        print(f"   {'ПРИЗНАК':<25} {'КОРРЕЛЯЦИЯ':<10} {'ЗНАЧИМОСТЬ'}")
        print(f"   {'-' * 25} {'-' * 10} {'-' * 10}")

        for feature, corr in top_correlated.items():
            if feature != target_col:
                actual_corr = features_df.corr()[target_col][feature]
                direction = "↑" if actual_corr > 0 else "↓"
                significance = "***" if abs(actual_corr) > 0.3 else "**" if abs(actual_corr) > 0.2 else "*" if abs(
                    actual_corr) > 0.1 else ""
                print(f"   {direction} {feature:<23} {actual_corr:+.3f}    {significance}")
    else:
        print(f"\nℹ️ Целевая переменная (оценки) не найдена в данных")


def analyze_feature_categories():
    """Анализ признаков по категориям"""

    try:
        features_df = pd.read_csv('real_data_features.csv', index_col=0)
    except FileNotFoundError:
        return

    # Группируем признаки по категориям
    categories = {
        '📝 ТЕКСТОВЫЕ': ['text_length', 'word_count', 'sentence_count', 'avg_sentence_length',
                        'avg_word_length', 'lexical_diversity', 'long_word_ratio', 'text_complexity'],
        '🎯 СЕМАНТИЧЕСКИЕ': ['semantic_similarity', 'keyword_overlap', 'tfidf_similarity', 'response_relevance'],
        '📚 ГРАММАТИЧЕСКИЕ': ['grammar_error_count', 'grammar_error_ratio', 'has_punctuation',
                             'sentence_completeness', 'proper_capitalization'],
        '💬 ДИСКУРС': ['has_greeting', 'has_questions', 'has_description', 'has_connectors',
                      'has_emotional_words', 'coherence_score'],
        '❓ ТИПЫ ВОПРОСОВ': ['dialog_initiation', 'response_adequacy', 'information_seeking',
                            'descriptive_detail', 'answer_length_sufficiency', 'question_type'],
        '⭐ КАЧЕСТВО': ['composite_quality_score', 'social_appropriateness', 'interaction_quality']
    }

    print(f"\n📂 РАСПРЕДЕЛЕНИЕ ПРИЗНАКОВ ПО КАТЕГОРИЯМ:")
    print("=" * 50)

    numeric_features = features_df.select_dtypes(include=[np.number])

    for category, features in categories.items():
        available_features = [f for f in features if f in numeric_features.columns]
        if available_features:
            print(f"\n{category} ({len(available_features)} признаков):")
            for feature in available_features:
                mean_val = numeric_features[feature].mean()
                std_val = numeric_features[feature].std()
                print(f"   • {feature:25} {mean_val:6.3f} ± {std_val:5.3f}")


if __name__ == "__main__":
    analyze_extracted_features()
    check_feature_correlations_with_target()
    analyze_feature_categories()

    print(f"\n✅ Анализ завершен!")
    print("💡 Рекомендации будут основаны на этом анализе")