import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extractor import RussianFeatureExtractor
import os
import sys
import subprocess
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def check_environment():
    """Проверка окружения и зависимостей"""
    print("=== ПРОВЕРКА ОКРУЖЕНИЯ ===")

    # Проверка пакетов
    packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 'torch']
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")

    # Проверка Java
    try:
        subprocess.run(['java', '-version'], capture_output=True, check=True)
        print("✅ Java установлена")
    except:
        print("❌ Java не установлена - грамматический анализ не будет работать")


def load_and_analyze_dataset():
    """Загрузка и анализ структуры данных"""
    print("\n=== АНАЛИЗ ДАННЫХ ===")

    try:
        # Пробуем разные варианты загрузки
        for filename in ['small.csv', 'dataset.csv', 'train.csv']:
            if os.path.exists(filename):
                print(f"Найден файл: {filename}")

                # Пробуем разные разделители
                for delimiter in [';', ',', '\t']:
                    try:
                        df = pd.read_csv(filename, encoding='utf-8', delimiter=delimiter)
                        if len(df.columns) > 1:  # Успешная загрузка
                            print(f"✅ Успешно загружен с разделителем '{delimiter}'")
                            break
                    except:
                        continue
                else:
                    print("❌ Не удалось определить разделитель")
                    return None

                print(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} колонок")
                print(f"Колонки: {df.columns.tolist()}")

                # Анализ содержания
                print("\n--- СТРУКТУРА ДАННЫХ ---")
                for col in df.columns:
                    print(f"{col}: {df[col].dtype}, пропусков: {df[col].isnull().sum()}")

                    if df[col].dtype == 'object':
                        sample = df[col].iloc[0] if not df[col].isnull().all() else "N/A"
                        print(f"  Пример: {str(sample)[:100]}...")

                # Поиск ключевых колонок
                question_cols = [col for col in df.columns if 'вопрос' in col.lower()]
                transcript_cols = [col for col in df.columns if 'транскрипт' in col.lower()]
                score_cols = [col for col in df.columns if 'оценк' in col.lower() or 'балл' in col.lower()]

                print(f"\n--- ВЫЯВЛЕННЫЕ КОЛОНКИ ---")
                print(f"Вопросы: {question_cols}")
                print(f"Транскрипты: {transcript_cols}")
                print(f"Оценки: {score_cols}")

                return df

        print("❌ Не найден файл с данными")
        return None

    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return None


def test_alternative_features(texts):
    """Тест альтернативных методов извлечения признаков"""
    print("\n=== ТЕСТ АЛЬТЕРНАТИВНЫХ ПРИЗНАКОВ ===")

    features_list = []

    for i, text in enumerate(texts):
        if pd.isna(text):
            features_list.append({})
            continue

        text_str = str(text)
        features = {}

        # Базовые текстовые метрики
        features['text_length'] = len(text_str)

        words = re.findall(r'\b[а-яёa-z]+\b', text_str.lower())
        features['word_count'] = len(words)

        sentences = re.split(r'[.!?]+', text_str)
        features['sentence_count'] = len([s for s in sentences if len(s.strip()) > 10])

        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['lexical_diversity'] = len(set(words)) / len(words) if words else 0

        # Стилистические особенности
        features['has_questions'] = int('?' in text_str)
        features['has_exclamations'] = int('!' in text_str)
        features['has_ellipsis'] = int('...' in text_str)

        # Сложность текста
        long_words = [w for w in words if len(w) > 6]
        features['long_word_ratio'] = len(long_words) / len(words) if words else 0

        features_list.append(features)

        if i < 3:  # Показать пример для первых 3 текстов
            print(f"Пример {i + 1}: {text_str[:80]}...")
            for k, v in features.items():
                print(f"  {k}: {v:.3f}")

    return pd.DataFrame(features_list)


def enhanced_feature_extraction(df):
    """Улучшенное извлечение признаков с резервными методами"""
    print("\n=== ЗАПУСК УЛУЧШЕННОГО ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ ===")

    # Определяем колонку с транскриптами
    transcript_cols = [col for col in df.columns if 'транскрипт' in col.lower()]
    if not transcript_cols:
        print("❌ Не найдена колонка с транскриптами")
        return pd.DataFrame()

    transcript_col = transcript_cols[0]
    texts = df[transcript_col].fillna('')

    print(f"Обработка {len(texts)} транскриптов...")

    try:
        # Пробуем основной экстрактор
        print("🔄 Попытка использовать RussianFeatureExtractor...")
        extractor = RussianFeatureExtractor()
        features_df = extractor.extract_features_for_dataframe(df)

        if not features_df.empty:
            print("✅ RussianFeatureExtractor успешно отработал")
            return features_df
        else:
            print("❌ RussianFeatureExtractor вернул пустой DataFrame")

    except Exception as e:
        print(f"❌ Ошибка в RussianFeatureExtractor: {e}")
        print("🔄 Переход на резервный метод...")

    # Резервный метод - базовые признаки
    print("Использование резервного метода извлечения признаков...")
    features_df = test_alternative_features(texts)

    return features_df


def analyze_correlations_with_scores(features_df, original_df):
    """Анализ корреляций с реальными оценками"""
    print("\n=== АНАЛИЗ КОРРЕЛЯЦИЙ ===")

    # Находим колонку с оценками
    score_cols = [col for col in original_df.columns if 'оценк' in col.lower() or 'балл' in col.lower()]
    if not score_cols:
        print("❌ Не найдены колонки с оценками")
        return

    score_col = score_cols[0]

    # Объединяем признаки с оценками
    analysis_df = features_df.copy()
    analysis_df['real_score'] = original_df[score_col].values

    # Удаляем строки с пропусками
    analysis_clean = analysis_df.dropna()

    if len(analysis_clean) < 2:
        print("❌ Недостаточно данных для анализа корреляций")
        return

    # Анализ корреляций
    correlations = analysis_clean.corr()['real_score'].sort_values(key=abs, ascending=False)

    print("\nТоп-15 наиболее коррелирующих признаков:")
    print("-" * 50)

    for feature, corr in correlations.items():
        if feature != 'real_score':
            direction = "+" if corr > 0 else "-"
            significance = "***" if abs(corr) > 0.3 else "**" if abs(corr) > 0.2 else "*" if abs(corr) > 0.1 else ""
            print(f"  {direction} {feature}: {corr:+.3f} {significance}")

    # Визуализация топ-признаков
    top_features = correlations.head(6).index.tolist()
    if 'real_score' in top_features:
        top_features.remove('real_score')

    if top_features:
        plt.figure(figsize=(12, 8))

        for i, feature in enumerate(top_features[:5]):
            plt.subplot(2, 3, i + 1)
            plt.scatter(analysis_clean[feature], analysis_clean['real_score'], alpha=0.6)
            plt.xlabel(feature)
            plt.ylabel('Real Score')
            plt.title(f'r = {correlations[feature]:.3f}')

        plt.tight_layout()
        plt.show()

    return analysis_clean


def save_detailed_report(features_df, original_df, analysis_df):
    """Сохранение детального отчета"""
    print("\n=== СОХРАНЕНИЕ ОТЧЕТА ===")

    # Сохраняем признаки
    features_df.to_csv('extracted_features_enhanced.csv', encoding='utf-8')
    print("✅ Признаки сохранены в extracted_features_enhanced.csv")

    # Детальный отчет
    with open('features_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("ДЕТАЛЬНЫЙ ОТЧЕТ ПО АНАЛИЗУ ПРИЗНАКОВ\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"ОБЩАЯ СТАТИСТИКА:\n")
        f.write(f"- Обработано строк: {len(features_df)}/{len(original_df)}\n")
        f.write(f"- Извлечено признаков: {len(features_df.columns)}\n")
        f.write(f"- Заполненность данных: {features_df.notna().mean().mean():.1%}\n\n")

        f.write("СПИСОК ПРИЗНАКОВ:\n")
        for col in features_df.columns:
            f.write(f"\n{col}:\n")
            f.write(f"  Тип: {features_df[col].dtype}\n")
            f.write(f"  Не-NULL: {features_df[col].notna().sum()}\n")
            f.write(f"  Среднее: {features_df[col].mean():.3f}\n")
            f.write(f"  Std: {features_df[col].std():.3f}\n")

            if analysis_df is not None and 'real_score' in analysis_df.columns:
                corr = analysis_df.corr()['real_score'].get(col, 0)
                f.write(f"  Корреляция с оценкой: {corr:+.3f}\n")

        if analysis_df is not None:
            f.write("\nКОРРЕЛЯЦИОННЫЙ АНАЛИЗ:\n")
            correlations = analysis_df.corr()['real_score'].sort_values(key=abs, ascending=False)
            for feature, corr in correlations.items():
                if feature != 'real_score' and abs(corr) > 0.1:
                    f.write(f"  {feature}: {corr:+.3f}\n")

    print("✅ Детальный отчет сохранен в features_analysis_report.txt")


def main():
    """Основная функция тестирования"""
    print("🚀 ЗАПУСК РАСШИРЕННОГО ТЕСТИРОВАНИЯ")
    print("=" * 60)

    # Проверка окружения
    check_environment()

    # Загрузка данных
    df = load_and_analyze_dataset()
    if df is None:
        print("❌ Не удалось загрузить данные для тестирования")
        return

    # Извлечение признаков
    features_df = enhanced_feature_extraction(df)

    if features_df.empty:
        print("❌ Не удалось извлечь признаки")
        return

    # Анализ корреляций
    analysis_df = analyze_correlations_with_scores(features_df, df)

    # Сохранение результатов
    save_detailed_report(features_df, df, analysis_df)

    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)


if __name__ == "__main__":
    main()