import pandas as pd
import numpy as np
from feature_extractor import RussianFeatureExtractor
import os
import time


def test_with_real_data():
    """Тестирование на реальных данных с замером времени"""
    print("🧪 ТЕСТИРОВАНИЕ НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 50)

    # Пробуем разные файлы
    data_files = ['small.csv', 'dataset.csv', 'train.csv']
    found_file = None

    for file in data_files:
        if os.path.exists(file):
            found_file = file
            print(f"✅ Найден файл данных: {file}")
            break

    if not found_file:
        print("❌ Файлы данных не найдены!")
        return

    # Загружаем данные
    try:
        df = pd.read_csv(found_file, encoding='utf-8', delimiter=';')
        print(f"📊 Загружено {len(df)} строк, {len(df.columns)} колонок")
        print(f"Колонки: {df.columns.tolist()}")
    except:
        try:
            df = pd.read_csv(found_file, encoding='utf-8', delimiter=',')
            print(f"📊 Загружено {len(df)} строк (разделитель ',')")
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return

    # Берем небольшую выборку для теста
    sample_size = min(50, len(df))
    sample_df = df.head(sample_size).copy()

    print(f"\n🔧 ИНИЦИАЛИЗАЦИЯ ЭКСТРАКТОРА...")
    start_time = time.time()

    # Создаем экстрактор
    extractor = RussianFeatureExtractor(use_heavy_models=False)

    init_time = time.time() - start_time
    print(f"✅ Экстрактор инициализирован за {init_time:.1f} сек")

    print(f"\n🎯 ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ДЛЯ {sample_size} СТРОК...")
    extract_start = time.time()

    # Извлекаем признаки
    features_df = extractor.extract_features_for_dataframe(sample_df)

    extract_time = time.time() - extract_start

    if not features_df.empty:
        print(f"✅ Извлечение завершено за {extract_time:.1f} сек")
        print(f"📈 Получено {len(features_df.columns)} признаков")

        # Анализ результатов
        print(f"\n📊 СТАТИСТИКА ПРИЗНАКОВ:")
        print(f"   - Успешно обработано: {len(features_df)}/{sample_size} строк")
        print(f"   - Заполненность данных: {features_df.notna().mean().mean():.1%}")

        # Показываем топ признаков по вариативности
        numeric_features = features_df.select_dtypes(include=[np.number])
        if not numeric_features.empty:
            std_dev = numeric_features.std().sort_values(ascending=False)
            print(f"\n🎯 ТОП-5 самых вариативных признаков:")
            for feature, std_val in std_dev.head(5).items():
                print(f"   {feature}: {std_val:.3f}")

        # Сохраняем результаты
        output_file = 'real_data_features.csv'
        features_df.to_csv(output_file, encoding='utf-8')
        print(f"\n💾 Результаты сохранены в {output_file}")

        # Сохраняем описания
        with open('features_description_detailed.txt', 'w', encoding='utf-8') as f:
            f.write("ПОДРОБНОЕ ОПИСАНИЕ ПРИЗНАКОВ\n")
            f.write("=" * 50 + "\n\n")

            for col in features_df.columns:
                f.write(f"{col}:\n")
                f.write(f"  Тип: {features_df[col].dtype}\n")
                f.write(f"  Не-NULL: {features_df[col].notna().sum()}\n")
                f.write(f"  Среднее: {features_df[col].mean():.3f}\n")
                f.write(f"  Std: {features_df[col].std():.3f}\n")
                f.write(f"  Min: {features_df[col].min():.3f}\n")
                f.write(f"  Max: {features_df[col].max():.3f}\n\n")

        print("📝 Описание признаков сохранено в features_description_detailed.txt")

    else:
        print("❌ Не удалось извлечь признаки")


def compare_old_vs_new():
    """Сравнение старого и нового экстрактора"""
    print("\n" + "=" * 50)
    print("СРАВНЕНИЕ СТАРОГО И НОВОГО МЕТОДОВ")
    print("=" * 50)

    # Тестовый текст
    test_text = "Привет! Меня зовут Мария. Я живу в Москве и учусь в университете."

    # Старый метод (базовые признаки)
    from feature_extractor import extract_quick_features
    quick_features = extract_quick_features(test_text)

    # Новый метод (полные признаки)
    test_data = {
        'Текст вопроса': ['Расскажите о себе'],
        'Транскрибация ответа': [test_text],
        '№ вопроса': [1]
    }
    test_df = pd.DataFrame(test_data)

    extractor = RussianFeatureExtractor(use_heavy_models=False)
    full_features_df = extractor.extract_features_for_dataframe(test_df)

    print("📊 СРАВНЕНИЕ:")
    print(f"Быстрый метод: {len(quick_features)} признаков")
    if not full_features_df.empty:
        print(f"Полный метод: {len(full_features_df.columns)} признаков")

        # Сравниваем общие признаки
        common_features = set(quick_features.keys()) & set(full_features_df.columns)
        print(f"Общих признаков: {len(common_features)}")

        print("\n📈 ЗНАЧЕНИЯ ОБЩИХ ПРИЗНАКОВ:")
        for feature in list(common_features)[:5]:  # Показываем первые 5
            old_val = quick_features[feature]
            new_val = full_features_df[feature].iloc[0]
            print(f"  {feature}: {old_val:.3f} -> {new_val:.3f}")


if __name__ == "__main__":
    test_with_real_data()
    compare_old_vs_new()