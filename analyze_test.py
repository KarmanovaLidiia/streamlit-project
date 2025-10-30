import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# Настройка отображения
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'


def load_and_analyze_data():
    """Загрузка тестовых данных"""

    file_path = 'test_data.csv'

    try:
        df = pd.read_csv(file_path, encoding='utf-8', delimiter=';')
        print("✅ Тестовый файл загружен успешно")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        print("Убедитесь, что файл test_data.csv находится в той же папке")
        return None

    print("=" * 60)
    print("ТЕСТОВЫЙ АНАЛИЗ AI-ОЦЕНОК")
    print("=" * 60)

    print(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} колонок")
    print(f"Колонки: {list(df.columns)}")
    print(f"\nПервые 3 строки:")
    print(df.head(3))

    return df


def basic_statistics(df):
    """Базовая статистика"""

    print("\n" + "=" * 40)
    print("БАЗОВАЯ СТАТИСТИКА")
    print("=" * 40)

    print("AI оценки (pred_score):")
    print(f"  Среднее: {df['pred_score'].mean():.3f}")
    print(f"  Медиана: {df['pred_score'].median():.3f}")
    print(f"  Стандартное отклонение: {df['pred_score'].std():.3f}")
    print(f"  Минимум: {df['pred_score'].min():.3f}")
    print(f"  Максимум: {df['pred_score'].max():.3f}")

    print("\nОценки экзаменатора:")
    print(f"  Среднее: {df['Оценка экзаменатора'].mean():.3f}")
    print(f"  Медиана: {df['Оценка экзаменатора'].median():.3f}")
    print(f"  Стандартное отклонение: {df['Оценка экзаменатора'].std():.3f}")

    print("\nРаспределение оценок экзаменатора:")
    распределение = df['Оценка экзаменатора'].value_counts().sort_index()
    for оценка, count in распределение.items():
        print(f"  {оценка}: {count} ответов ({count / len(df) * 100:.1f}%)")


def calculate_correlations(df):
    """Расчет корреляций"""

    print("\n" + "=" * 40)
    print("КОРРЕЛЯЦИИ И РАСХОЖДЕНИЯ")
    print("=" * 40)

    correlation = df[['Оценка экзаменатора', 'pred_score']].corr().iloc[0, 1]
    print(f"Корреляция между оценками: {correlation:.3f}")

    df['разница'] = df['pred_score'] - df['Оценка экзаменатора']
    df['abs_разница'] = abs(df['разница'])

    print(f"Средняя абсолютная разница: {df['abs_разница'].mean():.3f}")
    print(f"Максимальная разница: {df['abs_разница'].max():.3f}")
    print(f"Минимальная разница: {df['abs_разница'].min():.3f}")

    # Анализ согласованности
    print("\nСОГЛАСОВАННОСТЬ ОЦЕНОК:")
    for порог in [0.1, 0.3, 0.5, 1.0]:
        согласованные = df[df['abs_разница'] < порог].shape[0]
        процент = (согласованные / len(df)) * 100
        print(f"  Разница < {порог}: {согласованные} ответов ({процент:.1f}%)")


def create_visualizations(df):
    """Создание графиков"""

    print("\n" + "=" * 40)
    print("СОЗДАНИЕ ГРАФИКОВ")
    print("=" * 40)

    os.makedirs('graphs', exist_ok=True)

    # 1. Scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['Оценка экзаменатора'], df['pred_score'],
                          c=df['abs_разница'], cmap='viridis', alpha=0.7, s=60)
    plt.colorbar(scatter, label='Абсолютная разница')
    plt.plot([0, 2], [0, 2], 'r--', alpha=0.5, label='Идеальное соответствие')
    plt.xlabel('Оценка экзаменатора')
    plt.ylabel('AI оценка (pred_score)')
    plt.title('Сравнение человеческой и AI оценки')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('graphs/test_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Гистограмма разниц
    plt.figure(figsize=(10, 6))
    plt.hist(df['разница'], bins=15, alpha=0.7, edgecolor='black', color='skyblue')
    plt.xlabel('Разница (AI - Человек)')
    plt.ylabel('Количество ответов')
    plt.title('Распределение разниц оценок')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Нулевая разница')
    plt.legend()
    plt.savefig('graphs/test_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Графики сохранены в папку 'graphs/'")


def analyze_explanations(df):
    """Анализ объяснений"""

    print("\n" + "=" * 40)
    print("АНАЛИЗ ОБЪЯСНЕНИЙ")
    print("=" * 40)

    все_объяснения = ' '.join(df['объяснение_оценки'].dropna().astype(str))
    слова = [word.strip() for word in все_объяснения.split() if len(word.strip()) > 2]
    частотность = Counter(слова)

    print("Топ-10 характеристик в объяснениях:")
    for слово, count in частотность.most_common(10):
        print(f"  {слово}: {count}")


def main():
    """Основная функция"""

    df = load_and_analyze_data()
    if df is None:
        return

    basic_statistics(df)
    calculate_correlations(df)
    create_visualizations(df)
    analyze_explanations(df)

    print("\n" + "=" * 60)
    print("✅ ТЕСТОВЫЙ АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 60)
    print("📊 Созданные файлы:")
    print("   • graphs/test_scatter.png")
    print("   • graphs/test_histogram.png")


if __name__ == "__main__":
    main()