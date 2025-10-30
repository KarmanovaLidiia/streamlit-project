import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# Настройка отображения
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'  # Для поддержки кириллицы


def load_and_analyze_data():
    """Загрузка и базовый анализ данных"""

    # Загрузка данных с правильным разделителем
    file_path = 'small.csv'  # или полный путь к файлу

    # Пробуем разные разделители и кодировки
    try:
        # Сначала пробуем с разделителем точка с запятой
        df = pd.read_csv(file_path, encoding='utf-8', delimiter=';')
        print("✅ Файл загружен с разделителем ';' и кодировкой utf-8")
    except:
        try:
            df = pd.read_csv(file_path, encoding='cp1251', delimiter=';')
            print("✅ Файл загружен с разделителем ';' и кодировкой cp1251")
        except:
            try:
                df = pd.read_csv(file_path, encoding='utf-8', delimiter=',')
                print("✅ Файл загружен с разделителем ',' и кодировкой utf-8")
            except:
                try:
                    df = pd.read_csv(file_path, encoding='cp1251', delimiter=',')
                    print("✅ Файл загружен с разделителем ',' и кодировкой cp1251")
                except Exception as e:
                    print(f"❌ Ошибка загрузки файла: {e}")
                    return None

    print("=" * 60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ АВТОМАТИЧЕСКОЙ ОЦЕНКИ")
    print("=" * 60)

    # Базовая информация о данных
    print(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} колонок")
    print(f"\nВсе колонки: {list(df.columns)}")

    # Показываем первые несколько строк для проверки
    print(f"\nПервые 3 строки данных:")
    print(df.head(3))

    return df


def check_and_rename_columns(df):
    """Проверка и переименование колонок если нужно"""

    print("\n" + "=" * 40)
    print("ПРОВЕРКА СТРУКТУРЫ ДАННЫХ")
    print("=" * 40)

    # Если есть только одна колонка, возможно данные объединены
    if df.shape[1] == 1:
        first_column = df.columns[0]
        print(f"Обнаружена одна колонка: '{first_column}'")

        # Проверяем, содержит ли она все данные
        sample_value = str(df.iloc[0, 0])
        if ';' in sample_value:
            print("⚠️  Данные объединены в одну колонку, разделяем...")

            # Разделяем данные по точке с запятой
            split_data = df[first_column].str.split(';', expand=True)

            # Берем первую строку как заголовки
            if split_data.shape[0] > 1:
                new_columns = split_data.iloc[0].tolist()
                split_data = split_data[1:]  # Убираем строку с заголовками
                split_data.columns = new_columns
                df = split_data.reset_index(drop=True)
                print("✅ Данные успешно разделены")
                print(f"Новые колонки: {list(df.columns)}")

    return df


def basic_statistics(df):
    """Базовая статистика по оценкам"""

    print("\n" + "=" * 40)
    print("БАЗОВАЯ СТАТИСТИКА")
    print("=" * 40)

    # Проверяем наличие нужных колонок
    available_columns = list(df.columns)
    print(f"Доступные колонки: {available_columns}")

    # Статистика по AI оценкам (pred_score)
    if 'pred_score' in df.columns:
        print("\nAI оценки (pred_score):")
        print(f"  Среднее: {df['pred_score'].mean():.3f}")
        print(f"  Медиана: {df['pred_score'].median():.3f}")
        print(f"  Стандартное отклонение: {df['pred_score'].std():.3f}")
        print(f"  Минимум: {df['pred_score'].min():.3f}")
        print(f"  Максимум: {df['pred_score'].max():.3f}")
    else:
        print("❌ Колонка 'pred_score' не найдена")

    # Статистика по человеческим оценкам
    human_score_columns = ['Оценка экзаменатора', 'оценка', 'score', 'human_score']
    human_score_col = None

    for col in human_score_columns:
        if col in df.columns:
            human_score_col = col
            break

    if human_score_col:
        print(f"\nОценки экзаменатора ({human_score_col}):")
        print(f"  Среднее: {df[human_score_col].mean():.3f}")
        print(f"  Медиана: {df[human_score_col].median():.3f}")
        print(f"  Стандартное отклонение: {df[human_score_col].std():.3f}")

        # Распределение оценок
        print(f"\nРаспределение оценок экзаменатора:")
        распределение = df[human_score_col].value_counts().sort_index()
        for оценка, count in распределение.items():
            print(f"  {оценка}: {count} ответов ({count / len(df) * 100:.1f}%)")
    else:
        print("❌ Колонка с оценками экзаменатора не найдена")


def calculate_correlations(df):
    """Расчет корреляций и разниц"""

    print("\n" + "=" * 40)
    print("КОРРЕЛЯЦИИ И РАСХОЖДЕНИЯ")
    print("=" * 40)

    # Проверяем наличие обеих колонок
    if 'pred_score' not in df.columns:
        print("❌ Колонка 'pred_score' не найдена для расчета корреляций")
        return

    human_score_columns = ['Оценка экзаменатора', 'оценка', 'score', 'human_score']
    human_score_col = None

    for col in human_score_columns:
        if col in df.columns:
            human_score_col = col
            break

    if not human_score_col:
        print("❌ Колонка с оценками экзаменатора не найдена для расчета корреляций")
        return

    # Корреляция
    correlation = df[[human_score_col, 'pred_score']].corr().iloc[0, 1]
    print(f"Корреляция между оценками: {correlation:.3f}")

    # Разницы между оценками
    df['разница'] = df['pred_score'] - df[human_score_col]
    df['abs_разница'] = abs(df['разница'])

    print(f"\nСредняя абсолютная разница: {df['abs_разница'].mean():.3f}")
    print(f"Максимальная разница: {df['abs_разница'].max():.3f}")
    print(f"Минимальная разница: {df['abs_разница'].min():.3f}")

    # Анализ согласованности
    print("\nСОГЛАСОВАННОСТЬ ОЦЕНОК:")
    for порог in [0.1, 0.3, 0.5, 1.0]:
        согласованные = df[df['abs_разница'] < порог].shape[0]
        процент = (согласованные / len(df)) * 100
        print(f"  Разница < {порог}: {согласованные} ответов ({процент:.1f}%)")

    # Направление разниц
    завышение = len(df[df['разница'] > 0])
    занижение = len(df[df['разница'] < 0])
    совпадение = len(df[df['разница'] == 0])

    print(f"\nНАПРАВЛЕНИЕ РАЗНИЦ:")
    print(f"  AI завышает: {завышение} ({завышение / len(df) * 100:.1f}%)")
    print(f"  AI занижает: {занижение} ({занижение / len(df) * 100:.1f}%)")
    print(f"  Полное совпадение: {совпадение} ({совпадение / len(df) * 100:.1f}%)")


def create_visualizations(df):
    """Создание визуализаций"""

    print("\n" + "=" * 40)
    print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("=" * 40)

    # Проверяем наличие нужных колонок
    if 'pred_score' not in df.columns:
        print("❌ Колонка 'pred_score' не найдена для визуализации")
        return

    human_score_columns = ['Оценка экзаменатора', 'оценка', 'score', 'human_score']
    human_score_col = None

    for col in human_score_columns:
        if col in df.columns:
            human_score_col = col
            break

    if not human_score_col:
        print("❌ Колонка с оценками экзаменатора не найдена для визуализации")
        return

    # Создаем папку для графиков
    os.makedirs('graphs', exist_ok=True)

    # 1. Scatter plot сравнения оценок
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df[human_score_col], df['pred_score'],
                          c=df['abs_разница'], cmap='viridis', alpha=0.7, s=80)
    plt.colorbar(scatter, label='Абсолютная разница')

    # Определяем диапазон для линии идеального соответствия
    min_val = min(df[human_score_col].min(), df['pred_score'].min())
    max_val = max(df[human_score_col].max(), df['pred_score'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Идеальное соответствие')

    plt.xlabel(f'Оценка экзаменатора ({human_score_col})', fontsize=12)
    plt.ylabel('AI оценка (pred_score)', fontsize=12)
    plt.title('Сравнение человеческой и AI оценки\n(цвет показывает величину расхождения)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('graphs/scatter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Гистограмма разниц
    plt.figure(figsize=(12, 6))
    n, bins, patches = plt.hist(df['разница'], bins=30, alpha=0.7,
                                edgecolor='black', color='skyblue')
    plt.xlabel('Разница оценок (AI - Человек)', fontsize=12)
    plt.ylabel('Количество ответов', fontsize=12)
    plt.title('Распределение разниц между AI и человеческими оценками', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Нулевая разница')
    plt.axvline(x=df['разница'].mean(), color='orange', linestyle='--',
                alpha=0.8, linewidth=2, label=f'Средняя разница: {df["разница"].mean():.3f}')
    plt.legend()
    plt.savefig('graphs/difference_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Графики сохранены в папку 'graphs/'")


def analyze_extreme_cases(df):
    """Анализ крайних случаев"""

    print("\n" + "=" * 40)
    print("АНАЛИЗ КРАЙНИХ СЛУЧАЕВ")
    print("=" * 40)

    if 'abs_разница' not in df.columns:
        print("❌ Не найдены данные о разницах оценок")
        return

    human_score_columns = ['Оценка экзаменатора', 'оценка', 'score', 'human_score']
    human_score_col = None

    for col in human_score_columns:
        if col in df.columns:
            human_score_col = col
            break

    if not human_score_col:
        print("❌ Колонка с оценками экзаменатора не найдена")
        return

    # Наибольшие расхождения
    большие_расхождения = df.nlargest(8, 'abs_разница')[
        [human_score_col, 'pred_score', 'abs_разница', 'разница']
    ]

    # Добавляем ID если есть
    id_columns = ['Id экзамена', 'id', 'ID', 'exam_id']
    for col in id_columns:
        if col in df.columns:
            большие_расхождения[col] = df.loc[большие_расхождения.index, col]
            break

    question_columns = ['№ вопроса', 'question', 'вопрос', 'question_id']
    for col in question_columns:
        if col in df.columns:
            большие_расхождения[col] = df.loc[большие_расхождения.index, col]
            break

    print("Топ-8 наибольших расхождений:")
    print("-" * 80)
    for idx, row in большие_расхождения.iterrows():
        направление = "ЗАВЫШЕНИЕ" if row['разница'] > 0 else "ЗАНИЖЕНИЕ"

        # Формируем информацию об ID и вопросе
        id_info = ""
        if 'Id экзамена' in row:
            id_info = f"Экзамен {row['Id экзамена']}"
        elif 'id' in row:
            id_info = f"ID {row['id']}"

        question_info = ""
        if '№ вопроса' in row:
            question_info = f", Вопрос {row['№ вопроса']}"
        elif 'question' in row:
            question_info = f", Вопрос {row['question']}"

        print(f"\n📊 {id_info}{question_info} ({направление}):")
        print(f"   👤 Человек: {row[human_score_col]} | 🤖 AI: {row['pred_score']:.3f}")
        print(f"   📏 Разница: {row['abs_разница']:.3f} ({row['разница']:+.3f})")
        print("-" * 60)


def analyze_explanations(df):
    """Анализ объяснений оценок"""

    print("\n" + "=" * 40)
    print("АНАЛИЗ ОБЪЯСНЕНИЙ ОЦЕНОК")
    print("=" * 40)

    explanation_columns = ['объяснение_оценки', 'explanation', 'объяснение', 'комментарий']
    explanation_col = None

    for col in explanation_columns:
        if col in df.columns:
            explanation_col = col
            break

    if not explanation_col:
        print("❌ Колонка с объяснениями оценок не найдена")
        return

    # Собираем все объяснения
    все_объяснения = ' '.join(df[explanation_col].dropna().astype(str))

    # Разбиваем на слова и фильтруем
    слова = [word.strip() for word in все_объяснения.split() if len(word.strip()) > 2]

    # Анализ частотности
    частотность = Counter(слова)

    print("Топ-15 наиболее частых характеристик в объяснениях:")
    print("-" * 50)
    for слово, count in частотность.most_common(15):
        print(f"  {слово}: {count}")


def save_detailed_analysis(df):
    """Сохранение детального анализа в файл"""

    print("\n" + "=" * 40)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 40)

    if 'abs_разница' not in df.columns:
        print("❌ Нет данных для детального анализа")
        return

    # Создаем копию с анализом
    df_analysis = df.copy()

    human_score_columns = ['Оценка экзаменатора', 'оценка', 'score', 'human_score']
    human_score_col = None

    for col in human_score_columns:
        if col in df.columns:
            human_score_col = col
            break

    if human_score_col and 'pred_score' in df.columns:
        df_analysis['разница_ai_человек'] = df_analysis['pred_score'] - df_analysis[human_score_col]
        df_analysis['abs_разница'] = abs(df_analysis['разница_ai_человек'])

        # Добавляем категоризацию расхождений
        условия = [
            df_analysis['abs_разница'] < 0.1,
            df_analysis['abs_разница'] < 0.3,
            df_analysis['abs_разница'] < 0.5,
            df_analysis['abs_разница'] >= 0.5
        ]
        категории = ['Отличное', 'Хорошее', 'Умеренное', 'Низкое']
        df_analysis['качество_согласования'] = np.select(условия, категории, default='Низкое')

        # Сортируем по наибольшим расхождениям
        df_analysis = df_analysis.sort_values('abs_разница', ascending=False)

    try:
        # Сохраняем в Excel
        with pd.ExcelWriter('detailed_analysis.xlsx', engine='openpyxl') as writer:
            # Все данные
            df_analysis.to_excel(writer, sheet_name='Все_данные_с_анализом', index=False)
            print("✅ Детальный анализ сохранен в 'detailed_analysis.xlsx'")

    except Exception as e:
        print(f"⚠️  Не удалось сохранить Excel, сохраняем в CSV: {e}")
        df_analysis.to_csv('detailed_analysis.csv', index=False, encoding='utf-8')
        print("✅ Детальный анализ сохранен в 'detailed_analysis.csv'")


def main():
    """Основная функция"""

    try:
        # Загрузка данных
        df = load_and_analyze_data()

        if df is None:
            return

        # Проверка и корректировка структуры данных
        df = check_and_rename_columns(df)

        # Выполнение анализа
        basic_statistics(df)
        calculate_correlations(df)
        create_visualizations(df)
        analyze_extreme_cases(df)
        analyze_explanations(df)
        save_detailed_analysis(df)

        print("\n" + "=" * 60)
        print("✅ АНАЛИЗ ЗАВЕРШЕН!")
        print("=" * 60)

    except FileNotFoundError:
        print("❌ ОШИБКА: Файл 'small.csv' не найден в текущей директории")
        print("   Убедитесь, что файл находится в той же папке, что и скрипт")
    except Exception as e:
        print(f"❌ ОШИБКА при выполнении анализа: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()