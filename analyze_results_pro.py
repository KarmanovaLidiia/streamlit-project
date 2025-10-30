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
    """Загрузка и базовый анализ данных"""

    file_path = 'small.csv'

    try:
        df = pd.read_csv(file_path, encoding='utf-8', delimiter=';')
        print("Файл загружен с разделителем ';' и кодировкой utf-8")
    except:
        try:
            df = pd.read_csv(file_path, encoding='cp1251', delimiter=';')
            print("Файл загружен с разделителем ';' и кодировкой cp1251")
        except:
            try:
                df = pd.read_csv(file_path, encoding='utf-8', delimiter=',')
                print("Файл загружен с разделителем ',' и кодировкой utf-8")
            except:
                try:
                    df = pd.read_csv(file_path, encoding='cp1251', delimiter=',')
                    print("Файл загружен с разделителем ',' и кодировкой cp1251")
                except Exception as e:
                    print(f"Ошибка загрузки файла: {e}")
                    return None

    print("=" * 60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ АВТОМАТИЧЕСКОЙ ОЦЕНКИ")
    print("=" * 60)

    print(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} колонок")
    print(f"Колонки: {list(df.columns)}")

    return df


def basic_statistics(df):
    """Базовая статистика по оценкам"""

    print("\n" + "=" * 40)
    print("БАЗОВАЯ СТАТИСТИКА")
    print("=" * 40)

    # Статистика по AI оценкам
    print("AI оценки (pred_score):")
    print(f"  Среднее: {df['pred_score'].mean():.3f}")
    print(f"  Медиана: {df['pred_score'].median():.3f}")
    print(f"  Стандартное отклонение: {df['pred_score'].std():.3f}")
    print(f"  Минимум: {df['pred_score'].min():.3f}")
    print(f"  Максимум: {df['pred_score'].max():.3f}")

    # Статистика по человеческим оценкам
    print("\nОценки экзаменатора:")
    print(f"  Среднее: {df['Оценка экзаменатора'].mean():.3f}")
    print(f"  Медиана: {df['Оценка экзаменатора'].median():.3f}")
    print(f"  Стандартное отклонение: {df['Оценка экзаменатора'].std():.3f}")

    # Распределение оценок
    print("\nРаспределение оценок экзаменатора:")
    распределение = df['Оценка экзаменатора'].value_counts().sort_index()
    for оценка, count in распределение.items():
        print(f"  {оценка}: {count} ответов ({count / len(df) * 100:.1f}%)")


def calculate_correlations(df):
    """Расчет корреляций и разниц"""

    print("\n" + "=" * 40)
    print("КОРРЕЛЯЦИИ И РАСХОЖДЕНИЯ")
    print("=" * 40)

    # Корреляция
    correlation = df[['Оценка экзаменатора', 'pred_score']].corr().iloc[0, 1]
    print(f"Корреляция между оценками: {correlation:.3f}")

    # Разницы между оценками
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

    # Создаем папку для графиков
    os.makedirs('graphs', exist_ok=True)

    # 1. Scatter plot сравнения оценок
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Оценка экзаменатора'], df['pred_score'],
                          c=df['abs_разница'], cmap='viridis', alpha=0.7, s=80)
    plt.colorbar(scatter, label='Абсолютная разница')
    plt.plot([0, 2], [0, 2], 'r--', alpha=0.5, label='Идеальное соответствие')
    plt.xlabel('Оценка экзаменатора', fontsize=12)
    plt.ylabel('AI оценка (pred_score)', fontsize=12)
    plt.title('Сравнение человеческой и AI оценки', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks([0, 1, 2])
    plt.yticks(np.arange(0, 2.5, 0.5))
    plt.savefig('graphs/scatter_comparison_pro.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('graphs/difference_histogram_pro.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Box plot по типам вопросов
    plt.figure(figsize=(14, 8))
    box_data = [df[df['№ вопроса'] == question]['pred_score'].values
                for question in sorted(df['№ вопроса'].unique())]

    box_plot = plt.boxplot(box_data, labels=sorted(df['№ вопроса'].unique()),
                           patch_artist=True)

    # Раскрашиваем boxplot
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Распределение AI оценок по номерам вопросов', fontsize=14)
    plt.xlabel('Номер вопроса', fontsize=12)
    plt.ylabel('AI оценка (pred_score)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('graphs/question_boxplot_pro.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Графики сохранены в папку 'graphs/'")


def analyze_extreme_cases(df):
    """Анализ крайних случаев"""

    print("\n" + "=" * 40)
    print("АНАЛИЗ КРАЙНИХ СЛУЧАЕВ")
    print("=" * 40)

    # Наибольшие расхождения
    большие_расхождения = df.nlargest(8, 'abs_разница')[
        ['Id экзамена', '№ вопроса', 'Оценка экзаменатора', 'pred_score',
         'abs_разница', 'разница']
    ]

    print("Топ-8 наибольших расхождений:")
    print("-" * 80)
    for idx, row in большие_расхождения.iterrows():
        направление = "ЗАВЫШЕНИЕ" if row['разница'] > 0 else "ЗАНИЖЕНИЕ"
        print(f"\nЭкзамен {row['Id экзамена']}, Вопрос {row['№ вопроса']} ({направление}):")
        print(f"  Человек: {row['Оценка экзаменатора']} | AI: {row['pred_score']:.3f}")
        print(f"  Разница: {row['abs_разница']:.3f} ({row['разница']:+.3f})")
        print("-" * 60)


def analyze_explanations(df):
    """Анализ объяснений оценок"""

    print("\n" + "=" * 40)
    print("АНАЛИЗ ОБЪЯСНЕНИЙ ОЦЕНОК")
    print("=" * 40)

    explanation_columns = ['объяснение_оценки', 'explanation', 'объяснение']
    explanation_col = None

    for col in explanation_columns:
        if col in df.columns:
            explanation_col = col
            break

    if not explanation_col:
        print("Колонка с объяснениями оценок не найдена")
        return

    # Собираем все объяснения
    все_объяснения = ' '.join(df[explanation_col].dropna().astype(str))

    # Разбиваем на слова и фильтруем
    слова = [word.strip() for word in все_объяснения.split() if len(word.strip()) > 2]

    # Анализ частотности
    частотность = Counter(слова)

    print("Топ-15 наиболее частых характеристик в объяснениях:")
    for слово, count in частотность.most_common(15):
        print(f"  {слово}: {count}")

    # Анализ по ключевым категориям
    категории = {
        'Развернутый': 'Развернутый ответ',
        'смысловое': 'Смысловое соответствие',
        'соответствие': 'Смысловое соответствие',
        'Хорошая': 'Хорошая структура',
        'структура': 'Хорошая структура',
        'лексика': 'Разнообразная лексика',
        'Высокий': 'Высокий балл',
        'балл': 'Высокий балл',
        'описание': 'Подробное описание',
        'личный': 'Личный опыт',
        'покрытие': 'Покрытие вопросов'
    }

    print(f"\nСТАТИСТИКА ПО КАТЕГОРИЯМ:")
    for ключ, описание in категориями.items():
        count = sum(1 for слово in слова if ключ in слово)
        if count > 0:
            print(f"  {описание}: {count}")


def performance_by_question_type(df):
    """Анализ производительности по типам вопросов"""

    print("\n" + "=" * 40)
    print("АНАЛИЗ ПО ТИПАМ ВОПРОСОВ")
    print("=" * 40)

    вопросы_статистика = df.groupby('№ вопроса').agg({
        'Оценка экзаменатора': ['mean', 'std', 'count'],
        'pred_score': ['mean', 'std'],
        'abs_разница': 'mean',
        'разница': 'mean'
    }).round(3)

    # Переименовываем колонки для удобства
    вопросы_статистика.columns = ['чел_среднее', 'чел_стд', 'количество',
                                  'ai_среднее', 'ai_стд', 'ср_абс_разница', 'ср_разница']

    вопросы_статистика['расхождение'] = abs(вопросы_статистика['ср_разница'])

    print("СТАТИСТИКА ПО ВОПРОСАМ:")
    print("-" * 80)
    print(f"{'Вопрос':<6} {'Чел.ср':<8} {'AI ср':<8} {'Разн.':<8} {'Кол-во':<8} {'Описание'}")
    print("-" * 80)

    for вопрос, row in вопросы_статистика.iterrows():
        разница_знак = "+" if row['ср_разница'] > 0 else ""
        print(f"{вопрос:<6} {row['чел_среднее']:<8} {row['ai_среднее']:<8} "
              f"{разница_знак}{row['ср_разница']:<7} {int(row['количество']):<8} ", end="")

        if row['расхождение'] > 0.3:
            print("ВНИМАНИЕ: большое расхождение")
        elif row['расхождение'] > 0.1:
            print("Умеренное расхождение")
        else:
            print("Хорошее соответствие")


def save_detailed_analysis(df):
    """Сохранение детального анализа в файл"""

    print("\n" + "=" * 40)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 40)

    # Создаем копию с анализом
    df_analysis = df.copy()
    df_analysis['разница_ai_человек'] = df_analysis['pred_score'] - df_analysis['Оценка экзаменатора']
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
        with pd.ExcelWriter('detailed_analysis_pro.xlsx', engine='openpyxl') as writer:
            # Все данные
            df_analysis.to_excel(writer, sheet_name='Все_данные_с_анализом', index=False)

            # Сводная таблица по вопросам
            сводная = df_analysis.groupby('№ вопроса').agg({
                'Оценка экзаменатора': ['mean', 'std', 'min', 'max'],
                'pred_score': ['mean', 'std', 'min', 'max'],
                'abs_разница': ['mean', 'max'],
                'разница_ai_человек': 'mean',
                'Id экзамена': 'count'
            }).round(3)
            сводная.to_excel(writer, sheet_name='Сводка_по_вопросам')

            # Наибольшие расхождения
            большие_расхождения = df_analysis.nlargest(20, 'abs_разница')[
                ['Id экзамена', '№ вопроса', 'Оценка экзаменатора',
                 'pred_score', 'разница_ai_человек', 'abs_разница']
            ]
            большие_расхождения.to_excel(writer, sheet_name='Наибольшие_расхождения', index=False)

            # Статистика по качеству согласования
            качество_стат = df_analysis['качество_согласования'].value_counts()
            качество_стат.to_excel(writer, sheet_name='Качество_согласования')

        print("Детальный анализ сохранен в 'detailed_analysis_pro.xlsx'")

    except Exception as e:
        print(f"Не удалось сохранить Excel, сохраняем в CSV: {e}")
        df_analysis.to_csv('detailed_analysis_pro.csv', index=False, encoding='utf-8')
        print("Детальный анализ сохранен в 'detailed_analysis_pro.csv'")


def generate_summary_report(df):
    """Генерация итогового отчета"""

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 60)

    корреляция = df[['Оценка экзаменатора', 'pred_score']].corr().iloc[0, 1]
    ср_разница = df['abs_разница'].mean()

    print(f"\nОБЩАЯ СТАТИСТИКА:")
    print(f"  Всего ответов: {len(df)}")
    print(f"  Корреляция AI-Человек: {корреляция:.3f}")
    print(f"  Средняя абсолютная разница: {ср_разница:.3f}")

    # Оценка качества
    if корреляция > 0.8 and ср_разница < 0.2:
        оценка = "ОТЛИЧНОЕ"
    elif корреляция > 0.6 and ср_разница < 0.3:
        оценка = "ХОРОШЕЕ"
    elif корреляция > 0.4 and ср_разница < 0.4:
        оценка = "УДОВЛЕТВОРИТЕЛЬНОЕ"
    else:
        оценка = "НИЗКОЕ"

    print(f"\nОЦЕНКА КАЧЕСТВА СИСТЕМЫ: {оценка}")

    # Рекомендации
    print(f"\nРЕКОМЕНДАЦИИ:")
    if ср_разница > 0.3:
        print("  Проанализировать систематические ошибки в оценках")
    if корреляция < 0.6:
        print("  Улучшить согласованность с человеческими оценками")

    # Лучшие и худшие вопросы
    вопросы_стат = df.groupby('№ вопроса')['abs_разница'].mean().sort_values()
    лучший_вопрос = вопросы_стат.index[0]
    худший_вопрос = вопросы_стат.index[-1]

    print(f"\nЛУЧШИЙ ВОПРОС ПО СОГЛАСОВАННОСТИ: №{лучший_вопрос} (разница: {вопросы_стат.iloc[0]:.3f})")
    print(f"ХУДШИЙ ВОПРОС ПО СОГЛАСОВАННОСТИ: №{худший_вопрос} (разница: {вопросы_стат.iloc[-1]:.3f})")


def main():
    """Основная функция"""

    try:
        # Загрузка данных
        df = load_and_analyze_data()

        if df is None:
            return

        # Проверка необходимых колонок
        required_columns = ['Оценка экзаменатора', 'pred_score', '№ вопроса']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"ОШИБКА: Отсутствуют колонки: {missing_columns}")
            return

        # Выполнение анализа
        basic_statistics(df)
        calculate_correlations(df)
        create_visualizations(df)
        analyze_extreme_cases(df)
        analyze_explanations(df)
        performance_by_question_type(df)
        save_detailed_analysis(df)
        generate_summary_report(df)

        print("\n" + "=" * 60)
        print("АНАЛИЗ ЗАВЕРШЕН!")
        print("=" * 60)
        print("\nСОЗДАННЫЕ ФАЙЛЫ:")
        print("  graphs/scatter_comparison_pro.png - сравнение оценок")
        print("  graphs/difference_histogram_pro.png - распределение разниц")
        print("  graphs/question_boxplot_pro.png - оценки по вопросам")
        print("  detailed_analysis_pro.xlsx - детальный отчет")

    except FileNotFoundError:
        print("ОШИБКА: Файл 'small.csv' не найден в текущей директории")
    except Exception as e:
        print(f"ОШИБКА при выполнении анализа: {str(e)}")


if __name__ == "__main__":
    main()