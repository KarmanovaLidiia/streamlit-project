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


def create_test_data():
    """Создание тестовых данных"""

    test_data = """Id экзамена;Id вопроса;№ вопроса;Текст вопроса;Оценка экзаменатора;Транскрибация ответа;pred_score;объяснение_оценки
3373871;30625752;1;"<p>Добро пожаловать на экзамен!</p>";1;"Экзаменатор: Начните диалог. Тестируемый: Здравствуйте, я хотел бы извиниться, что не смогу прийти на день рождения. Что бы вы хотели в подарок?";0.99;"🟢 Развернутый ответ | ✅ Высокое смысловое соответствие | 📊 Хорошая структура ответа | 💬 Разнообразная лексика | ⭐ Высокий балл"
3373871;30625753;2;"<p>Расскажите о вашем жилье</p>";2;"Экзаменатор: Вы живёте в квартире или доме? Тестируемый: Я живу в квартире в центре города. Это трёхкомнатная квартира с балконом. Квартира новая, построена в 2020 году.";1.62;"🟢 Развернутый ответ | ✅ Высокое смысловое соответствие | 📊 Хорошая структура ответа | 🏠 Подробное описание | ⭐ Высокий балл"
3373872;30625790;1;"<p>Начните диалог о работе</p>";1;"Экзаменатор: Узнайте о требованиях к работе. Тестируемый: Здравствуйте, я увидел ваше объявление о вакансии. Какие требования к соискателю? Какие документы нужны?";0.87;"🟢 Развернутый ответ | ⚠️ Умеренное смысловое соответствие | 📊 Хорошая структура ответа | 💬 Разнообразная лексика | ⭐ Высокий балл"
3373872;30625791;2;"<p>Опишите ваше жилье</p>";1;"Экзаменатор: Расскажите о вашей квартире. Тестируемый: У меня квартира. Она хорошая. Три комнаты.";0.45;"📉 Мало предложений | ❌ Низкое смысловое соответствие | 📊 Хорошая структура ответа"
3373873;30625828;1;"<p>Оформление документов</p>";2;"Экзаменатор: Объясните ситуацию в миграционной службе. Тестируемый: Здравствуйте, мне нужно оформить миграционную карту. Я приехал две недели назад. Можете дать мне бланк для заполнения?";1.85;"🟢 Развернутый ответ | ✅ Высокое смысловое соответствие | 📊 Хорошая структура ответа | 💬 Разнообразная лексика | ⭐ Высокий балл"
3373873;30625829;2;"<p>Ваши любимые фильмы</p>";1;"Экзаменатор: Какие фильмы вы любите? Тестируемый: Я смотрю фантастику и детективы. Люблю новые цветные фильмы. Мой любимый фильм - Интерстеллар, он о космосе и времени.";1.15;"🟢 Развернутый ответ | ⚠️ Умеренное смысловое соответствие | 📊 Хорошая структура ответа | 💬 Разнообразная лексика"
3373874;30625866;3;"<p>Опишите картинку</p>";2;"Экзаменатор: Что изображено на картинке? Тестируемый: На картинке изображена семья в парке. Дети играют в мяч, родители сидят на скамейке. Яркий солнечный день, лето.";1.92;"🟢 Развернутый ответ | ✅ Высокое смысловое соответствие | 🎨 Есть вступление с описанием картинки | 👤 Есть личный опыт | ⭐ Высокий балл"
3373874;30625867;4;"<p>Расскажите о хобби</p>";1;"Экзаменатор: Чем увлекаетесь? Тестируемый: Я читаю книги. Иногда смотрю фильмы.";0.35;"📉 Мало предложений | ❌ Низкое смысловое соответствие | 📊 Хорошая структура ответа"
3373875;30625904;1;"<p>Ситуация в больнице</p>";1;"Экзаменатор: Узнайте о приеме врача. Тестируемый: Здравствуйте, мне нужно записаться к терапевту на обследование. Когда принимает врач и какие документы нужны?";0.95;"🟢 Развернутый ответ | ✅ Высокое смысловое соответствие | 📊 Хорошая структура ответа | 💬 Разнообразная лексика | ⭐ Высокий балл"
3373875;30625905;2;"<p>Кулинарные предпочтения</p>";2;"Экзаменатор: Какая ваша любимая кухня? Тестируемый: Я очень люблю итальянскую кухню, особенно пасту и пиццу. Также нравится японская кухня - суши и роллы. Люблю готовить сам, особенно выпечку.";1.78;"🟢 Развернутый ответ | ✅ Высокое смысловое соответствие | 📊 Хорошая структура ответа | 🏠 Подробное описание | ⭐ Высокий балл"
"""

    # Сохраняем тестовые данные в файл
    with open('test_data.csv', 'w', encoding='utf-8') as f:
        f.write(test_data)

    print("✅ Тестовый файл 'test_data.csv' создан успешно")
    return True


def load_and_analyze_data():
    """Загрузка тестовых данных"""

    file_path = 'test_data.csv'

    try:
        df = pd.read_csv(file_path, encoding='utf-8', delimiter=';')
        print("✅ Тестовый файл загружен успешно")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
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

    # Направление разниц
    завышение = len(df[df['разница'] > 0])
    занижение = len(df[df['разница'] < 0])
    совпадение = len(df[df['разница'] == 0])

    print(f"\nНАПРАВЛЕНИЕ РАЗНИЦ:")
    print(f"  AI завышает: {завышение} ({завышение / len(df) * 100:.1f}%)")
    print(f"  AI занижает: {занижение} ({занижение / len(df) * 100:.1f}%)")
    print(f"  Полное совпадение: {совпадение} ({совпадение / len(df) * 100:.1f}%)")


def create_visualizations(df):
    """Создание графиков"""

    print("\n" + "=" * 40)
    print("СОЗДАНИЕ ГРАФИКОВ")
    print("=" * 40)

    os.makedirs('graphs', exist_ok=True)

    # 1. Scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Оценка экзаменатора'], df['pred_score'],
                          c=df['abs_разница'], cmap='viridis', alpha=0.7, s=80)
    plt.colorbar(scatter, label='Абсолютная разница')
    plt.plot([0, 2], [0, 2], 'r--', alpha=0.5, label='Идеальное соответствие')
    plt.xlabel('Оценка экзаменатора', fontsize=12)
    plt.ylabel('AI оценка (pred_score)', fontsize=12)
    plt.title('Сравнение человеческой и AI оценки\n(цвет показывает величину расхождения)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks([1, 2])
    plt.savefig('graphs/test_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Гистограмма разниц
    plt.figure(figsize=(12, 6))
    plt.hist(df['разница'], bins=15, alpha=0.7, edgecolor='black', color='skyblue')
    plt.xlabel('Разница (AI - Человек)', fontsize=12)
    plt.ylabel('Количество ответов', fontsize=12)
    plt.title('Распределение разниц между AI и человеческими оценками', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Нулевая разница')
    plt.axvline(x=df['разница'].mean(), color='orange', linestyle='--',
                alpha=0.8, label=f'Средняя разница: {df["разница"].mean():.3f}')
    plt.legend()
    plt.savefig('graphs/test_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Box plot по вопросам
    plt.figure(figsize=(12, 6))
    box_data = [df[df['№ вопроса'] == question]['pred_score'].values
                for question in sorted(df['№ вопроса'].unique())]

    box_plot = plt.boxplot(box_data, labels=sorted(df['№ вопроса'].unique()),
                           patch_artist=True)

    # Раскрашиваем boxplot
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Распределение AI оценок по номерам вопросов', fontsize=14)
    plt.xlabel('Номер вопроса', fontsize=12)
    plt.ylabel('AI оценка (pred_score)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('graphs/test_boxplot.png', dpi=300, bbox_inches='tight')
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


def save_detailed_analysis(df):
    """Сохранение детального анализа"""

    print("\n" + "=" * 40)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 40)

    # Создаем копию с анализом
    df_analysis = df.copy()

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
            df_analysis.to_excel(writer, sheet_name='Все_данные_с_анализом', index=False)
        print("✅ Детальный анализ сохранен в 'detailed_analysis.xlsx'")
    except Exception as e:
        print(f"⚠️ Не удалось сохранить Excel: {e}")


def main():
    """Основная функция"""

    print("Создание тестовых данных...")
    if not create_test_data():
        return

    df = load_and_analyze_data()
    if df is None:
        return

    basic_statistics(df)
    calculate_correlations(df)
    create_visualizations(df)
    analyze_explanations(df)
    save_detailed_analysis(df)

    print("\n" + "=" * 60)
    print("✅ ТЕСТОВЫЙ АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 60)
    print("📊 Созданные файлы:")
    print("   • test_data.csv - тестовые данные")
    print("   • graphs/test_scatter.png - сравнение оценок")
    print("   • graphs/test_histogram.png - распределение разниц")
    print("   • graphs/test_boxplot.png - оценки по вопросам")
    print("   • detailed_analysis.xlsx - детальный отчет")


if __name__ == "__main__":
    main()