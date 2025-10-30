import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Пути
pred_path = Path("data/processed/predicted.csv")

# Читаем CSV
df = pd.read_csv(pred_path, encoding="utf-8-sig")


# Удаляем HTML и NBSP, если есть
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r"<[^>]*>", "", text)  # убираем HTML-теги
        text = text.replace("&nbsp;", " ")  # заменяем HTML NBSP
        text = text.replace(" ", " ")  # убираем неразрывные пробелы
    return text


df = df.applymap(clean_text)

# Проверяем нужные колонки
if "Оценка экзаменатора" in df.columns and "pred_score" in df.columns:
    df["abs_error"] = (df["pred_score"] - df["Оценка экзаменатора"]).abs()

    print(f"Средняя ошибка (MAE): {df['abs_error'].mean():.3f}")
    print(f"Максимальная ошибка: {df['abs_error'].max():.2f}")

    # Строим гистограмму ошибок
    plt.figure(figsize=(8, 5))
    plt.hist(df["abs_error"], bins=10, color='skyblue', edgecolor='black')
    plt.title("Распределение ошибок предсказаний (|pred - human|)")
    plt.xlabel("Абсолютная ошибка")
    plt.ylabel("Количество ответов")
    plt.grid(alpha=0.3)
    plt.show()
else:
    print("⚠️ Нет нужных колонок ('Оценка экзаменатора' и 'pred_score') в predicted.csv")
