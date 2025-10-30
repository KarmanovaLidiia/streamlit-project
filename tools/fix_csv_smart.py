# tools/fix_csv_smart.py
from __future__ import annotations
import re
import sys
import pandas as pd
from pathlib import Path

INP = Path("predicted_big.csv")              # вход
OUT = Path("predicted_big_clean2.csv")       # выход

EXPECTED_COLS = [
    "Id экзамена",
    "Id вопроса",
    "№ вопроса",
    "Текст вопроса",
    "Картинка из вопроса",
    "Оценка экзаменатора",
    "Транскрибация ответа",
    "Ссылка на оригинальный файл запис",
    "pred_score",
]

if not INP.exists():
    print(f"Файл не найден: {INP}")
    sys.exit(1)

# 1) читаем как «;», позволяем странные строки, не роняемся
df = pd.read_csv(
    INP,
    sep=";",
    engine="python",
    encoding="utf-8-sig",
    dtype=str,                  # всё как строки, потом почистим
    on_bad_lines="skip"
)

# 2) Нормализуем имена колонок (обрежем пробелы)
df.columns = [c.strip() for c in df.columns]

# Если вдруг колонок больше/меньше — попытаемся привести к ожидаемым,
# оставляя только нужные и создавая отсутствующие пустыми.
for col in EXPECTED_COLS:
    if col not in df.columns:
        df[col] = ""

df = df[EXPECTED_COLS]  # упорядочим

# 3) Удалим «дубли заголовка», случайно попавшие как строки данных:
mask_dup_hdr = (
    (df["Id экзамена"].astype(str).str.strip() == "Id экзамена")
    | (df["pred_score"].astype(str).str.contains(r"\bpred_score\b", na=False))
)
df = df.loc[~mask_dup_hdr].copy()

# 4) Чистим pred_score.
#    Берём САМУЮ ПЕРВУЮ числовую подпоследовательность (с точкой/запятой) из строки,
#    меняем запятую на точку и приводим к float.
def clean_first_float(s: str) -> float | None:
    if pd.isna(s):
        return None
    s = str(s)
    # иногда встречается "...;;0.0" — нас интересует первое число
    m = re.search(r"[+-]?\d+(?:[.,]\d+)?", s)
    if not m:
        return None
    val = m.group(0).replace(",", ".")
    try:
        return float(val)
    except:
        return None

df["pred_score"] = df["pred_score"].apply(clean_first_float)

# 5) Чистим «Оценка экзаменатора» (аналогично, но это может быть пусто):
def clean_grade(s: str) -> float | None:
    if pd.isna(s):
        return None
    s = str(s)
    m = re.search(r"[+-]?\d+(?:[.,]\d+)?", s)
    if not m:
        return None
    val = m.group(0).replace(",", ".")
    try:
        return float(val)
    except:
        return None

df["Оценка экзаменатора"] = df["Оценка экзаменатора"].apply(clean_grade)

# 6) Удалим полностью пустые строки (все поля пустые)
df = df.dropna(how="all").copy()

# 7) Сохраняем «как есть» с разделителем «;»
df.to_csv(OUT, sep=";", index=False, encoding="utf-8-sig")

print(f"OK -> {OUT.name} | rows: {len(df)} | cols: {df.shape[1]}")
print("columns:", "; ".join(df.columns))
