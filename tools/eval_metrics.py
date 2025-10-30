# tools/eval_metrics.py
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score

INP = Path("predicted_big_final.csv")
OUT = Path("reports/metrics.md")
OUT.parent.mkdir(parents=True, exist_ok=True)

def norm_name(s: str) -> str:
    # унифицируем имена: нижний регистр, убираем лишние пробелы
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def coerce_number(x: str | float | int) -> float | None:
    """
    Приводим строку к числу:
    - обрезаем пробелы
    - убираем мусор вроде ';;'
    - меняем запятую на точку
    - оставляем только допустимые символы 0-9 . - e
    """
    if pd.isna(x):
        return None
    s = str(x).strip()

    # часто встречавшийся мусор вида "0.999...;;0.0" — берём первое число слева
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?", s)
    if m:
        s = m.group(0)

    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

# --- читаем CSV ---
df = pd.read_csv(INP, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)

# нормализуем имена колонок
renamed = {c: norm_name(c) for c in df.columns}
df.columns = [renamed[c] for c in df.columns]

# пытаемся найти нужные колонки по нормализованным именам
# "оценка экзаменатора" и "pred_score"
true_candidates = [c for c in df.columns if "оценка" in c and "экзамен" in c]
pred_candidates = [c for c in df.columns if "pred_score" in c or ("pred" in c and "score" in c)]

if not true_candidates:
    print("⚠️ Не найден столбец с истинной оценкой (например, 'Оценка экзаменатора').")
if not pred_candidates:
    print("⚠️ Не найден столбец с предсказанием (например, 'pred_score').")

true_col = true_candidates[0] if true_candidates else None
pred_col = pred_candidates[0] if pred_candidates else None

# Диагностика: покажем первые 3 значения как есть
if true_col:
    print("Примеры 'истины':", list(df[true_col].head(3)))
if pred_col:
    print("Примеры 'предикта':", list(df[pred_col].head(3)))

# Если нет хотя бы одного из столбцов — пишем отчёт-заглушку и выходим
if not true_col or not pred_col:
    OUT.write_text("# Метрики качества\nНе удалось найти нужные столбцы.\n", encoding="utf-8")
    print("Отчёт записан:", OUT)
    raise SystemExit(0)

# Приводим к числам robust-способом
y_true_raw = df[true_col].apply(coerce_number)
y_pred_raw = df[pred_col].apply(coerce_number)

# Маска валидных строк
mask = y_true_raw.notna() & y_pred_raw.notna()
n_all = len(df)
n_ok = int(mask.sum())

print(f"Строк всего: {n_all} | пригодных для расчёта: {n_ok}")

# Если нет ни одной валидной строки — фиксируем отчёт и выходим мягко
if n_ok == 0:
    OUT.write_text(
        "# Метрики качества\n"
        f"- Строк всего: {n_all}\n"
        f"- Пригодных для расчёта: {n_ok}\n"
        "- Похоже, в файле отсутствуют валидные пары (истина/предикт). Проверьте данные.\n",
        encoding="utf-8",
    )
    print("❌ Нет валидных пар значений. Отчёт-заглушка записан:", OUT)
    raise SystemExit(0)

# Подмножество валидных
dfv = df[mask].copy()
y_true = y_true_raw[mask].astype(float)
y_pred = y_pred_raw[mask].astype(float)

# Бинарная/трёхуровневая шкала (клип и округление)
y_pred_rounded = y_pred.round().clip(0, 2)

# MAE
mae_all = mean_absolute_error(y_true, y_pred)

# По номеру вопроса, если он есть
q_col_candidates = [c for c in df.columns if "№ вопроса" in c or "номер вопроса" in c or norm_name(c) == "№ вопроса"]
mae_per_q = []
if q_col_candidates:
    qcol = q_col_candidates[0]
    # аккуратно приводим к числу
    qnum = dfv[qcol].apply(coerce_number)
    dfv = dfv[qcol].to_frame().assign(qnum=qnum)
    # индексы валидных вопросов
    qnum = qnum.dropna().astype(int)
    # Нам нужны те же индексы и для y_true/y_pred
    idx = qnum.index
    for q in sorted(qnum.unique()):
        ii = idx[qnum[idx] == q]
        mae_q = mean_absolute_error(y_true.loc[ii], y_pred.loc[ii])
        mae_per_q.append((int(q), mae_q))

# Accuracy по округлённым
acc_all = accuracy_score(y_true, y_pred_rounded)

# Распределения
dist_true = y_true.round().clip(0, 2).value_counts().sort_index()
dist_pred = y_pred_rounded.value_counts().sort_index()

# Пишем отчёт
lines = []
lines.append("# Метрики качества (predicted_big_final.csv)\n")
lines.append(f"- Строк всего: {n_all}")
lines.append(f"- Строк использовано: {n_ok}")
lines.append(f"- MAE (всё): **{mae_all:.4f}**")
if mae_per_q:
    for q, m in mae_per_q:
        lines.append(f"- MAE (№ вопроса = {q}): **{m:.4f}**")
lines.append(f"- Accuracy (округлённые 0–2): **{acc_all:.4f}**\n")
lines.append("## Распределение оценок (после округления)")
lines.append("**Истинные:**\n```\n" + dist_true.to_string() + "\n```")
lines.append("**Предсказанные:**\n```\n" + dist_pred.to_string() + "\n```")

OUT.write_text("\n".join(lines), encoding="utf-8")
print("OK ->", OUT)
print(f"MAE={mae_all:.4f} | ACC={acc_all:.4f}")
