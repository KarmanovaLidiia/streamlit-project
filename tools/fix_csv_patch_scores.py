# tools/fix_csv_patch_scores.py
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

RAW = Path("predicted_big.csv")           # сырой файл из API (грязный)
CLEAN = Path("predicted_big_clean2.csv")  # наш уже очищенный 9-колоночный
OUT = Path("predicted_big_final.csv")

if not RAW.exists():
    raise SystemExit(f"Не найден {RAW}")
if not CLEAN.exists():
    raise SystemExit(f"Не найден {CLEAN}")

# 1) читаем очищенный (ровный) — будем туда подставлять исправленные pred_score
df = pd.read_csv(CLEAN, sep=";", encoding="utf-8-sig", dtype=str)
n = len(df)

# 2) выдёргиваем pred_score из СЫРОГО файла построчно
lines = RAW.read_text(encoding="utf-8-sig", errors="ignore").splitlines()

# обычно 1-я строка — заголовок; берём столько строк данных, сколько в чистом df
# если в RAW заголовков несколько, просто игнорируем первые, пока не совпадёт длина
float_pat = re.compile(r"[+-]?\d+(?:[.,]\d+)?")

def pick_score_from_text(s: str) -> float | None:
    nums = [x.replace(",", ".") for x in float_pat.findall(s)]
    vals = []
    for x in nums:
        try:
            vals.append(float(x))
        except Exception:
            pass
    if not vals:
        return None
    cand = [v for v in vals if 0.0 <= v <= 2.0]
    if cand:
        nz = [v for v in cand if v > 0]
        return max(nz) if nz else 0.0
    return min(vals, key=lambda v: abs(v))

# найдём «окно» данных в RAW такой длины, чтобы извлечь n оценок
scores: list[float] = []
start_idx = 0
while start_idx < len(lines) and len(scores) < n:
    scores_tmp: list[float] = []
    for ln in lines[start_idx:start_idx + n]:
        v = pick_score_from_text(ln)
        scores_tmp.append(0.0 if v is None else float(v))
    if len(scores_tmp) == n:
        scores = scores_tmp
        break
    start_idx += 1

if len(scores) != n:
    raise SystemExit(f"Не удалось выровнять длину: RAW={len(lines)} / CLEAN={n}")

# 3) подставляем и сохраняем
df["pred_score"] = pd.Series(pd.to_numeric(scores, errors="coerce")).fillna(0.0).astype(float)
df.to_csv(OUT, sep=";", index=False, encoding="utf-8-sig")

print(f"OK -> {OUT.name} | rows: {len(df)} | cols: {df.shape[1]}")
print("pred_score sample:", df["pred_score"].head(5).tolist())
print("pred_score stats: min=", df["pred_score"].min(), "max=", df["pred_score"].max())
