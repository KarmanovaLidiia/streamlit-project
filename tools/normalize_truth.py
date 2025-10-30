import sys, pandas as pd

if len(sys.argv) < 3:
    print("Usage: python tools/normalize_truth.py <input_csv> <output_csv>")
    sys.exit(1)

inp, outp = sys.argv[1], sys.argv[2]

# читаем с авто-детектом разделителя (engine='python' умеет sniff)
df = pd.read_csv(inp, sep=None, engine="python", encoding="utf-8-sig")

# убедимся, что ключевые колонки на месте
need = ['Id экзамена','Id вопроса','№ вопроса','Оценка экзаменатора']
missing = [c for c in need if c not in df.columns]
if missing:
    raise RuntimeError(f"Нет колонок {missing}. Найдены: {list(df.columns)}")

# сохраняем в ;-CSV в utf-8-sig (как во всех наших пайплайнах)
df.to_csv(outp, sep=';', index=False, encoding='utf-8-sig')
print(f"OK -> {outp} | rows={len(df)} | cols={len(df.columns)}")
