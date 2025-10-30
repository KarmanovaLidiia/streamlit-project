# src/prepare_data.py
from pathlib import Path
import pandas as pd
import re

from src.data_cleaning import prepare_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH  = PROJECT_ROOT / "data" / "raw" / "Данные для кейса.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "clean_data.csv"

ENCODINGS = ["utf-8-sig", "utf-8", "cp1251", "windows-1251", "koi8-r", "iso-8859-5"]
SEPARATORS = [",", ";", "\t", "|"]

def has_cyrillic(s: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", s))

def smart_read_csv(path: Path) -> tuple[pd.DataFrame, str, str]:
    best = None
    for enc in ENCODINGS:
        for sep in SEPARATORS:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                cols_joined = " ".join(map(str, df.columns))
                if df.shape[1] > 1 and has_cyrillic(cols_joined):
                    print(f"[i] Выбраны encoding='{enc}', sep='{sep}'")
                    return df, enc, sep
                # запомним хоть какой-то валидный вариант на случай без кириллицы
                if best is None and df.shape[1] > 1:
                    best = (df, enc, sep)
            except Exception:
                pass
    if best:
        print("[!] Кириллица в заголовках не обнаружена, берём первый валидный вариант.")
        return best
    raise RuntimeError("Не удалось прочитать CSV: перепроверьте файл и кодировку.")

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {INPUT_PATH}")

    df, enc, sep = smart_read_csv(INPUT_PATH)
    print(f"[i] колонки: {list(df.columns)}")
    print(f"[i] размер: {df.shape}")

    clean_df = prepare_dataframe(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ Готово: {OUTPUT_PATH}")
    print(clean_df.head(3))

if __name__ == "__main__":
    main()
