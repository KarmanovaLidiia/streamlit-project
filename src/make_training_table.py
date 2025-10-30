# src/make_training_table.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.data_cleaning import prepare_dataframe
from src.features import build_baseline_features          # базовые признаки (len, ttr, punctuation, etc.)
from src.semantic_features import add_semantic_similarity # semantic_sim (ruSBERT + cache)
from src.features_q4 import add_q4_features               # rule-based фичи для Q4

RAW = Path("data/raw/Данные для кейса.csv")
CLEAN = Path("data/processed/clean_data.csv")
OUT = Path("data/processed/features_with_semantics_q4.csv")

def read_input() -> pd.DataFrame:
    # стараемся воспроизвести ту же «устойчивую» загрузку, что и в predict
    tries = [
        ("utf-8-sig", ";"),
        ("utf-8", ";"),
        ("utf-8-sig", ","),
        ("utf-8", ","),
        ("utf-8-sig", None),
        ("utf-8", None),
    ]
    last_err = None
    for enc, sep in tries:
        try:
            if sep is None:
                df = pd.read_csv(RAW, encoding=enc, sep=None, engine="python")
            else:
                df = pd.read_csv(RAW, encoding=enc, sep=sep)
            print(f"[i] CSV прочитан с encoding='{enc}', sep='{sep or 'auto'}'")
            return df
        except Exception as e:
            last_err = e
    raise last_err

def main():
    # 1) читаем сырой CSV → приводим к стандартной схеме
    if CLEAN.exists():
        print(f"[i] Использую уже подготовленный clean: {CLEAN}")
        df_clean = pd.read_csv(CLEAN, encoding="utf-8-sig")
    else:
        df_raw = read_input()
        df_clean = prepare_dataframe(df_raw)
        CLEAN.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(CLEAN, index=False, encoding="utf-8-sig")
        print(f"✅ Сохранён clean: {CLEAN}")

    # 2) базовые признаки
    feats = build_baseline_features(df_clean)

    # 3) семантическая близость (кэш ruSBERT)
    print("🔹 Вычисляем semantic_sim (ruSBERT + cache)...")
    feats = add_semantic_similarity(feats, batch_size=64)

    # 4) rule-based признаки Q4
    print("🔹 Добавляю rule-based признаки Q4...")
    feats = add_q4_features(feats)

    # 5) сохраняем обучающую таблицу
    OUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"✅ Готово: {OUT}")
    print("Превью:")
    print(feats.head())

if __name__ == "__main__":
    main()
