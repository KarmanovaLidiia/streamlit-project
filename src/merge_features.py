# src/merge_features.py
from pathlib import Path
import pandas as pd

from src.semantic_features import add_semantic_similarity
from src.features_q4 import q4_slot_features  # <- берём функцию с признаками для Q4

ROOT = Path(__file__).resolve().parents[1]
FEAT_PATH = ROOT / "data" / "processed" / "features_baseline.csv"
OUT_PATH  = ROOT / "data" / "processed" / "features_with_semantics_q4.csv"

def main():
    print(f"🔹 Читаю базовые фичи: {FEAT_PATH}", flush=True)
    df = pd.read_csv(FEAT_PATH, encoding="utf-8-sig")

    print("🔹 Добавляю семантическую близость (ruSBERT)...", flush=True)
    df = add_semantic_similarity(df, batch_size=64)  # будет использовать кэш

    print("🔹 Добавляю rule-based признаки для Q4...", flush=True)
    df = q4_slot_features(df)

    print(f"🔹 Сохраняю итог: {OUT_PATH}", flush=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("✅ Готово. Превью:", flush=True)
    print(df[['question_number','semantic_sim','q4_slots_covered','q4_answered_personal','score']].head(), flush=True)

if __name__ == "__main__":
    main()
