# src/add_q4_features.py
from pathlib import Path
import pandas as pd
from src.features_q4 import q4_slot_features

ROOT = Path(__file__).resolve().parents[1]
INP  = ROOT / "data" / "processed" / "features_with_semantics.csv"   # уже есть
OUT  = ROOT / "data" / "processed" / "features_with_semantics_q4.csv"

def main():
    df = pd.read_csv(INP, encoding="utf-8-sig")
    df2 = q4_slot_features(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("✅ Сохранено:", OUT)
    print(df2[[
        "question_number","semantic_sim",
        "q4_slots_covered","q4_answered_personal","q4_non_cyr_ratio","score"
    ]].head())

if __name__ == "__main__":
    main()
