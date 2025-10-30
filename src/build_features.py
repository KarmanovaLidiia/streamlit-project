from pathlib import Path
import pandas as pd
from src.features import build_basic_features

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "clean_data.csv"
OUT   = ROOT / "data" / "processed" / "features_baseline.csv"

def main():
    if not CLEAN.exists():
        raise FileNotFoundError(f"Не найден {CLEAN}")
    df = pd.read_csv(CLEAN, encoding="utf-8-sig")
    feats = build_basic_features(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("✅ Сохранено:", OUT)
    print(feats[[
        "question_number","ans_len_chars","ans_len_words","ans_n_sents",
        "ans_avg_sent_len","ans_ttr","ans_short_sent_rt","ans_punct_rt",
        "q_len_words","score"
    ]].head(5))

if __name__ == "__main__":
    main()
