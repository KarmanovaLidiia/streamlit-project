# src/evaluate_local.py
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw" / "Данные для кейса.csv"
PRED = ROOT / "data" / "processed" / "predicted.csv"

def main():
    # читаем сырой (для реальной оценки берём «живую» колонку оценок)
    raw = pd.read_csv(RAW, encoding="utf-8-sig", sep=";")
    pred = pd.read_csv(PRED, encoding="utf-8-sig")

    # аккуратно сопоставим по двум id, если есть; иначе по порядку строк
    cols = list(raw.columns)
    if {"Id экзамена","Id вопроса"}.issubset(cols):
        key = ["Id экзамена","Id вопроса"]
        df = raw[key + ["Оценка экзаменатора"]].merge(
            pred[key + ["pred_score"]], on=key, how="inner"
        )
    else:
        df = pd.DataFrame({
            "Оценка экзаменатора": raw["Оценка экзаменатора"],
            "pred_score": pred["pred_score"],
            "№ вопроса": raw["№ вопроса"] if "№ вопроса" in raw.columns else None
        })

    df = df.rename(columns={"№ вопроса":"question_number"})
    df = df.dropna(subset=["Оценка экзаменатора"]).copy()
    df["err"] = (df["Оценка экзаменатора"] - df["pred_score"]).abs()

    overall = df["err"].mean()
    print(f"MAE (вся выборка): {overall:.3f}")

    if "question_number" in df.columns and df["question_number"].notna().any():
        for q in [1,2,3,4]:
            mae_q = df.loc[df["question_number"]==q, "err"].mean()
            print(f"  Q{q}: MAE={mae_q:.3f}")

if __name__ == "__main__":
    main()
