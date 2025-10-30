# src/train_baseline.py
from pathlib import Path
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "features_with_semantics.csv"
MODEL_PATH = ROOT / "models" / "catboost_baseline.cbm"

def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # Разделяем признаки и таргет
    target = df["score"]
    features = df.drop(columns=["score", "question_text", "answer_text"])

    # Кросс-валидация
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
        X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

        model = CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function="MAE",
            verbose=False,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        maes.append(mae)
        print(f"Fold {fold+1}: MAE = {mae:.3f}")

    print(f"🔹 Среднее MAE: {np.mean(maes):.3f}")
    model.save_model(MODEL_PATH)
    print(f"✅ Модель сохранена: {MODEL_PATH}")

if __name__ == "__main__":
    main()
