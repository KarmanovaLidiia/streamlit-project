# src/train_qmodels.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool, cv

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# ВАЖНО: учимся именно на файле с семантикой и фичами Q4
FEATURES_CSV = DATA / "features_with_semantics_q4.csv"

EXCLUDE_COLS = {
    "question_number", "question_text", "answer_text", "score"
}

# Границы оценок по каждому вопросу (для клиппинга и sanity-check)
BOUNDS = {
    1: (0.0, 1.0),
    2: (0.0, 2.0),
    3: (0.0, 1.0),
    4: (0.0, 2.0),
}

def load_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV, encoding="utf-8-sig")
    # sanity: приведём типы
    df["question_number"] = df["question_number"].astype(int)
    df["score"] = df["score"].astype(float)
    return df

def pick_feature_columns(df: pd.DataFrame) -> list[str]:
    # берём все числовые, кроме служебных
    candidates = [c for c in df.columns if c not in EXCLUDE_COLS]
    num_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols

def train_one_q(df: pd.DataFrame, q: int, feature_cols: list[str]) -> float:
    mask = df["question_number"] == q
    d = df.loc[mask].copy()
    X = d[feature_cols]
    y = d["score"].values

    params = dict(
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=RANDOM_STATE,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=6.0,
        iterations=2000,
        od_type="Iter",
        od_wait=200,
        verbose=False
    )

    train_pool = Pool(X, y)
    cv_res = cv(
        params=params,
        pool=train_pool,
        fold_count=5,
        partition_random_seed=RANDOM_STATE,
        shuffle=True,
        verbose=False
    )
    mae_cv = float(cv_res["test-MAE-mean"].iloc[-1])

    print(f"\n=== Обучаем модель для Q{q} (N={len(d)}) ===")
    for i in range(5):
        # Просто показываем сводный CV — фолды уже в cv_res усреднены
        pass
    print(f"Q{q} | CV MAE: {mae_cv:.3f}")

    # Финальная дообученная на всех данных модель
    model = CatBoostRegressor(**params)
    model.fit(train_pool, verbose=False)

    MODELS.mkdir(parents=True, exist_ok=True)
    model_path = MODELS / f"catboost_Q{q}.cbm"
    model.save_model(str(model_path))
    return mae_cv

def main():
    df = load_data()
    feature_cols = pick_feature_columns(df)

    # быстрый sanity-check: ключевые новые фичи должны быть в списке
    must_have = {"semantic_sim", "q4_slots_covered"}
    missing = [c for c in must_have if c not in feature_cols]
    if missing:
        print(f"⚠️ Внимание: не найдено фич {missing} среди обучающих столбцов!")

    maes = {}
    for q in (1, 2, 3, 4):
        maes[q] = train_one_q(df, q, feature_cols)

    print("\n------ ИТОГО ------")
    for q in (1, 2, 3, 4):
        print(f"Q{q}: MAE={maes[q]:.3f}")
    print(f"Среднее MAE по всем вопросам: {np.mean(list(maes.values())):.3f}")

if __name__ == "__main__":
    main()
