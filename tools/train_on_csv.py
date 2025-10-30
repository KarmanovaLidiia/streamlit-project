# tools/train_on_csv.py
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool, cv

# Добавляем путь к src
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_cleaning import prepare_dataframe
from src.features import build_baseline_features
from src.semantic_features import add_semantic_similarity
from src.features_q4 import add_q4_features


# Добавляем путь к src, чтобы импорты работали
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

EXCLUDE = {"question_number", "question_text", "answer_text", "score"}

def _pick_num_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]

def _make_features(raw_csv: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(raw_csv, encoding="utf-8-sig", sep=";")
    df = prepare_dataframe(df_raw)
    feats = build_baseline_features(df)
    feats = add_semantic_similarity(feats)   # ruSBERT + cache
    feats = add_q4_features(feats)           # правила для Q4
    return feats

def _train_one_q(df: pd.DataFrame, q: int, fcols: list[str]) -> float:
    d = df.loc[df["question_number"] == q].copy()
    pool = Pool(d[fcols], d["score"].values)
    params = dict(
        loss_function="MAE", eval_metric="MAE", random_seed=RANDOM_STATE,
        learning_rate=0.08, depth=6, l2_leaf_reg=6.0,
        iterations=2000, od_type="Iter", od_wait=200, verbose=False
    )
    cv_res = cv(params=params, pool=pool, fold_count=5,
                partition_random_seed=RANDOM_STATE, shuffle=True, verbose=False)
    mae_cv = float(cv_res["test-MAE-mean"].iloc[-1])
    print(f"Q{q} | CV MAE: {mae_cv:.3f}")

    model = CatBoostRegressor(**params)
    model.fit(pool, verbose=False)
    MODELS.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODELS / f"catboost_Q{q}.cbm"))
    return mae_cv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV для обучения (train_split.csv)")
    args = ap.parse_args()
    feats = _make_features(Path(args.input))
    fcols = _pick_num_cols(feats)
    maes = {}
    for q in (1, 2, 3, 4):
        maes[q] = _train_one_q(feats, q, fcols)
    print("\n------ ИТОГО (CV на train) ------")
    for q in (1,2,3,4):
        print(f"Q{q}: MAE={maes[q]:.3f}")
    print(f"Среднее: {np.mean(list(maes.values())):.3f}")

if __name__ == "__main__":
    main()
