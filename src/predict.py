from __future__ import annotations

from pathlib import Path
import argparse
import sys
import tempfile
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor

from src.data_cleaning import prepare_dataframe
from src.features import build_baseline_features
from src.features_q4 import add_q4_features
from src.semantic_features import add_semantic_similarity
from src.explanations import add_score_explanations

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

def _read_csv_safely(path: Path) -> pd.DataFrame:
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
                df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
                used_sep = "auto"
            else:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                used_sep = sep
            print(f"[i] CSV прочитан с encoding='{enc}', sep='{used_sep}'")
            return df
        except Exception as e:
            last_err = e
    raise last_err

def _clip_by_q(qnum: int, preds: np.ndarray) -> np.ndarray:
    if qnum in (1, 3):
        lo, hi = 0.0, 1.0
    elif qnum in (2, 4):
        lo, hi = 0.0, 2.0
    else:
        lo, hi = 0.0, 2.0
    return np.clip(preds, lo, hi)

def _load_model(qnum: int) -> CatBoostRegressor:
    model_path = MODELS_DIR / f"model_q{qnum}.cbm"
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model

def _align_to_model_features(model: CatBoostRegressor, X: pd.DataFrame) -> pd.DataFrame:
    names = list(model.feature_names_)
    if names:
        Z = pd.DataFrame(index=X.index, dtype=float)
        for col in names:
            Z[col] = X[col] if col in X.columns else 0.0
        return Z
    return X

def _maybe_add_on_topic(df_feats: pd.DataFrame) -> pd.DataFrame:
    out = df_feats.copy()
    path = MODELS_DIR / "on_topic.pkl"
    if not path.exists():
        out["on_topic_prob"] = 0.0
        return out
    pack = joblib.load(path)
    clf = pack["model"]
    need_feats = pack["features"]
    for f in need_feats:
        if f not in out.columns:
            out[f] = 0.0
    X_on = out[need_feats].fillna(0).values
    out["on_topic_prob"] = clf.predict_proba(X_on)[:, 1].astype("float32")
    return out

def pipeline_infer(input_csv: Path, output_csv: Path) -> None:
    df_raw = _read_csv_safely(input_csv)
    df_clean = prepare_dataframe(df_raw)
    feats = build_baseline_features(df_clean)
    feats = _maybe_add_on_topic(feats)

    preds = np.zeros(len(feats), dtype=float)
    for q in (1, 2, 3, 4):
        mask = feats["question_number"] == q
        fcols = [
            c for c in feats.columns
            if c not in ("question_number", "question_text", "answer_text", "score")
            and pd.api.types.is_numeric_dtype(feats[c])
        ]
        Xq = feats.loc[mask, fcols]
        model = _load_model(q)
        Xq = _align_to_model_features(model, Xq)
        pq = model.predict(Xq)
        preds[mask.values] = _clip_by_q(q, pq)

    try:
        feats_with_explanations = add_score_explanations(feats, preds)
    except Exception as e:
        print(f"Не удалось добавить объяснения: {e}")
        feats_with_explanations = feats.copy()
        feats_with_explanations["score_explanation"] = "Объяснение недоступно"

    out = df_raw.copy()
    if "Оценка экзаменатора" not in out.columns:
        out["Оценка экзаменатора"] = np.nan
    out["pred_score"] = preds
    out["pred_score_rounded"] = out["pred_score"].round()
    if "score_explanation" in feats_with_explanations.columns:
        out["объяснение_оценки"] = feats_with_explanations["score_explanation"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig", sep=";")
    print(f"Готово: {output_csv}")

def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    tmp_dir = Path(tempfile.mkdtemp(prefix="predict_df_"))
    tmp_in = tmp_dir / "input.csv"
    tmp_out = tmp_dir / "output.csv"
    df.to_csv(tmp_in, index=False, encoding="utf-8-sig", sep=";")
    pipeline_infer(tmp_in, tmp_out)
    return pd.read_csv(tmp_out, encoding="utf-8-sig", sep=";")

__all__ = ["pipeline_infer", "predict_dataframe"]
