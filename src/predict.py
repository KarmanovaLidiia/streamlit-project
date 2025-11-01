# src/predict.py
from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys
import tempfile
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor

# --- импорты проекта ---
HERE = Path(__file__).parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    # если feature_engineering.py лежит в корне проекта
    from feature_engineering import FeatureExtractor
except ModuleNotFoundError:
    # если файл лежит в src/
    from src.feature_engineering import FeatureExtractor  # type: ignore

# --- пути ---
MODELS_DIR = ROOT / "models"                   # catboost_Q1.cbm ... catboost_Q4.cbm
ON_TOPIC_PATH = MODELS_DIR / "on_topic.pkl"    # опционально

# --- служебные колонки (не подавать в модель) ---
NON_NUMERIC_KEEP = {"question_number", "question_text", "answer_text"}
TARGET_COLS = {"score", "Оценка экзаменатора"}


# =========================
# Утилиты
# =========================
def _read_csv_safely(path: Path) -> pd.DataFrame:
    """Надёжное чтение CSV: пробуем разные кодировки/разделители."""
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
    raise last_err  # type: ignore[misc]


def _clip_by_q(qnum: int, preds: np.ndarray) -> np.ndarray:
    """Клип по допустимому диапазону оценок для каждого вопроса."""
    if qnum in (1, 3):
        lo, hi = 0.0, 1.0
    elif qnum in (2, 4):
        lo, hi = 0.0, 2.0
    else:
        lo, hi = 0.0, 2.0
    return np.clip(preds, lo, hi)


def _load_model(qnum: int) -> CatBoostRegressor:
    """Загрузка CatBoost-модели для указанного вопроса."""
    model_path = MODELS_DIR / f"catboost_Q{qnum}.cbm"
    if not model_path.exists():
        raise FileNotFoundError(f"Не найден файл модели: {model_path}")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model


def _align_to_model_features(model: CatBoostRegressor, X: pd.DataFrame) -> pd.DataFrame:
    """Выравниваем матрицу признаков под порядок/набор, с которым обучалась модель."""
    names = list(getattr(model, "feature_names_", []))
    if not names:
        return X
    Z = pd.DataFrame(index=X.index, dtype=float)
    for col in names:
        Z[col] = X[col] if col in X.columns else 0.0
    return Z


def _maybe_add_on_topic(df_feats: pd.DataFrame) -> pd.DataFrame:
    """
    Если есть on_topic.pkl (pack = {'model': clf, 'features': [...]})
    — добавляем вероятность 'on_topic_prob'. Иначе 0.0.
    """
    out = df_feats.copy()
    if not ON_TOPIC_PATH.exists():
        out["on_topic_prob"] = 0.0
        return out

    try:
        pack = joblib.load(ON_TOPIC_PATH)
        clf = pack["model"]
        need_feats: List[str] = pack.get("features", [])
        for f in need_feats:
            if f not in out.columns:
                out[f] = 0.0
        X_on = out[need_feats].fillna(0).values
        out["on_topic_prob"] = clf.predict_proba(X_on)[:, 1].astype("float32")
    except Exception as e:
        print(f"[!] Не удалось применить on_topic.pkl: {e}")
        out["on_topic_prob"] = 0.0
    return out


def _select_numeric_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Оставляем только числовые признаки, исключая служебные/текстовые колонки."""
    cols = []
    for c in feats.columns:
        if c in NON_NUMERIC_KEEP or c in TARGET_COLS:
            continue
        if pd.api.types.is_numeric_dtype(feats[c]):
            cols.append(c)
    X = feats[cols].copy()
    return X.fillna(0.0)


# =========================
# Основной конвейер
# =========================
def pipeline_infer(input_csv: Path, output_csv: Path) -> None:
    """
    1) читаем входной CSV
    2) строим признаки (FeatureExtractor)
    3) (опц.) добавляем on_topic_prob
    4) предсказываем по 4 моделям CatBoost
    5) сохраняем исходный CSV + pred_score + pred_score_rounded
    """
    # 1) входной CSV
    df_raw = _read_csv_safely(input_csv)

    # 2) извлечение признаков (быстрый режим по умолчанию)
    fast_mode = os.environ.get("FAST_MODE", "1") == "1"
    # лёгкая русская модель эмбеддингов — быстрее на CPU
    sbert_name = "cointegrated/rubert-tiny" if fast_mode else "ai-forever/sbert_large_nlu_ru"
    use_grammar = False if fast_mode else True

    fe = FeatureExtractor(
        sbert_model_name=sbert_name,
        use_grammar=use_grammar,     # на HF лучше False
        strip_examiner=True
    )
    feats = fe.extract_all_features(df_raw)

    # 3) on_topic (если есть)
    feats = _maybe_add_on_topic(feats)

    # 4) предсказания
    preds = np.zeros(len(feats), dtype=float)
    models_cache: Dict[int, CatBoostRegressor] = {}
    X_all = _select_numeric_features(feats)

    for q in (1, 2, 3, 4):
        mask = feats["question_number"] == q
        if not mask.any():
            continue
        if q not in models_cache:
            models_cache[q] = _load_model(q)
        model = models_cache[q]
        Xq = _align_to_model_features(model, X_all.loc[mask])
        pq = model.predict(Xq)
        pq = np.asarray(pq, dtype=float).reshape(-1)
        preds[mask.values] = _clip_by_q(q, pq)

    # --- новое надёжное округление (без .loc по индексам) ---
    qnums = feats["question_number"].astype(int).to_numpy()
    rounded = np.rint(preds).astype(np.float32)
    mask13 = (qnums == 1) | (qnums == 3)
    mask24 = (qnums == 2) | (qnums == 4)
    rounded[mask13] = np.clip(rounded[mask13], 0, 1)
    rounded[mask24] = np.clip(rounded[mask24], 0, 2)
    rounded = rounded.astype(int)

    # 5) сборка результата
    out = df_raw.copy()
    if "Оценка экзаменатора" not in out.columns:
        out["Оценка экзаменатора"] = np.nan
    out["pred_score"] = preds
    out["pred_score_rounded"] = rounded

    # безопасная запись
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_csv.with_suffix(".tmp.csv")
    out.to_csv(tmp_out, index=False, encoding="utf-8-sig", sep=";")
    os.replace(tmp_out, output_csv)
    print(f"[✓] Готово: {output_csv}")


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Инференс для DataFrame (без файлов)."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="predict_df_"))
    tmp_in = tmp_dir / "input.csv"
    tmp_out = tmp_dir / "output.csv"
    df.to_csv(tmp_in, index=False, encoding="utf-8-sig", sep=";")
    pipeline_infer(tmp_in, tmp_out)
    return pd.read_csv(tmp_out, encoding="utf-8-sig", sep=";")


# =========================
# CLI
# =========================
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auto-grader inference pipeline")
    p.add_argument("-i", "--input", type=str, required=True, help="Путь к входному CSV")
    p.add_argument("-o", "--output", type=str, required=True, help="Путь к выходному CSV")
    return p


def main():
    args = _build_argparser().parse_args()
    input_csv = Path(args.input).resolve()
    output_csv = Path(args.output).resolve()
    pipeline_infer(input_csv, output_csv)


if __name__ == "__main__":
    main()
