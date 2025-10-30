from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys
import tempfile
import math

import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from huggingface_hub import hf_hub_download  # <— fallback-скачивание моделей

from src.data_cleaning import prepare_dataframe
from src.features import build_baseline_features
from src.features_q4 import add_q4_features
from src.semantic_features import add_semantic_similarity
from src.explanations import add_score_explanations

# ----------------------- PATHS -----------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Репозиторий Space, откуда подкачиваем модели при отсутствии локально
SPACE_REPO_ID = os.getenv("SPACE_REPO_ID", "lidiiakarmanova/exam-evaluator")

# ----------------------- CONFIG (env) -----------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
FAST_ROW_LIMIT = os.getenv("FAST_ROW_LIMIT")  # e.g. "2000"
DISABLE_EXPLANATIONS = os.getenv("DISABLE_EXPLANATIONS", "0") == "1"

# ----------------------- IO helpers ------------------------
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

# ----------------------- Model cache ------------------------
_MODEL_CACHE: dict[int, CatBoostRegressor] = {}

def _load_model(qnum: int) -> CatBoostRegressor:
    """
    Пытаемся загрузить модель Q{qnum} из локальной папки models/.
    Если файла нет (часто так бывает в контейнере), скачиваем из Space.
    """
    if qnum in _MODEL_CACHE:
        return _MODEL_CACHE[qnum]

    local_path = MODELS_DIR / f"catboost_Q{qnum}.cbm"
    if not local_path.exists():
        print(f"[models] {local_path} не найден. Скачиваю из Space…")
        downloaded = hf_hub_download(
            repo_id=SPACE_REPO_ID,
            repo_type="space",                 # ВАЖНО: это Space-репозиторий
            filename=f"models/catboost_Q{qnum}.cbm",
        )
        local_path = Path(downloaded)

    model = CatBoostRegressor()
    model.load_model(str(local_path))
    _MODEL_CACHE[qnum] = model
    return model

def _align_to_model_features(model: CatBoostRegressor, X: pd.DataFrame) -> pd.DataFrame:
    names = list(model.feature_names_)
    if names:
        Z = pd.DataFrame(index=X.index, dtype=float)
        for col in names:
            Z[col] = X[col] if col in X.columns else 0.0
        return Z[names].astype("float32")
    return X.reindex(sorted(X.columns), axis=1).astype("float32")

def _maybe_add_on_topic(df_feats: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляем вероятность «по теме» (если есть классifier on_topic.pkl).
    При отсутствии — пытаемся скачать из Space, иначе ставим 0.0.
    """
    out = df_feats.copy()
    path = MODELS_DIR / "on_topic.pkl"
    if not path.exists():
        try:
            print("[models] on_topic.pkl не найден. Скачиваю из Space…")
            downloaded = hf_hub_download(
                repo_id=SPACE_REPO_ID,
                repo_type="space",
                filename="models/on_topic.pkl",
            )
            path = Path(downloaded)
        except Exception as e:
            print(f"[models] on_topic.pkl недоступен ({e}). Пропускаю.")
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

# ----------------------- Chunked apply ----------------------
def _apply_in_chunks(df: pd.DataFrame, fn, chunk_size: int) -> pd.DataFrame:
    """
    Разбивает df на куски по chunk_size и применяет fn к каждому куску, конкатенируя результат.
    fn: (pd.DataFrame) -> pd.DataFrame  (должен сохранять порядок и индексы)
    """
    if len(df) == 0:
        return df
    if chunk_size <= 0 or len(df) <= chunk_size:
        return fn(df)

    parts = []
    total = len(df)
    n_chunks = math.ceil(total / chunk_size)
    for i in range(n_chunks):
        lo = i * chunk_size
        hi = min((i + 1) * chunk_size, total)
        chunk = df.iloc[lo:hi].copy()
        print(f"[chunk] {i+1}/{n_chunks} → строки [{lo}:{hi})")
        out_chunk = fn(chunk)
        parts.append(out_chunk)
    res = pd.concat(parts, axis=0)
    res = res.loc[df.index]  # вернуть исходный порядок
    return res

# ----------------------- Main pipeline ----------------------
def pipeline_infer(input_csv: Path, output_csv: Path) -> None:
    df_raw = _read_csv_safely(input_csv)
    print(f"CSV прочитан: {len(df_raw):,} строк")

    # Быстрый режим (по env), если задан
    if FAST_ROW_LIMIT:
        try:
            n = int(FAST_ROW_LIMIT)
            if n < len(df_raw):
                df_raw = df_raw.iloc[:n].copy()
                print(f"[FAST] Используем только первые {n} строк по FAST_ROW_LIMIT")
        except Exception:
            pass

    # 1) Cleaning
    try:
        df_clean = prepare_dataframe(df_raw)
        print("DataFrame подготовлен")
    except Exception as e:
        print(f"Ошибка в prepare_dataframe: {e}")
        raise

    # 2) Baseline features
    try:
        feats = build_baseline_features(df_clean)
        print("Базовые фичи построены")
    except Exception as e:
        print(f"Ошибка в build_baseline_features: {e}")
        raise

    # 3) Semantic similarity (самый тяжёлый шаг) — батчами
    print("Вычисляем семантическую близость (ruSBERT) батчами…")
    try:
        feats = _apply_in_chunks(
            feats,
            fn=add_semantic_similarity,
            chunk_size=CHUNK_SIZE,
        )
        print("Семантические фичи добавлены")
    except Exception as e:
        print(f"Ошибка в add_semantic_similarity: {e}")
        raise

    # 4) Q4 features — тоже батчами (на всякий случай)
    try:
        feats = _apply_in_chunks(
            feats,
            fn=add_q4_features,
            chunk_size=CHUNK_SIZE,
        )
        print("Q4 фичи добавлены")
    except Exception as e:
        print(f"Ошибка в add_q4_features: {e}")
        raise

    # 5) Optional on-topic
    feats = _maybe_add_on_topic(feats)

    # 6) Inference per question
    preds = np.zeros(len(feats), dtype=float)
    for q in (1, 2, 3, 4):
        mask = feats["question_number"] == q
        if not mask.any():
            continue
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

    # 7) Explanations (можно отключить env-переменной)
    if not DISABLE_EXPLANATIONS:
        try:
            feats_with_explanations = add_score_explanations(feats, preds)
            print("Объяснения оценок добавлены")
        except Exception as e:
            print(f"Не удалось добавить объяснения: {e}")
            feats_with_explanations = feats.copy()
            feats_with_explanations["score_explanation"] = "Объяснение недоступно"
    else:
        print("[FAST] DISABLE_EXPLANATIONS=1 → пропускаем объяснения")
        feats_with_explanations = feats.copy()
        feats_with_explanations["score_explanation"] = "Пропущено (быстрый режим)"

    # 8) Output
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

# ----------------------- Helper API ----------------------
def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    tmp_dir = Path(tempfile.mkdtemp(prefix="predict_df_"))
    tmp_in = tmp_dir / "input.csv"
    tmp_out = tmp_dir / "output.csv"
    df.to_csv(tmp_in, index=False, encoding="utf-8-sig", sep=";")
    pipeline_infer(tmp_in, tmp_out)
    df_pred = pd.read_csv(tmp_out, encoding="utf-8-sig", sep=";")
    return df_pred

# ----------------------- CLI ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Путь к исходному CSV")
    parser.add_argument("--output", type=str, required=True, help="Куда сохранить CSV с предсказаниями")
    args = parser.parse_args()
    try:
        pipeline_infer(Path(args.input), Path(args.output))
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        raise
