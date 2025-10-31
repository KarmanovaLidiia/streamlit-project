from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys
import tempfile
import math
import json

import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from huggingface_hub import hf_hub_download

from src.data_cleaning import prepare_dataframe
from src.features import build_baseline_features
from src.features_q4 import add_q4_features
from src.semantic_features import add_semantic_similarity
from src.explanations import add_score_explanations

# ----------------------- PATHS -----------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SPACE_REPO_ID = os.getenv("SPACE_REPO_ID", "lidiiakarmanova/exam-evaluator")

# ----------------------- CONFIG -----------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
FAST_ROW_LIMIT = os.getenv("FAST_ROW_LIMIT")
DISABLE_EXPLANATIONS = os.getenv("DISABLE_EXPLANATIONS", "0") == "1"
PROGRESS_FILE = Path(os.getenv("PROGRESS_FILE", "/tmp/progress.json"))

# ----------------------- PROGRESS -----------------------
def _progress_write(stage: str, current: int, total: int, note: str = "") -> None:
    try:
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"stage": stage, "current": int(current), "total": int(total), "note": str(note)},
                f,
                ensure_ascii=False,
            )
    except Exception:
        pass


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
    if qnum in _MODEL_CACHE:
        return _MODEL_CACHE[qnum]

    local_path = MODELS_DIR / f"catboost_Q{qnum}.cbm"
    if not local_path.exists():
        print(f"[models] {local_path} не найден. Скачиваю из Space…")
        downloaded = hf_hub_download(
            repo_id=SPACE_REPO_ID,
            repo_type="space",
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


# ----------------------- Feature alias adapter -----------------------
def _add_feature_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт алиасы фичей с/без префикса 'q4_',
    чтобы избежать KeyError при несовпадении имён в модели и фичах.
    """
    out = df.copy()
    cols = list(df.columns)
    for col in cols:
        if col.startswith("q4_"):
            alias = col.replace("q4_", "", 1)
            if alias not in out.columns:
                out[alias] = out[col]
        else:
            prefixed = f"q4_{col}"
            if prefixed not in out.columns:
                out[prefixed] = out[col]
    return out


# ----------------------- Optional on-topic -----------------------
def _maybe_add_on_topic(df_feats: pd.DataFrame) -> pd.DataFrame:
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
def _apply_in_chunks(df: pd.DataFrame, fn, chunk_size: int, *, stage_name: str | None = None) -> pd.DataFrame:
    if len(df) == 0:
        if stage_name:
            _progress_write(stage_name, 1, 1, "empty")
        return df

    if chunk_size <= 0 or len(df) <= chunk_size:
        if stage_name:
            _progress_write(stage_name, 1, 1, "single chunk")
        return fn(df)

    parts = []
    total = len(df)
    n_chunks = math.ceil(total / chunk_size)
    for i in range(n_chunks):
        lo, hi = i * chunk_size, min((i + 1) * chunk_size, total)
        chunk = df.iloc[lo:hi].copy()
        out_chunk = fn(chunk)
        parts.append(out_chunk)
        if stage_name:
            _progress_write(stage_name, i + 1, n_chunks, f"строки [{lo}:{hi})")
    res = pd.concat(parts, axis=0)
    return res.loc[df.index]


# ----------------------- Main pipeline ----------------------
def pipeline_infer(input_csv: Path, output_csv: Path) -> None:
    _progress_write("загрузка CSV", 0, 1)
    df_raw = _read_csv_safely(input_csv)
    print(f"CSV прочитан: {len(df_raw):,} строк")

    # Быстрый режим
    if FAST_ROW_LIMIT:
        try:
            n = int(FAST_ROW_LIMIT)
            if n < len(df_raw):
                df_raw = df_raw.iloc[:n].copy()
                print(f"[FAST] Используем только первые {n} строк по FAST_ROW_LIMIT")
        except Exception:
            pass

    # 1) Очистка
    _progress_write("очистка", 0, 1)
    df_clean = prepare_dataframe(df_raw)
    _progress_write("очистка", 1, 1, "ok")

    # 2) Базовые фичи
    _progress_write("базовые фичи", 0, 1)
    feats = build_baseline_features(df_clean)
    _progress_write("базовые фичи", 1, 1, "ok")

    # 3) Семантика
    feats = _apply_in_chunks(feats, fn=add_semantic_similarity, chunk_size=CHUNK_SIZE, stage_name="семантика")

    # 4) Q4-фичи
    feats = _apply_in_chunks(feats, fn=add_q4_features, chunk_size=CHUNK_SIZE, stage_name="q4-фичи")

    # 5) Алиасы фичей (важно!)
    feats = _add_feature_aliases(feats)

    # 6) On-topic
    _progress_write("on-topic", 0, 1)
    feats = _maybe_add_on_topic(feats)
    _progress_write("on-topic", 1, 1, "ok")

    # 7) CatBoost
    total_q, done_q = 4, 0
    preds = np.zeros(len(feats), dtype=float)
    for q in (1, 2, 3, 4):
        _progress_write("инференс CatBoost", done_q, total_q, f"Q{q}")
        mask = feats["question_number"] == q
        if mask.any():
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
        done_q += 1
        _progress_write("инференс CatBoost", done_q, total_q, f"Q{q} ✓")

    # 8) Объяснения
    if not DISABLE_EXPLANATIONS:
        _progress_write("объяснения", 0, 1)
        try:
            feats_with_explanations = add_score_explanations(feats, preds)
        except Exception as e:
            print(f"Не удалось добавить объяснения: {e}")
            feats_with_explanations = feats.copy()
            feats_with_explanations["score_explanation"] = "Объяснение недоступно"
        _progress_write("объяснения", 1, 1, "ok")
    else:
        feats_with_explanations = feats.copy()
        feats_with_explanations["score_explanation"] = "Пропущено (быстрый режим)"

    # 9) Вывод
    _progress_write("сохранение", 0, 1)
    out = df_raw.copy()
    if "Оценка экзаменатора" not in out.columns:
        out["Оценка экзаменатора"] = np.nan
    out["pred_score"] = preds
    out["pred_score_rounded"] = out["pred_score"].round()
    if "score_explanation" in feats_with_explanations.columns:
        out["объяснение_оценки"] = feats_with_explanations["score_explanation"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig", sep=";")
    _progress_write("готово", 1, 1)
    print(f"Готово: {output_csv}")


# ----------------------- Helper API ----------------------
def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    tmp_dir = Path(tempfile.mkdtemp(prefix="predict_df_"))
    tmp_in, tmp_out = tmp_dir / "input.csv", tmp_dir / "output.csv"
    df.to_csv(tmp_in, index=False, encoding="utf-8-sig", sep=";")
    pipeline_infer(tmp_in, tmp_out)
    return pd.read_csv(tmp_out, encoding="utf-8-sig", sep=";")


# ----------------------- CLI ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    try:
        pipeline_infer(Path(args.input), Path(args.output))
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        raise
