# src/semantic_features.py
from __future__ import annotations
from functools import lru_cache
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.semantic_cache import embed_with_cache

_MODEL_NAME = "ai-forever/sbert_large_nlu_ru"


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Ленивая загрузка модели (кэшируется в процессе)."""
    return SentenceTransformer(_MODEL_NAME)


def add_semantic_similarity(
    df: pd.DataFrame,
    batch_size: int = 16,      # меньше батч по умолчанию, чтобы точно писался кэш
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Добавляет колонку 'semantic_sim' — косинусное сходство вопроса и ответа.
    Эмбеддинги извлекаются через embed_with_cache (нормализованы),
    поэтому cos_sim == dot(a, q).
    """
    out = df.copy()

    # гарантируем наличие столбцов
    for col in ("question_text", "answer_text"):
        if col not in out.columns:
            out[col] = ""

    if len(out) == 0:
        out["semantic_sim"] = np.array([], dtype=np.float32)
        return out

    model = _load_model()

    q_texts = out["question_text"].fillna("").astype(str).tolist()
    a_texts = out["answer_text"].fillna("").astype(str).tolist()

    if verbose:
        print("🔹 Проверяем кэш (вопросы)...")
    q_emb = embed_with_cache(q_texts, model, batch_size=batch_size, verbose=verbose)  # (N, D) float32

    if verbose:
        print("🔹 Проверяем кэш (ответы)...")
    a_emb = embed_with_cache(a_texts, model, batch_size=batch_size, verbose=verbose)  # (N, D) float32

    # косинус = скалярное произведение (векторы уже нормированы в embed_with_cache)
    sims = (a_emb * q_emb).sum(axis=1).astype(np.float32)
    np.clip(sims, -1.0, 1.0, out=sims)

    out["semantic_sim"] = sims
    return out
