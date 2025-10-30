# src/semantic_cache.py
from __future__ import annotations
import sqlite3
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Абсолютный путь к БД: <корень проекта>/data/cache/embeddings.sqlite
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "cache" / "embeddings.sqlite"

def _norm_text(t: str) -> str:
    return " ".join((t or "").strip().split())

def _hash_text(t: str) -> str:
    return hashlib.blake2s(_norm_text(t).encode("utf-8")).hexdigest()

def _ensure_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        # Немного настроек для Windows, чтобы запись была надёжнее
        con.execute("PRAGMA journal_mode = WAL;")
        con.execute("PRAGMA synchronous = NORMAL;")
        con.execute("""
            CREATE TABLE IF NOT EXISTS emb (
                h    TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                dim  INTEGER NOT NULL,
                vec  BLOB NOT NULL
            )
        """)
        con.commit()

def fetch_from_cache(texts: List[str]) -> Dict[str, np.ndarray]:
    _ensure_db()
    hashes = [_hash_text(t) for t in texts]
    out: Dict[str, np.ndarray] = {}
    if not hashes:
        return out
    with sqlite3.connect(DB_PATH) as con:
        q = "SELECT h, dim, vec FROM emb WHERE h IN ({})".format(",".join("?" * len(hashes)))
        for h, dim, blob in con.execute(q, hashes):
            arr = np.frombuffer(blob, dtype=np.float32).reshape(dim)
            out[h] = arr
    return out

def write_to_cache(items: List[Tuple[str, str, np.ndarray]]) -> None:
    if not items:
        return
    _ensure_db()
    with sqlite3.connect(DB_PATH) as con:
        con.executemany(
            "INSERT OR REPLACE INTO emb(h, text, dim, vec) VALUES (?,?,?,?)",
            [(h, _norm_text(t), v.size, v.astype(np.float32).tobytes()) for h, t, v in items]
        )
        con.commit()

def embed_with_cache(texts: List[str], model, batch_size: int = 16, verbose: bool = True) -> np.ndarray:
    """
    Возвращает эмбеддинги для texts. Сначала достаёт из кэша,
    для недостающих — считает моделью и кладёт в кэш.
    """
    _ensure_db()
    hashes = [_hash_text(t) for t in texts]
    cached = fetch_from_cache(texts)  # hash -> vec

    out: List[np.ndarray | None] = [None] * len(texts)
    missing_idx = [i for i, h in enumerate(hashes) if h not in cached]

    if verbose:
        print(f"[cache] DB: {DB_PATH}")
        print(f"[cache] всего текстов: {len(texts)} | из кэша найдено: {len(texts) - len(missing_idx)} | посчитать: {len(missing_idx)}")

    # Из кэша
    for i, h in enumerate(hashes):
        if h in cached:
            out[i] = cached[h]

    # Досчитываем моделью
    if missing_idx:
        to_compute = [texts[i] for i in missing_idx]
        vecs = []
        if verbose:
            print(f"[cache] считаем моделью в батчах по {batch_size}...")

        for b in range(0, len(to_compute), batch_size):
            chunk = to_compute[b:b + batch_size]
            if verbose:
                print(f"[cache] batch {b//batch_size + 1}/{(len(to_compute)+batch_size-1)//batch_size} | {len(chunk)} примеров")
            vecs_chunk = model.encode(
                chunk,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True  # включаем прогрессбар
            )
            vecs.append(vecs_chunk)
        vecs = np.vstack(vecs)

        # Запись в out и в кэш
        items_for_cache: List[Tuple[str, str, np.ndarray]] = []
        for j, idx in enumerate(missing_idx):
            v = vecs[j]
            out[idx] = v
            items_for_cache.append((hashes[idx], texts[idx], v))

        write_to_cache(items_for_cache)
        if verbose:
            print(f"[cache] записано в кэш: {len(items_for_cache)} векторов")

    # Упаковываем результат
    result = np.vstack(out).astype(np.float32)
    if verbose:
        print(f"[cache] готово: shape={result.shape}, dtype={result.dtype}")
    return result
