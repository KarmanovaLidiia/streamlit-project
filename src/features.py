# src/features.py
from __future__ import annotations
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"[.!?]+[\s\n]+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _tokenize_words(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    return _WORD_RE.findall(text)


def _split_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Принимает датафрейм со столбцами:
      - question_number
      - question_text
      - answer_text
      - score (может отсутствовать на инференсе, тогда создадим NaN)

    Возвращает df с базовыми признаками:
      ans_len_chars, ans_len_words, ans_n_sents, ans_avg_sent_len,
      ans_ttr, ans_short_sent_rt, ans_punct_rt, q_len_words, has_intro
    """

    out = df.copy()

    if "score" not in out.columns:
        out["score"] = np.nan

    # гарантируем строки
    out["question_text"] = out["question_text"].fillna("").astype(str)
    out["answer_text"]   = out["answer_text"].fillna("").astype(str)

    # длины/токены
    ans_words = out["answer_text"].apply(_tokenize_words)
    q_words   = out["question_text"].apply(_tokenize_words)
    ans_sents = out["answer_text"].apply(_split_sentences)

    out["ans_len_chars"] = out["answer_text"].str.len()
    out["ans_len_words"] = ans_words.apply(len).astype(int)

    out["ans_n_sents"] = ans_sents.apply(len).astype(int)
    out["ans_avg_sent_len"] = (
        out["ans_len_words"] / out["ans_n_sents"].replace({0: np.nan})
    ).fillna(0).astype(float)

    # Type-Token Ratio
    def _ttr(ws: List[str]) -> float:
        return 0.0 if not ws else len(set(map(str.lower, ws))) / float(len(ws))

    out["ans_ttr"] = ans_words.apply(_ttr).astype(float)

    # доля коротких предложений (<= 5 слов)
    def _short_rate(sents: List[str]) -> float:
        if not sents:
            return 0.0
        cnt = 0
        for s in sents:
            if len(_tokenize_words(s)) <= 5:
                cnt += 1
        return cnt / float(len(sents))

    out["ans_short_sent_rt"] = ans_sents.apply(_short_rate).astype(float)

    # доля пунктуации в ответе
    def _punct_ratio(text: str) -> float:
        if not text:
            return 0.0
        punct = len(_PUNCT_RE.findall(text))
        return punct / float(len(text))

    out["ans_punct_rt"] = out["answer_text"].apply(_punct_ratio).astype(float)

    # длина вопроса в словах
    out["q_len_words"] = q_words.apply(len).astype(int)

    # наличие вводной части — простая эвристика
    out["has_intro"] = out["answer_text"].str.contains(
        r"\b(во-первых|например|сначала|итак|сперва|прежде всего)\b",
        case=False, na=False
    ).astype(float)

    # порядок колонок
    cols = [
        "question_number", "question_text", "answer_text", "score",
        "ans_len_chars", "ans_len_words", "ans_n_sents", "ans_avg_sent_len",
        "ans_ttr", "ans_short_sent_rt", "ans_punct_rt", "q_len_words", "has_intro",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols]
