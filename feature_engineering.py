# feature_engineering.py
from __future__ import annotations

import re
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
except Exception:  # чтобы не падать на установке
    SentenceTransformer = None  # type: ignore
    sbert_util = None  # type: ignore

try:
    import language_tool_python
except Exception:
    language_tool_python = None  # type: ignore


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s?!.,:;ёЁа-яА-Я-]", re.UNICODE)

# мини-лексиконы под критерии
POLITE_WORDS = {"здравствуйте", "здравствуй", "пожалуйста", "спасибо", "будьте добры"}
APOLOGY_WORDS = {"извините", "простите", "прошу прощения"}
FAMILY_WORDS = {"семья", "сын", "дочь", "дети", "ребёнок", "муж", "жена", "родители"}
SEASON_WORDS = {"зима", "весна", "лето", "осень"}
SHOP_WORDS = {"рассрочка", "гарантия", "характеристики", "документы", "касса"}
YESNO_WORDS = {"да", "нет", "наверное", "возможно"}


def _strip_html(s: str) -> str:
    s = _HTML_TAG_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _only_text(s: str) -> str:
    s = s.lower()
    s = _strip_html(s)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _split_sentences(s: str) -> List[str]:
    # простая сегментация
    parts = re.split(r"(?<=[.!?])\s+", s)
    return [p.strip() for p in parts if p.strip()]


def _strip_examiner_lines(text: str) -> str:
    """
    Убираем вероятные реплики экзаменатора: предложения с '?',
    короткие управляющие фразы ("хорошо.", "итак, ...").
    """
    sents = _split_sentences(text)
    kept = []
    for i, sent in enumerate(sents):
        low = sent.lower()
        if "?" in sent:
            continue
        if low in {"хорошо.", "отлично.", "прекрасно.", "молодец."}:
            continue
        if low.startswith(("итак", "следующий", "теперь", "будьте", "ответьте")) and "?" in low:
            continue
        kept.append(sent)
    return " ".join(kept) if kept else text


def _count_matches(words: Iterable[str], tokens: Iterable[str]) -> int:
    wset = set(w.lower() for w in words)
    return sum(1 for t in tokens if t in wset)


class FeatureExtractor:
    """
    Лёгкий экстрактор признаков:
    - очистка текста/HTML
    - отделение реплик экзаменатора (эвристика)
    - семантическая близость (SBERT)
    - длины, кол-во предложений, вопросительных/восклицательных и пр.
    - индикаторы по заданиям (вежливость, извинение, семья, рассрочка, …)
    - (опц.) grammar_error_count через LanguageTool
    """

    def __init__(
        self,
        sbert_model_name: str = "cointegrated/rubert-tiny",
        use_grammar: bool = False,
        strip_examiner: bool = True,
    ) -> None:
        self.strip_examiner = strip_examiner

        # SBERT
        self.sbert: Optional[SentenceTransformer]
        if SentenceTransformer is None:
            self.sbert = None
        else:
            self.sbert = SentenceTransformer(sbert_model_name)

        # Grammar
        self.grammar = None
        if use_grammar and language_tool_python is not None:
            try:
                self.grammar = language_tool_python.LanguageTool("ru")
            except Exception:
                self.grammar = None  # безопасно отключаем

    # --------- примитивные фичи ----------
    def _basic_text_stats(self, text: str) -> Tuple[int, int, int, int, int, float]:
        cleaned = _only_text(text)
        tokens = cleaned.split()
        sents = _split_sentences(text)
        qmarks = text.count("?")
        emarks = text.count("!")
        avg_sent_len = (len(tokens) / max(len(sents), 1)) if tokens else 0.0
        return len(tokens), len(sents), qmarks, emarks, len(set(tokens)), float(avg_sent_len)

    def _semantic_sim(self, q: str, a: str) -> float:
        if not self.sbert or sbert_util is None:
            return 0.0
        try:
            emb_q = self.sbert.encode([q], convert_to_tensor=True, normalize_embeddings=True)
            emb_a = self.sbert.encode([a], convert_to_tensor=True, normalize_embeddings=True)
            sim = float(sbert_util.cos_sim(emb_q, emb_a)[0][0].cpu().item())
            # нормализуем к [0..1] примерно
            return max(0.0, min(1.0, (sim + 1.0) / 2.0))
        except Exception:
            return 0.0

    def _grammar_errors(self, text: str) -> int:
        if not self.grammar:
            return 0
        try:
            matches = self.grammar.check(text)
            return len(matches)
        except Exception:
            return 0

    # --------- фичи под задания ----------
    def _question_specific_flags(self, qnum: int, answer_text: str, question_text: str) -> dict:
        a_clean = _only_text(answer_text)
        a_tokens = a_clean.split()

        flags = {
            "has_politeness": int(_count_matches(POLITE_WORDS, a_tokens) > 0),
            "has_apology": int(_count_matches(APOLOGY_WORDS, a_tokens) > 0),
            "has_yesno": int(_count_matches(YESNO_WORDS, a_tokens) > 0),
            "mentions_family": int(_count_matches(FAMILY_WORDS, a_tokens) > 0),
            "mentions_season": int(_count_matches(SEASON_WORDS, a_tokens) > 0),
            "mentions_shop": int(_count_matches(SHOP_WORDS, a_tokens) > 0),
            "has_question_mark": int("?" in answer_text),
        }

        # лёгкие правила по задачам
        if qnum == 1:  # извиниться + спросить
            flags["task_completed_like_q1"] = int(flags["has_apology"] and flags["has_question_mark"])
        elif qnum == 2:  # диалоговые ответы
            flags["task_completed_like_q2"] = int(flags["has_yesno"] or len(a_tokens) > 12)
        elif qnum == 3:  # магазин: документы/рассрочка/характеристики
            flags["task_completed_like_q3"] = int(flags["mentions_shop"] or len(a_tokens) > 25)
        elif qnum == 4:  # описание картинки + семья/дети
            flags["task_completed_like_q4"] = int(flags["mentions_family"] or flags["mentions_season"])
        else:
            flags["task_completed_like_q1"] = 0

        # семантика вопрос-ответ
        flags["qa_semantic_sim"] = self._semantic_sim(question_text, answer_text)
        return flags

    # --------- публичное API ----------
    def extract_row_features(self, row: pd.Series) -> dict:
        qnum = int(row.get("№ вопроса") or row.get("question_number") or 0)
        qtext_raw = str(row.get("Текст вопроса") or row.get("question_text") or "")
        atext_raw = str(row.get("Транскрибация") or row.get("transcript") or row.get("answer_text") or "")

        qtext = _strip_html(qtext_raw)
        atext = _strip_html(atext_raw)
        if self.strip_examiner:
            atext = _strip_examiner_lines(atext)

        tok_len, sent_cnt, qmarks, emarks, uniq, avg_sent = self._basic_text_stats(atext)
        grams = self._grammar_errors(atext)

        base = {
            "question_number": qnum,
            "question_text": qtext,
            "answer_text": atext,
            "tokens_len": tok_len,
            "sent_count": sent_cnt,
            "q_mark_count": qmarks,
            "excl_mark_count": emarks,
            "uniq_tokens": uniq,
            "avg_sent_len": avg_sent,
            "grammar_errors": grams,
            "answer_len_chars": len(atext),
        }
        base.update(self._question_specific_flags(qnum, atext, qtext))
        return base

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = [self.extract_row_features(r) for _, r in df.iterrows()]
        out = pd.DataFrame(feats)

        # защитимся от NaN и типов
        num_cols = [c for c in out.columns if c not in {"question_text", "answer_text"}]
        for c in num_cols:
            if c not in {"question_text", "answer_text"}:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.fillna(
            {c: 0 for c in out.columns if c not in {"question_text", "answer_text"}}
        )
        return out
