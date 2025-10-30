# src/data_cleaning.py
import re
import pandas as pd
from bs4 import BeautifulSoup

# ⚠️ новый импорт: извлекаем речь тестируемого
from src.text_roles import extract_tester_reply


# ---------- утилиты очистки ----------
def clean_html(text: str) -> str:
    """Удаляем HTML/разметку из текста вопроса/ответа."""
    if pd.isna(text):
        return ""
    return BeautifulSoup(str(text), "lxml").get_text(separator=" ", strip=True)


# эвристический парсер на случай, если в транскрипте есть роли
# (оставляем куски после "Тестируемый:/Кандидат:/Студент:" до следующего "Экзаменатор:")
_SPEAKER_PAT = re.compile(
    r"(?:Тестируемый|Кандидат|Студент)\s*:\s*(.+?)(?=(?:Экзаменатор|Преподаватель|Собеседник)\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)

def extract_answer(transcript: str) -> str:
    """Базовое извлечение ответа из общей транскрипции (если есть метки ролей)."""
    if not isinstance(transcript, str) or not transcript.strip():
        return ""
    t = transcript.replace("\r", "\n")
    chunks = _SPEAKER_PAT.findall(t)
    joined = " ".join(x.strip() for x in chunks) if chunks else t
    return re.sub(r"\s+", " ", joined).strip()


# ---------- поиски колонок ----------
_CANDIDATES = {
    "question_number": [
        "номер вопроса", "порядковый номер", "порядковый номер вопроса",
        "№ вопроса", "вопрос №", "номер", "question_number"
    ],
    "question_text": [
        "текст вопроса", "вопрос", "формулировка вопроса",
        "question_text", "question"
    ],
    "transcript": [
        "транскрипция ответа", "транскрибация ответа", "транскрипт",
        "диалог", "ответ (текст)", "аудио транскрипт", "текст ответа",
        "transcript", "answer_text"
    ],
    "score": [
        "оценка", "оценка экзаменатора", "балл", "баллы",
        "score", "target"
    ],
}

def _find_column(df: pd.DataFrame, keys: list[str]) -> str:
    """Ищем колонку по списку рус/англ вариантов (точно или по подстроке)."""
    # уже стандартизированный файл? — возвращаем ключ, если он есть
    for k in keys:
        if k in df.columns:
            return k

    norm = {str(c).lower().strip(): c for c in df.columns}
    for key in keys:
        k = key.lower().strip()
        if k in norm:
            return norm[k]
        for nk, orig in norm.items():
            if k in nk:  # частичное совпадение
                return orig
    raise KeyError(f"Не удалось найти колонку из набора: {keys} в {list(df.columns)}")


# ---------- основная функция ----------
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводим датафрейм к стандартному виду:
    columns = [question_number, question_text, answer_text, score]

    Умеет работать и с «сырым» CSV из задания, и с уже обработанным,
    где колонки могли быть: question_number, question_text, answer_text, score.
    """
    cols = set(df.columns)

    # кейс: файл уже в стандарте — просто мягко нормализуем
    if {"question_number", "question_text", "answer_text", "score"}.issubset(cols):
        out = df[["question_number", "question_text", "answer_text", "score"]].copy()

        # подчистим HTML и лишние пробелы
        out["question_text"] = out["question_text"].apply(clean_html).str.replace(r"\s+", " ", regex=True).str.strip()
        out["answer_text"]   = (
            out["answer_text"]
            .fillna("").astype(str)
            .apply(clean_html)
            .apply(extract_tester_reply)  # ⚠️ извлекаем реплики тестируемого
            .str.replace(r"\s+", " ", regex=True).str.strip()
        )
        # приведение типа номера вопроса (если возможно)
        with pd.option_context("mode.chained_assignment", None):
            try:
                out["question_number"] = pd.to_numeric(out["question_number"], errors="coerce").astype("Int64")
            except Exception:
                pass
        return out

    # кейс: «сырой» файл из задания — ищем русские колонки
    qnum_col  = _find_column(df, _CANDIDATES["question_number"])
    qtxt_col  = _find_column(df, _CANDIDATES["question_text"])
    tran_col  = _find_column(df, _CANDIDATES["transcript"])
    score_col = _find_column(df, _CANDIDATES["score"])

    out = pd.DataFrame()
    out["question_number"] = df[qnum_col]
    out["question_text"]   = df[qtxt_col].apply(clean_html)
    # 1) базовое извлечение по ролям (если есть метки)
    # 2) затем более мягкая эвристика extract_tester_reply (из src.text_roles)
    out["answer_text"]     = (
        df[tran_col].apply(extract_answer)
                    .fillna("").astype(str)
                    .apply(clean_html)
                    .apply(extract_tester_reply)
    )
    out["score"]           = df[score_col]

    # финальная нормализация пробелов
    out["question_text"] = out["question_text"].str.replace(r"\s+", " ", regex=True).str.strip()
    out["answer_text"]   = out["answer_text"].str.replace(r"\s+", " ", regex=True).str.strip()

    # аккуратно приводим номер вопроса к целочисленному типу, если возможно
    with pd.option_context("mode.chained_assignment", None):
        try:
            out["question_number"] = pd.to_numeric(out["question_number"], errors="coerce").astype("Int64")
        except Exception:
            pass

    return out
