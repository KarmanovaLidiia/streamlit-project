# src/text_roles.py
import re

TESTER_MARKERS = [
    r"^(?:тестируемый|кандидат|студент)\s*[:\-]\s*",
    r"^(?:я|ну|значит|смотрите)\b",  # мягкая эвристика на начало своей реплики
]
EXAMINER_MARKERS = [
    r"^(?:экзаменатор|собеседник|преподаватель|интервьюер)\s*[:\-]\s*",
    r"^(?:вопрос|подскажите|расскажите|опишите)\b",  # часто задают вопрос
]

def extract_tester_reply(transcript: str) -> str:
    """
    Пытаемся оставить только реплики тестируемого из общей транскрибации.
    Простая эвристика: разбиваем по строкам / точкам с запятой / переносам,
    фильтруем явные метки 'Экзаменатор:' и оставляем нейтральные предложения.
    """
    if not isinstance(transcript, str) or not transcript.strip():
        return ""

    # разбивка на строки/квази-реплики
    parts = re.split(r"(?:\r?\n|[.;]{1}\s+)", transcript)
    cleaned = []
    for p in parts:
        t = p.strip()
        if not t:
            continue

        # отбрасываем явные реплики экзаменатора
        if any(re.match(pat, t, flags=re.IGNORECASE) for pat in EXAMINER_MARKERS):
            continue

        # если явно помечен тестируемый — оставляем без метки
        for pat in TESTER_MARKERS:
            t = re.sub(pat, "", t, flags=re.IGNORECASE).strip()

        # отбрасываем чисто служебные фразы
        if re.fullmatch(r"(хорошо|угу|да|нет|ладно|понятно)", t, flags=re.IGNORECASE):
            continue

        cleaned.append(t)

    # если после фильтра пусто — вернём исходник (лучше не потерять текст)
    return " ".join(cleaned) if cleaned else transcript.strip()
