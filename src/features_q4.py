from __future__ import annotations
import re
import pandas as pd
from typing import List


def enhanced_q4_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Улучшенные фичи для вопроса 4 - ВСЯ ЛОГИКА В ОДНОЙ ФУНКЦИИ
    """
    out = df.copy()

    if "question_number" not in out.columns:
        raise ValueError("В датафрейме нет колонки 'question_number'")

    for col in ["question_text", "answer_text"]:
        if col not in out.columns:
            out[col] = ""

    mask = out["question_number"] == 4
    q = out.loc[mask, "question_text"].fillna("").astype(str)
    a = out.loc[mask, "answer_text"].fillna("").astype(str)

    # --- БАЗОВЫЕ ФИЧИ ---
    PLACE_WORDS = r"(?:кухн|парк|сквер|берег|река|дом|улиц|квартир|комнат|набережн)"
    SEASON_WORDS = r"(?:лето|зим|весн|осен|снег|жарко|холодно|листопад|сосулк)"
    PEOPLE_WORDS = r"(?:мама|папа|дедушк|бабушк|женщин|мужчин|ребен|дет|сем|дочка|сын|парень|девушк)"
    ACTION_WORDS = r"(?:игра|моет|готов|накрыва|бежит|катает|кормит|сидит|спит|несет|перепрыг|гуляет)"
    DETAIL_WORDS = r"(?:одет|рост|волос|глаз|характер|возраст|пальто|рубашк|кроссовк|плать|кофт|ботинк)"
    PIC_INTRO = r"(?:на картинке|на рисунке|я вижу|изображен)"

    CHILDREN_Q = r"(?:сколько детей|детям|о них|как.*играете.*дет(?:ями|ьми))"
    FREE_TIME_Q = r"(?:свободн(?:ое|ым)\s+врем|как.*проводите.*время|выходн(?:ой|ые))"

    # Базовые детекции
    has_place_time = a.str.contains(PLACE_WORDS, case=False, regex=True) | a.str.contains(SEASON_WORDS, case=False,
                                                                                          regex=True)
    has_people = a.str.contains(PEOPLE_WORDS, case=False, regex=True)
    has_actions = a.str.contains(ACTION_WORDS, case=False, regex=True)
    has_detail = a.str.contains(DETAIL_WORDS, case=False, regex=True)

    expects_children = q.str.contains(CHILDREN_Q, case=False, regex=True)
    expects_free = q.str.contains(FREE_TIME_Q, case=False, regex=True)

    answered_children = a.str.contains(r"(?:у меня.*дет(?:и|ей)|в моей семье.*дет|сын|дочк|мы.*играем)", case=False,
                                       regex=True)
    answered_free = a.str.contains(
        r"(?:свободн.*врем|обычно.*(?:в выходн|по выходн)|люблю|занимаюсь|хожу|смотрю|рисую|спорт|танц)", case=False,
        regex=True)

    non_cyr_ratio = a.apply(lambda t: (len(re.findall(r"[A-Za-z]", t)) / max(1, len(t))))
    non_cyr_ratio = pd.to_numeric(non_cyr_ratio, errors="coerce").fillna(0.0).astype(float)

    picture_first = a.str.contains(PIC_INTRO, case=False, regex=True)
    dont_know = a.str.contains(r"(?:не знаю|не понимаю|трудно сказать|не могу описать)", case=False, regex=True)

    # --- УЛУЧШЕННЫЕ ФИЧИ ---

    # 1. Детекция всех подвопросов
    subquestion_patterns = {
        'place_time': r'(место|время|сезон|лето|зима|весна|осень|кухня|парк|улица)',
        'people': r'(люди|человек|мужчина|женщина|ребенок|дети|семья|бабушка|дедушка)',
        'actions': r'(делает|стоит|сидит|играет|готовит|моет|читает|смотрит)',
        'person_detail': r'(одет|носит|платье|рубашка|брюки|волосы|глаза)',
        'family_children': r'(дет[еи]|семь[яеи]|сын|дочь|брат|сестра)',
        'playing': r'(игра[ею]|играем|гуля[ею]|занимаюсь)'
    }

    for name, pattern in subquestion_patterns.items():
        out.loc[mask, f'q4_has_{name}'] = a.str.contains(pattern, case=False, regex=True).astype(int)

    # 2. Структура ответа
    def analyze_structure(text: str) -> dict:
        text_lower = text.lower()

        has_intro = any(marker in text_lower for marker in [
            'на картинке', 'на рисунке', 'изображен', 'вижу', 'показан'
        ])

        has_personal = any(marker in text_lower for marker in [
            'у меня', 'в моей', 'мои', 'я ', 'мы ', 'наш'
        ])

        sentences = re.split(r'[.!?]+', text)
        num_sentences = len([s for s in sentences if len(s.strip()) > 10])

        return {
            'has_intro': has_intro,
            'has_personal': has_personal,
            'num_sentences': num_sentences
        }

    structure_features = a.apply(analyze_structure).apply(pd.Series)
    out.loc[mask, 'q4_has_intro'] = structure_features['has_intro'].astype(int)
    out.loc[mask, 'q4_has_personal'] = structure_features['has_personal'].astype(int)
    out.loc[mask, 'q4_num_sentences'] = structure_features['num_sentences']

    # 3. Полнота ответа
    subq_columns = [f'q4_has_{name}' for name in subquestion_patterns.keys()]
    out.loc[mask, 'q4_coverage_ratio'] = out.loc[mask, subq_columns].sum(axis=1) / len(subq_columns)

    # 4. Базовые слоты (оригинальные фичи)
    slots = (has_place_time.astype(int) + has_people.astype(int) + has_actions.astype(int) + has_detail.astype(int))
    slots_covered = (slots / 4.0).clip(0, 1)

    personal_ok = (
            (expects_children & answered_children) |
            (expects_free & answered_free) |
            (~expects_children & ~expects_free)
    )

    # Заполняем базовые фичи
    float_cols = ["q4_slots_covered", "q4_non_cyr_ratio"]
    int_cols = [
        "q4_has_place_time", "q4_has_people", "q4_has_actions", "q4_has_detail",
        "q4_expects_children", "q4_expects_free", "q4_answered_personal",
        "q4_picture_first", "q4_dont_know"
    ]

    for name in float_cols:
        out[name] = 0.0
    for name in int_cols:
        out[name] = 0

    out.loc[mask, "q4_slots_covered"] = slots_covered.astype(float).to_numpy()
    out.loc[mask, "q4_has_place_time"] = has_place_time.astype(int).to_numpy()
    out.loc[mask, "q4_has_people"] = has_people.astype(int).to_numpy()
    out.loc[mask, "q4_has_actions"] = has_actions.astype(int).to_numpy()
    out.loc[mask, "q4_has_detail"] = has_detail.astype(int).to_numpy()
    out.loc[mask, "q4_expects_children"] = expects_children.astype(int).to_numpy()
    out.loc[mask, "q4_expects_free"] = expects_free.astype(int).to_numpy()
    out.loc[mask, "q4_answered_personal"] = personal_ok.astype(int).to_numpy()
    out.loc[mask, "q4_non_cyr_ratio"] = non_cyr_ratio.astype(float).to_numpy()
    out.loc[mask, "q4_picture_first"] = picture_first.astype(int).to_numpy()
    out.loc[mask, "q4_dont_know"] = dont_know.astype(int).to_numpy()

    return out


# Совместимость с существующим кодом
def add_q4_features(df: pd.DataFrame) -> pd.DataFrame:
    """Совместимое имя функции для predict.py"""
    return enhanced_q4_features(df)