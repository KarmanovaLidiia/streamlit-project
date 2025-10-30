# src/features_q4.py
import re
import pandas as pd

# мини-словарики (можно расширять)
PLACE_WORDS   = r"(кухн|парк|сквер|берег|река|дом|улиц|квартир|комнат|набережн)"
SEASON_WORDS  = r"(лето|зим|весн|осен|снег|жарко|холодно|листопад|сосулк)"
PEOPLE_WORDS  = r"(мама|папа|дедушк|бабушк|женщин|мужчин|ребен|дет|сем|дочка|сын|парень|девушк)"
ACTION_WORDS  = r"(игра|моет|готов|накрыва|бежит|катает|кормит|сидит|спит|несет|перепрыг|гуляет)"
DETAIL_WORDS  = r"(одет|рост|волос|глаз|характер|возраст|пальто|рубашк|кроссовк|плать|кофт|ботинк)"
PIC_INTRO     = r"(на картинке|на рисунке|я вижу|изображен)"

CHILDREN_Q    = r"(сколько детей|детям|о них|как.*играете.*дет(ями|ьми))"
FREE_TIME_Q   = r"(свободн(ое|ым)\s+врем|как.*проводите.*время|выходн(ой|ые))"

def q4_slot_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "question_number" not in out.columns:
        raise ValueError("В датафрейме нет колонки 'question_number'")
    for col in ["question_text", "answer_text"]:
        if col not in out.columns:
            out[col] = ""  # на всякий случай

    mask = out["question_number"] == 4
    q = out.loc[mask, "question_text"].fillna("").astype(str)
    a = out.loc[mask, "answer_text"].fillna("").astype(str)

    expects_children = q.str.contains(CHILDREN_Q, case=False, regex=True)
    expects_free     = q.str.contains(FREE_TIME_Q, case=False, regex=True)

    has_place_time = a.str.contains(PLACE_WORDS, case=False, regex=True) | a.str.contains(SEASON_WORDS, case=False, regex=True)
    has_people     = a.str.contains(PEOPLE_WORDS, case=False, regex=True)
    has_actions    = a.str_contains(ACTION_WORDS, case=False, regex=True) if hasattr(a, "str_contains") else a.str.contains(ACTION_WORDS, case=False, regex=True)
    has_detail     = a.str.contains(DETAIL_WORDS, case=False, regex=True)

    answered_children = a.str.contains(r"(у меня.*дет(и|ей)|в моей семье.*дет|сын|дочк|мы.*играем)", case=False, regex=True)
    answered_free     = a.str.contains(r"(свободн.*врем|обычно.*(в выходн|по выходн)|люблю|занимаюсь|хожу|смотрю|рисую|спорт|танц)", case=False, regex=True)

    non_cyr_ratio = a.apply(lambda t: (len(re.findall(r"[A-Za-z]", t)) / max(1, len(t))))
    picture_first = a.str.contains(PIC_INTRO, case=False, regex=True)
    dont_know     = a.str.contains(r"(не знаю|не понимаю|трудно сказать|не могу описать)", case=False, regex=True)

    slots = (has_place_time.astype(int) + has_people.astype(int) + has_actions.astype(int) + has_detail.astype(int))
    slots_covered = (slots / 4.0).clip(0, 1)

    personal_ok = (
        (expects_children & answered_children) |
        (expects_free & answered_free) |
        (~expects_children & ~expects_free)
    )

    # создаём колонки с нулями; для Q4 — заполняем значениями
    cols = {
        "q4_slots_covered": slots_covered,
        "q4_has_place_time": has_place_time.astype(int),
        "q4_has_people":     has_people.astype(int),
        "q4_has_actions":    has_actions.astype(int),
        "q4_has_detail":     has_detail.astype(int),
        "q4_expects_children": expects_children.astype(int),
        "q4_expects_free":     expects_free.astype(int),
        "q4_answered_personal": personal_ok.astype(int),
        "q4_non_cyr_ratio":   non_cyr_ratio,
        "q4_picture_first":   picture_first.astype(int),
        "q4_dont_know":       dont_know.astype(int),
    }
    for name, series in cols.items():
        out[name] = 0
        out.loc[mask, name] = series.values

    return out
