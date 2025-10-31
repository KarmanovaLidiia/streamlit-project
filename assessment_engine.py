# assessment_engine.py
import pandas as pd

# Импортируй твои шаги — подставь правильные модули:
# from src.data_cleaning import prepare_dataframe
# from src.features import build_baseline_features
# from src.features_q4 import add_q4_features
# from src.semantic_features import add_semantic_features
# from src.explanations import build_explanations
# from your_models_loader import load_models, predict_batch

# Заглушка: здесь покажу форму, ты подставишь свои вызовы
def run_inference_df(df: pd.DataFrame, with_explanations: bool = True) -> pd.DataFrame:
    data = df.copy()

    # 1) Очистка/нормализация
    # data = prepare_dataframe(data)

    # 2) Базовые фичи
    # data = build_baseline_features(data)

    # 3) Спецфичи для Q4
    # data = add_q4_features(data)

    # 4) Семантические фичи
    # data = add_semantic_features(data)

    # 5) Предсказания CatBoost по каждому вопросу
    # models = load_models("models")  # твоя реализация
    # data = predict_batch(data, models)  # должна добавить колонку predicted_score

    # 6) Клип значений по диапазонам (на всякий случай)
    if "question_number" in data.columns and "predicted_score" in data.columns:
        def clip_score(row):
            q = int(row["question_number"])
            s = float(row["predicted_score"])
            if q in (1, 3):
                return int(min(1, max(0, round(s))))
            return int(min(2, max(0, round(s))))
        data["predicted_score"] = data.apply(clip_score, axis=1)

    # 7) Объяснения (если есть)
    # if with_explanations:
    #     data = build_explanations(data)

    return data
