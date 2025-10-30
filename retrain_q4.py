import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import sys
import os

sys.path.append('src')

from features_q4 import enhanced_q4_features
from features import build_baseline_features
from semantic_features import add_semantic_similarity
from data_cleaning import prepare_dataframe


def retrain_q4_model():
    print("🔄 Переобучение модели Q4 с улучшенными фичами...")

    # 1. Загрузи данные
    df = pd.read_csv('data/raw/Данные для кейса.csv', sep=';')
    print(f"📊 Загружено {len(df)} строк")

    # 2. Подготовь данные только для Q4
    df_clean = prepare_dataframe(df)
    df_q4 = df_clean[df_clean['question_number'] == 4]
    print(f"📋 Q4 данных: {len(df_q4)} строк")

    # 3. Построй все фичи
    print("🔨 Строим фичи...")
    feats = build_baseline_features(df_q4)
    feats = add_semantic_similarity(feats, verbose=False)
    feats = enhanced_q4_features(feats)

    # 4. Выдели фичи и целевую переменную
    feature_cols = [c for c in feats.columns if c.startswith('q4_') or c in [
        'semantic_sim', 'ans_len_words', 'ans_n_sents', 'ans_ttr',
        'ans_short_sent_rt', 'ans_punct_rt', 'q_len_words'
    ]]

    X = feats[feature_cols].fillna(0)
    y = feats['score'].fillna(0)

    print(f"🎯 Фичей: {len(feature_cols)}, Примеров: {len(X)}")
    print(f"📈 Фичи: {feature_cols}")

    # 5. Обучи новую модель
    print("🤖 Обучаем CatBoost...")
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        verbose=100,
        random_state=42
    )

    model.fit(X, y)

    # 6. Сохрани модель
    model.save_model('models/catboost_Q4_enhanced.cbm')
    print("✅ Модель Q4 переобучена с улучшенными фичами!")

    # 7. Проверим важность фич
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)

    print("\n📊 Важность фич:")
    print(feature_importance.head(10))


if __name__ == "__main__":
    retrain_q4_model()