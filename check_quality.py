import pandas as pd
import numpy as np

# Загрузи скачанный файл
df = pd.read_csv('predicted_from_api.csv', sep=';')

print("📊 АНАЛИЗ КАЧЕСТВА ПРЕДСКАЗАНИЙ")
print("=" * 50)

# Проверяем наличие колонок
if 'Оценка экзаменатора' in df.columns and 'pred_score' in df.columns:
    # Убираем строки где нет истинных оценок
    df_clean = df.dropna(subset=['Оценка экзаменатора'])

    if len(df_clean) > 0:
        true_scores = df_clean['Оценка экзаменатора'].astype(float)
        pred_scores = df_clean['pred_score'].astype(float)

        # Основные метрики
        mae = (abs(true_scores - pred_scores)).mean()
        rmse = ((true_scores - pred_scores) ** 2).mean() ** 0.5

        print(f"📈 Общие метрики:")
        print(f"   MAE (средняя абсолютная ошибка): {mae:.3f}")
        print(f"   RMSE (среднеквадратичная ошибка): {rmse:.3f}")
        print(f"   Корреляция: {true_scores.corr(pred_scores):.3f}")

        # По вопросам
        print(f"\n📋 По типам вопросов:")
        for q in [1, 2, 3, 4]:
            mask = df_clean['№ вопроса'] == q
            if mask.any():
                q_true = true_scores[mask]
                q_pred = pred_scores[mask]
                q_mae = (abs(q_true - q_pred)).mean()

                # Диапазон баллов для вопроса
                if q in [1, 3]:
                    max_score = 1
                else:
                    max_score = 2

                print(f"   Вопрос {q} (0-{max_score}): MAE = {q_mae:.3f}, примеров = {len(q_true)}")

    else:
        print("❌ В файле нет строк с оценками экзаменатора")

else:
    print("❌ В файле отсутствуют колонки 'Оценка экзаменатора' или 'pred_score'")

# Статистика предсказаний
print(f"\n📊 Статистика предсказаний:")
for q in [1, 2, 3, 4]:
    mask = df['№ вопроса'] == q
    if mask.any():
        scores = df.loc[mask, 'pred_score'].astype(float)
        print(f"   Вопрос {q}: ср.={scores.mean():.2f}, мин={scores.min():.2f}, макс={scores.max():.2f}")