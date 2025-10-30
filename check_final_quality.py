import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# Читаем наши предсказания
df = pd.read_csv('test_output.csv', delimiter=';', encoding='utf-8-sig')

# Фильтруем только строки с истинными оценками
df_with_truth = df[df['Оценка экзаменатора'].notna()]

if len(df_with_truth) > 0:
    true_scores = df_with_truth['Оценка экзаменатора']
    pred_scores = df_with_truth['pred_score']

    mae_total = mean_absolute_error(true_scores, pred_scores)
    print(f'📊 ОБЩЕЕ КАЧЕСТВО (MAE): {mae_total:.3f} балла')
    print()

    # По типам вопросов
    for q in [1, 2, 3, 4]:
        q_data = df_with_truth[df_with_truth['№ вопроса'] == q]
        if len(q_data) > 0:
            mae_q = mean_absolute_error(q_data['Оценка экзаменатора'], q_data['pred_score'])
            count_q = len(q_data)
            print(f'  Вопрос {q}: MAE = {mae_q:.3f} балла (примеров: {count_q})')
else:
    print('❌ Нет данных с истинными оценками для проверки')