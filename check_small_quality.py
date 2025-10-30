import pandas as pd

df = pd.read_csv('test_small.csv', sep=';')
print("🔍 АНАЛИЗ SMALL.CSV С УЛУЧШЕННОЙ МОДЕЛЬЮ:")
print("=" * 50)

for q in [1, 4]:  # В small.csv есть только Q1 и Q4
    q_data = df[df['№ вопроса'] == q]
    if len(q_data) > 0:
        scores = q_data['pred_score']
        true_scores = q_data['Оценка экзаменатора']

        print(f"📊 Вопрос {q}:")
        print(f"   Предсказания: {scores.tolist()}")
        print(f"   Истинные: {true_scores.tolist()}")

        if len(true_scores) > 0:
            mae = (abs(true_scores - scores)).mean()
            print(f"   MAE: {mae:.3f}")
        print()