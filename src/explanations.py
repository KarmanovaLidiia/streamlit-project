from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List


def add_score_explanations(feats: pd.DataFrame, pred_scores: np.ndarray) -> pd.DataFrame:
    """
    Добавляет объяснения для предсказанных оценок на основе фичей
    """
    out = feats.copy()
    explanations = []

    for idx, row in feats.iterrows():
        q_num = row['question_number']
        pred_score = pred_scores[idx]
        explanation_parts = []

        # Базовые метрики
        if row.get('ans_len_words', 0) < 10:
            explanation_parts.append("🔴 Короткий ответ")
        elif row.get('ans_len_words', 0) > 50:
            explanation_parts.append("🟢 Развернутый ответ")
        else:
            explanation_parts.append("🟡 Средняя длина ответа")

        # Семантическое сходство
        semantic_sim = row.get('semantic_sim', 0)
        if semantic_sim > 0.7:
            explanation_parts.append("✅ Высокое смысловое соответствие")
        elif semantic_sim > 0.4:
            explanation_parts.append("⚠️  Умеренное смысловое соответствие")
        else:
            explanation_parts.append("❌ Низкое смысловое соответствие")

        # Структура ответа
        if row.get('ans_n_sents', 0) >= 3:
            explanation_parts.append("📊 Хорошая структура ответа")
        else:
            explanation_parts.append("📉 Мало предложений")

        # Специфичные для вопросов объяснения
        if q_num == 4:
            # Для вопроса 4 - описание картинки
            if row.get('q4_has_intro', 0) == 1:
                explanation_parts.append("🎨 Есть вступление с описанием картинки")
            if row.get('q4_has_personal', 0) == 1:
                explanation_parts.append("👤 Есть личный опыт")
            if row.get('q4_coverage_ratio', 0) > 0.6:
                explanation_parts.append("📋 Хорошее покрытие подвопросов")

        elif q_num in [1, 3]:
            # Для диалоговых вопросов
            if row.get('ans_ttr', 0) > 0.6:
                explanation_parts.append("💬 Разнообразная лексика")

        elif q_num == 2:
            # Для вопросов про жилье
            if row.get('ans_len_words', 0) > 30:
                explanation_parts.append("🏠 Подробное описание")

        # Оценка качества
        if pred_score >= (2.0 if q_num in [2, 4] else 1.0) * 0.8:
            explanation_parts.append("⭐ Высокий балл")
        elif pred_score >= (2.0 if q_num in [2, 4] else 1.0) * 0.5:
            explanation_parts.append("⚡ Средний балл")
        else:
            explanation_parts.append("💤 Низкий балл")

        explanations.append(" | ".join(explanation_parts))

    out['score_explanation'] = explanations
    return out