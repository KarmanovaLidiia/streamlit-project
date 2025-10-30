import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')


class RussianFeatureExtractor:
    """Исправленная версия экстрактора признаков с работающим composite_quality_score"""

    def __init__(self, use_heavy_models: bool = False):
        print("Инициализация исправленного экстрактора признаков...")

        self.use_heavy_models = use_heavy_models
        self.sbert_model = None

        # Инициализация моделей
        self._initialize_models()

        # Списки ключевых слов
        self.greeting_words = ['здравствуйте', 'привет', 'добрый', 'здравствуй', 'доброе', 'приветствую']
        self.question_words = ['как', 'что', 'где', 'когда', 'почему', 'можно', 'сколько', 'какой', 'какая']
        self.descriptive_words = ['вижу', 'изображен', 'находится', 'делает', 'одет', 'стоит', 'сидит']
        self.connector_words = ['потому что', 'поэтому', 'так как', 'например', 'кроме того']
        self.emotional_words = ['красиво', 'интересно', 'замечательно', 'прекрасно', 'нравится']
        self.spatial_words = ['слева', 'справа', 'вверху', 'внизу', 'рядом', 'около']

        print("✅ Инициализация завершена!")

    def _initialize_models(self):
        """Инициализация моделей"""
        if self.use_heavy_models:
            print("ℹ️ Тяжелые модели отключены для стабильности")
        print("ℹ️ Используем легкие методы (TF-IDF)")

    def clean_text(self, text: str) -> str:
        """Очистка текста"""
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\sа-яА-ЯёЁ.,!?;:()-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_basic_features(self, text: str) -> Dict[str, float]:
        """Базовые текстовые признаки"""
        text_clean = self.clean_text(text)

        if not text_clean:
            return {
                'text_length': 0, 'word_count': 0, 'sentence_count': 0,
                'avg_word_length': 0, 'lexical_diversity': 0,
                'has_questions': 0, 'has_exclamations': 0
            }

        # Базовые метрики
        words = re.findall(r'\b[а-яёa-z]+\b', text_clean.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', text_clean) if s.strip()]

        word_count = len(words)
        text_length = len(text_clean)
        sentence_count = len(sentences)

        features = {
            'text_length': text_length,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': sum(len(w) for w in words) / max(word_count, 1),
            'lexical_diversity': len(set(words)) / max(word_count, 1),
            'has_questions': int('?' in text_clean),
            'has_exclamations': int('!' in text_clean),
        }

        return features

    def extract_semantic_features(self, question: str, answer: str) -> Dict[str, float]:
        """Семантические признаки"""
        question_clean = self.clean_text(question)
        answer_clean = self.clean_text(answer)

        features = {
            'keyword_overlap': 0.0,
            'response_relevance': 0.0
        }

        if not answer_clean or not question_clean:
            return features

        try:
            # Упрощенный анализ ключевых слов
            question_words = set(re.findall(r'\b[а-яё]+\b', question_clean.lower()))
            answer_words = set(re.findall(r'\b[а-яё]+\b', answer_clean.lower()))

            if question_words:
                common_words = question_words.intersection(answer_words)
                features['keyword_overlap'] = len(common_words) / max(len(question_words), 1)
                features['response_relevance'] = min(1.0, len(answer_words) / max(len(question_words), 1))

        except Exception as e:
            print(f"Ошибка семантических признаков: {e}")

        return features

    def extract_grammar_features(self, text: str) -> Dict[str, float]:
        """Грамматические признаки"""
        text_clean = self.clean_text(text)

        features = {
            'grammar_quality': 0.5,  # Базовая оценка
            'has_punctuation': 0.0,
            'sentence_completeness': 0.0
        }

        if not text_clean:
            return features

        sentences = [s.strip() for s in re.split(r'[.!?]+', text_clean) if s.strip()]
        words = text_clean.split()

        if sentences:
            # Проверка пунктуации
            features['has_punctuation'] = 1.0 if any(mark in text_clean for mark in '.!?') else 0.0

            # Полнота предложений
            complete_sentences = sum(1 for s in sentences if len(s.split()) >= 3)
            features['sentence_completeness'] = complete_sentences / max(len(sentences), 1)

            # Улучшенная эвристика грамматического качества
            grammar_score = 0.0
            grammar_score += features['has_punctuation'] * 0.3
            grammar_score += features['sentence_completeness'] * 0.4

            # Дополнительные эвристики
            if len(words) > 5:
                avg_sentence_len = len(words) / len(sentences)
                if 5 <= avg_sentence_len <= 20:
                    grammar_score += 0.2
                elif avg_sentence_len > 20:
                    grammar_score += 0.1

            features['grammar_quality'] = min(1.0, grammar_score)

        return features

    def extract_style_features(self, text: str) -> Dict[str, float]:
        """Стилистические признаки"""
        text_clean = self.clean_text(text).lower()

        features = {
            'has_greeting': 0.0,
            'has_description': 0.0,
            'has_connectors': 0.0,
            'has_emotional_words': 0.0,
            'style_score': 0.0
        }

        if not text_clean:
            return features

        # Стилистические маркеры
        features.update({
            'has_greeting': float(any(greet in text_clean for greet in self.greeting_words)),
            'has_description': float(any(desc in text_clean for desc in self.descriptive_words)),
            'has_connectors': float(any(conn in text_clean for conn in self.connector_words)),
            'has_emotional_words': float(any(emot in text_clean for emot in self.emotional_words)),
        })

        # Оценка стиля
        style_indicators = sum([
            features['has_greeting'],
            features['has_connectors'],
            features['has_emotional_words']
        ])
        features['style_score'] = min(1.0, style_indicators / 3)

        return features

    def extract_quality_features(self, text: str, question_type: int) -> Dict[str, float]:
        """Признаки качества ответа"""
        text_clean = self.clean_text(text)
        words = text_clean.split()
        word_count = len(words)

        features = {
            'answer_length_sufficiency': min(1.0, word_count / 30),  # Нормализованная длина
            'content_richness': 0.0,
            'engagement_level': 0.0
        }

        if not text_clean:
            return features

        # Богатство контента (лексическое разнообразие + длина)
        lexical_diversity = len(set(words)) / max(word_count, 1)
        features['content_richness'] = min(1.0, (lexical_diversity + features['answer_length_sufficiency']) / 2)

        # Уровень вовлеченности
        engagement = 0.0
        engagement += features['answer_length_sufficiency'] * 0.4
        engagement += lexical_diversity * 0.3
        engagement += (1.0 if '?' in text_clean else 0.0) * 0.3
        features['engagement_level'] = engagement

        return features

    def extract_all_features(self, row: pd.Series) -> Dict[str, float]:
        """Извлечение всех признаков - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
        try:
            # Безопасное извлечение данных
            question = row.get('Текст вопроса', row.get('Вопрос', ''))
            answer = row.get('Транскрибация ответа', row.get('Транскрипт', row.get('Ответ', '')))
            question_type = row.get('№ вопроса', row.get('Тип вопроса', 1))

            try:
                question_type = int(question_type)
            except:
                question_type = 1

            features = {}

            # 1. Базовые признаки (надежные)
            basic_features = self.extract_basic_features(answer)
            features.update(basic_features)

            # 2. Семантические признаки
            semantic_features = self.extract_semantic_features(question, answer)
            features.update(semantic_features)

            # 3. Грамматические признаки
            grammar_features = self.extract_grammar_features(answer)
            features.update(grammar_features)

            # 4. Стилистические признаки
            style_features = self.extract_style_features(answer)
            features.update(style_features)

            # 5. Признаки качества
            quality_features = self.extract_quality_features(answer, question_type)
            features.update(quality_features)

            # 6. Тип вопроса
            features['question_type'] = float(question_type)

            # 7. ИСПРАВЛЕННЫЙ композитный показатель
            features['composite_quality_score'] = self._calculate_quality_score(features)

            return features

        except Exception as e:
            print(f"❌ Ошибка при извлечении признаков: {e}")
            # Возвращаем базовые признаки
            return self._get_fallback_features()

    def _calculate_quality_score(self, features: Dict[str, float]) -> float:
        """ИСПРАВЛЕННЫЙ расчет качества ответа"""

        # Веса для разных категорий
        weights = {
            # Семантика и релевантность (35%)
            'keyword_overlap': 0.20,
            'response_relevance': 0.15,

            # Грамматика и структура (25%)
            'grammar_quality': 0.15,
            'sentence_completeness': 0.10,

            # Стиль и вовлеченность (25%)
            'style_score': 0.10,
            'engagement_level': 0.15,

            # Содержание (15%)
            'content_richness': 0.15
        }

        total_score = 0.0
        total_weight = 0.0

        for feature, weight in weights.items():
            if feature in features:
                value = features[feature]
                total_score += value * weight
                total_weight += weight

        # Нормализация на случай отсутствующих признаков
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.5  # нейтральная оценка

        return min(1.0, max(0.0, final_score))

    def _get_fallback_features(self) -> Dict[str, float]:
        """Базовые признаки при ошибке"""
        return {
            'text_length': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_word_length': 0, 'lexical_diversity': 0,
            'has_questions': 0, 'has_exclamations': 0,
            'keyword_overlap': 0, 'response_relevance': 0,
            'grammar_quality': 0.5, 'has_punctuation': 0, 'sentence_completeness': 0,
            'has_greeting': 0, 'has_description': 0, 'has_connectors': 0,
            'has_emotional_words': 0, 'style_score': 0,
            'answer_length_sufficiency': 0, 'content_richness': 0, 'engagement_level': 0,
            'question_type': 1, 'composite_quality_score': 0.5
        }

    def extract_features_for_dataframe(self, df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
        """Извлечение признаков для датафрейма"""
        if sample_size and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)
            print(f"Взята выборка: {len(df)} строк")

        print(f"Извлечение признаков для {len(df)} строк...")
        features_list = []
        successful = 0

        for idx, row in df.iterrows():
            if idx % 50 == 0 and idx > 0:
                print(f"Обработано {idx}/{len(df)} строк...")

            try:
                features = self.extract_all_features(row)
                features['original_index'] = idx
                features_list.append(features)
                successful += 1
            except Exception as e:
                print(f"❌ Ошибка в строке {idx}: {e}")
                continue

        if features_list:
            features_df = pd.DataFrame(features_list)
            features_df.set_index('original_index', inplace=True)

            success_rate = successful / len(df)
            print(f"✅ Извлечение завершено! Успешно: {successful}/{len(df)} ({success_rate:.1%})")

            return features_df
        else:
            print("❌ Не удалось извлечь признаки")
            return pd.DataFrame()


# Быстрая функция для тестирования
def extract_quick_features(text: str) -> Dict[str, float]:
    extractor = RussianFeatureExtractor()
    return extractor.extract_basic_features(text)


if __name__ == "__main__":
    # Тест исправленной версии
    extractor = RussianFeatureExtractor()
    test_data = {
        'Текст вопроса': ['Расскажите о вашем городе'],
        'Транскрибация ответа': ['Привет! Я живу в Москве. Это большой и красивый город с множеством парков и музеев.'],
        '№ вопроса': [1]
    }
    test_df = pd.DataFrame(test_data)
    features = extractor.extract_all_features(test_df.iloc[0])

    print("🎯 ТЕСТ ИСПРАВЛЕННОЙ ВЕРСИИ:")
    print(f"Композитный показатель: {features['composite_quality_score']:.3f}")
    print(f"Грамматическое качество: {features['grammar_quality']:.3f}")
    print(f"Стилевой показатель: {features['style_score']:.3f}")
    print(f"Количество слов: {features['word_count']}")