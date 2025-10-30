import pandas as pd
from sentence_transformers import SentenceTransformer
import language_tool_python


class FeatureExtractor:
    def __init__(self):
        self.sbert_model = SentenceTransformer('sbert_large_nlu_ru')
        self.grammar_tool = language_tool_python.LanguageTool('ru')

    def extract_semantic_features(self, question, answer):
        """Извлечение семантических признаков"""
        # Реализация анализа содержания
        pass

    def extract_grammar_features(self, text):
        """Извлечение грамматических признаков"""
        # Реализация анализа грамматики
        pass

    def extract_all_features(self, df):
        """Извлечение всех признаков для датасета"""
        # Применение ко всем строкам
        pass