# src/debug_cache.py
from sentence_transformers import SentenceTransformer
from src.semantic_cache import embed_with_cache
from pathlib import Path

if __name__ == "__main__":
    print("🔧 Smoke-test кэша")
    model = SentenceTransformer("ai-forever/sbert_large_nlu_ru")
    texts = ["привет", "как дела", "экзамен по русскому языку", "описание картинки в парке"]
    vecs = embed_with_cache(texts, model, batch_size=4, verbose=True)
    print("OK, vecs:", vecs.shape)
