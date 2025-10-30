# src/on_topic.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "features_with_semantics_q4.csv"
MODEL_PATH = ROOT / "models" / "on_topic.pkl"

FEATURES = ["semantic_sim","ans_len_words","ans_ttr","ans_avg_sent_len"]

def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    # бинарная цель: >0 считается «по теме»
    y = (df["score"] > 0).astype(int).values
    X = df[FEATURES].fillna(0).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=200, n_jobs=None)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:,1]
    print(f"AUC on holdout: {roc_auc_score(yte, p):.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": FEATURES}, MODEL_PATH)
    print(f"✅ on_topic model saved: {MODEL_PATH}")

if __name__ == "__main__":
    main()
