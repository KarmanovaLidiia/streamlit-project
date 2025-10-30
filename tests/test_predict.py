import pandas as pd
from pathlib import Path
from src.predict import predict_dataframe

def test_predict_smoke():
    df = pd.read_csv("data/raw/small.csv", sep=";", encoding="utf-8-sig")
    out = predict_dataframe(df.copy())
    assert "pred_score" in out.columns
    assert len(out) == len(df)
