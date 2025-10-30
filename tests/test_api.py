# tests/test_api.py
from pathlib import Path
import io
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_predict_csv_endpoint():
    # делаем маленький CSV на лету
    df = pd.DataFrame({
        "Id экзамена": [1,1],
        "Id вопроса": [11,12],
        "№ вопроса":  [1,2],
        "Текст вопроса": ["Привет", "Где живёте?"],
        "Транскрибация ответа": ["Здравствуйте, я готов", "Я живу в общежитии"],
        "Оценка экзаменатора": [1, 2]
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig", sep=";")
    buf.seek(0)

    files = {"file": ("mini.csv", buf.getvalue(), "text/csv")}
    r = client.post("/predict_csv", files=files)
    assert r.status_code == 200
    # ответ — это CSV-байты; проверим, что в нём есть столбец pred_score
    content = r.content.decode("utf-8-sig")
    assert "pred_score" in content
