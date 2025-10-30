from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import requests

app = FastAPI(title="Scoring UI")
templates = Jinja2Templates(directory="templates")

# 🔧 Локальный адрес FastAPI-сервера
API_URL = "http://localhost:8000/predict_csv"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    return {"status": "ok", "service": "scoring-ui"}

@app.post("/predict")
async def predict_csv(file: UploadFile = File(...)):
    files = {"file": (file.filename, await file.read(), file.content_type)}
    try:
        resp = requests.post(API_URL, files=files, timeout=1800)
        resp.raise_for_status()
        return StreamingResponse(
            iter([resp.content]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="predicted_{file.filename}"'}
        )
    except Exception as e:
        return {"error": str(e)}
