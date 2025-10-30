from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Простой HTML интерфейс
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Система оценки ответов</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 600px; margin: 0 auto; }
        .upload-form { border: 2px dashed #ccc; padding: 20px; text-align: center; }
        .btn { background: #007cba; color: white; padding: 10px 20px; border: none; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📝 Система автоматической оценки ответов</h1>
        <p>Загрузите CSV файл с ответами студентов для оценки</p>

        <form class="upload-form" action="/predict_csv" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".csv" required>
            <br><br>
            <button type="submit" class="btn">Оценить ответы</button>
        </form>

        <div style="margin-top: 30px;">
            <h3>API Endpoints:</h3>
            <ul>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/docs">API Documentation</a></li>
            </ul>
        </div>
    </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return HTML_FORM


@app.get("/ui")
async def ui_page():
    return HTMLResponse(HTML_FORM)