from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import shutil
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api = FastAPI(title="Scoring API")

# CORS
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML интерфейс
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Scoring API</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
        .header { text-align: center; margin-bottom: 30px; }
        .upload-area { border: 2px dashed #ccc; border-radius: 10px; padding: 40px; text-align: center; margin: 20px 0; background: #fafafa; }
        .btn { background: #007bff; color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .btn:disabled { background: #6c757d; cursor: not-allowed; }
        .btn:hover:not(:disabled) { background: #0056b3; }
        .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .status.info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Scoring API</h1>
            <p>Upload CSV file for Russian language exam scoring</p>
        </div>

        <div id="statusArea"></div>

        <div class="upload-area">
            <input type="file" id="fileInput" accept=".csv" style="display: none;">
            <h3>📁 Upload CSV File</h3>
            <p>Select a CSV file for prediction</p>
            <button class="btn" onclick="document.getElementById('fileInput').click()">Select File</button>
        </div>

        <div id="fileInfo" class="hidden">
            <h3>Selected: <span id="fileName"></span></h3>
            <button class="btn" id="predictBtn" onclick="predict()">Run Prediction</button>
        </div>

        <div id="results" class="hidden">
            <h3>📊 Results</h3>
            <pre id="resultsContent"></pre>
        </div>
    </div>

    <script>
        let currentFile = null;

        document.getElementById('fileInput').addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                if (!file.name.toLowerCase().endsWith('.csv')) {
                    showStatus('Please select a CSV file', 'error');
                    return;
                }
                currentFile = file;
                document.getElementById('fileName').textContent = file.name;
                document.getElementById('fileInfo').classList.remove('hidden');
                showStatus('File selected: ' + file.name, 'info');
            }
        });

        async function predict() {
            if (!currentFile) return;

            const predictBtn = document.getElementById('predictBtn');
            predictBtn.disabled = true;
            predictBtn.textContent = 'Processing...';

            const formData = new FormData();
            formData.append('file', currentFile);

            try {
                showStatus('Processing...', 'info');
                const response = await fetch('/predict_csv', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'predicted_' + currentFile.name;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    showStatus('✅ Download started!', 'success');
                } else {
                    const error = await response.text();
                    showStatus('❌ Error: ' + error, 'error');
                }
            } catch (error) {
                showStatus('❌ Network error: ' + error.message, 'error');
            } finally {
                predictBtn.disabled = false;
                predictBtn.textContent = 'Run Prediction';
            }
        }

        function showStatus(message, type) {
            document.getElementById('statusArea').innerHTML = '<div class="status ' + type + '">' + message + '</div>';
        }

        // Check health on load
        fetch('/health')
            .then(r => r.json())
            .then(data => showStatus('✅ API is healthy', 'success'))
            .catch(err => showStatus('❌ API connection failed', 'error'));
    </script>
</body>
</html>
"""

@api.get("/")
async def home():
    """Главная страница с UI"""
    return HTMLResponse(content=HTML_CONTENT)

@api.get("/health")
async def health():
    return {"status": "ok"}

def _pipeline_infer_bytes(csv_bytes: bytes) -> bytes:
    # ваш существующий код пайплайна
    try:
        from src.predict import pipeline_infer
    except Exception as e:
        raise RuntimeError(f"Не удалось импортировать pipeline: {e}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="tmp_api_"))
    inp = tmp_dir / "input.csv"
    outp = tmp_dir / "predicted.csv"
    inp.write_bytes(csv_bytes)
    try:
        pipeline_infer(inp, outp)
        return outp.read_bytes()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@api.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Ожидается CSV-файл")
    raw = await file.read()
    try:
        out_bytes = _pipeline_infer_bytes(raw)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Inference failed: {e}"})
    return StreamingResponse(
        iter([out_bytes]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="predicted.csv"'},
    )