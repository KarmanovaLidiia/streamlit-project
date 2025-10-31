import os
import json
import time
import zipfile
import threading
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import pipeline_infer

# --------------------------- UI / PAGE ---------------------------
st.set_page_config(
    page_title="Автоматическая оценка ответа иностранного гражданина",
    layout="centered"
)

st.title("Автоматическая оценка ответа иностранного гражданина")
st.caption("Загрузите CSV как в задании. На выходе получите файл с колонкой `pred_score`.")

# Небольшое предупреждение про крупные файлы
st.info(
    "⚠️ Загрузка больших файлов (>10 МБ) может занять 1–3 минуты из-за сети. "
    "После выбора файла дождитесь появления сообщения об успешной загрузке."
)

# Опции ускорения (не влияет на качество полной обработки)
with st.expander("⚙️ Параметры запуска", expanded=True):
    fast_mode = st.checkbox(
        "Быстрый прогон (для проверки)", value=True,
        help="Обрабатывает только первые N строк, чтобы быстро проверить пайплайн."
    )
    row_limit = st.number_input(
        "Сколько строк взять в быстром режиме", min_value=100, max_value=100_000,
        value=2_000, step=100
    )
    st.caption(
        "Быстрый режим не меняет качество полноценной обработки — "
        "он просто делает сэмпл первых N строк."
    )

# Обёртка вокруг uploader, чтобы показать короткий спиннер и инфо после выбора файла
placeholder = st.empty()
uploaded = placeholder.file_uploader("Загрузите CSV (или ZIP с CSV внутри)", type=["csv", "zip"])
if uploaded is not None:
    with st.spinner("📦 Файл загружается..."):
        time.sleep(1.0)  # маленькая задержка, чтобы UI не казался «замёрзшим»
    size_mb = getattr(uploaded, "size", 0) / 1e6 if hasattr(uploaded, "size") else 0
    st.success(f"✅ Файл `{uploaded.name}` загружен ({size_mb:.1f} МБ).")
    placeholder.empty()

run = st.button("Оценить", type="primary", disabled=uploaded is None)

# --------------------------- RUNTIME PATHS ---------------------------
TMP_DIR_IN = Path("data/raw")
TMP_DIR_OUT = Path("data/processed")
OUTPUTS_DIR = Path("data/outputs")  # <-- новые автосохранения сюда
TMP_DIR_IN.mkdir(parents=True, exist_ok=True)
TMP_DIR_OUT.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

TMP_IN = TMP_DIR_IN / "tmp_input.csv"
TMP_OUT = TMP_DIR_OUT / "tmp_output.csv"

# Файл прогресса, его пишет src/predict.py
PROGRESS_FILE = Path("/tmp/progress.json")
os.environ["PROGRESS_FILE"] = str(PROGRESS_FILE)

# Веса стадий для общего прогресса (сумма ≈ 100)
STAGE_WEIGHTS = {
    "загрузка CSV": 2,
    "очистка": 6,
    "базовые фичи": 6,
    "семантика": 55,
    "q4-фичи": 10,
    "on-topic": 2,
    "инференс CatBoost": 15,
    "объяснения": 3,
    "сохранение": 1,
    "готово": 0,
}

def _save_bytes_to(tmp_path: Path, raw: bytes):
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(raw)

def _read_any_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Универсальное чтение CSV с авто-детектом разделителя."""
    try:
        return pd.read_csv(path, nrows=nrows, encoding="utf-8-sig", sep=None, engine="python")
    except Exception:
        return pd.read_csv(path, nrows=nrows, encoding="utf-8", sep=None, engine="python")

def _write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def _calc_total_progress(cur_stage: str, current: int, total: int) -> float:
    """Перевод локального прогресса стадии в общий % с учётом весов."""
    stages = list(STAGE_WEIGHTS.keys())
    total_weight = sum(STAGE_WEIGHTS.values()) or 100.0
    done_weight = 0.0
    for s in stages:
        if s == cur_stage:
            break
        done_weight += STAGE_WEIGHTS.get(s, 0)
    stage_weight = STAGE_WEIGHTS.get(cur_stage, 0)
    frac = 0.0
    if total and total > 0:
        frac = min(max(current / float(total), 0.0), 1.0)
    return (done_weight + stage_weight * frac) / total_weight

def _read_progress():
    """Безопасное чтение прогресса из файла."""
    try:
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"stage": "подготовка", "current": 0, "total": 1, "note": ""}

def _extract_zip_first_csv_to(dest_csv_path: Path, uploaded_file) -> bool:
    """
    Если загружен ZIP — извлечь **первый** CSV прямо в dest_csv_path.
    Возвращает True, если ZIP обработан и CSV записан; иначе False.
    """
    if not uploaded_file.name.lower().endswith(".zip"):
        return False
    try:
        with zipfile.ZipFile(uploaded_file) as zf:
            csv_name = None
            for name in zf.namelist():
                if name.lower().endswith(".csv"):
                    csv_name = name
                    break
            if csv_name is None:
                st.error("В ZIP не найден CSV-файл.")
                st.stop()
            with zf.open(csv_name) as src, open(dest_csv_path, "wb") as dst:
                dst.write(src.read())
        return True
    except zipfile.BadZipFile:
        st.error("Не удалось открыть ZIP-файл (повреждён?).")
        st.stop()
    return False

def _list_outputs(max_items: int = 10):
    """Последние сохранённые результаты (для блока истории)."""
    files = sorted(
        OUTPUTS_DIR.glob("predicted-*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return files[:max_items]

# --------------------------- MAIN ---------------------------
if uploaded and run:
    start_time = time.time()
    # очистим старый прогресс
    try:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    with st.status("⏳ Подготовка к запуску...", expanded=True) as status:
        st.write("Сохраняю и подготавливаю файл…")

        # Если это ZIP — извлечём CSV, иначе — сохраняем байты
        handled_zip = _extract_zip_first_csv_to(TMP_IN, uploaded)
        if handled_zip:
            st.info(f"📦 ZIP распакован, найден CSV → `{TMP_IN.name}`.")
        else:
            raw_bytes = uploaded.read()
            _save_bytes_to(TMP_IN, raw_bytes)

        # Быстрый прогон — перезапишем TMP_IN первыми N строками (только если не ZIP)
        if fast_mode and not handled_zip:
            st.write(f"Читаю первые {row_limit} строк для быстрого прогона…")
            try:
                df_head = _read_any_csv(TMP_IN, nrows=int(row_limit))
                _write_csv(df_head, TMP_IN)
                st.success(f"Подготовлен сэмпл из {len(df_head):,} строк.")
            except Exception as e:
                st.warning(f"Не удалось подготовить быстрый сэмпл: {e}. Пойду по полному файлу.")

        # >>> переменные окружения для пайплайна <<<
        if fast_mode:
            os.environ["FAST_ROW_LIMIT"] = str(int(row_limit))
            os.environ["DISABLE_EXPLANATIONS"] = "1"
        else:
            os.environ.pop("FAST_ROW_LIMIT", None)
            os.environ.pop("DISABLE_EXPLANATIONS", None)
        os.environ["CHUNK_SIZE"] = "500"
        os.environ["PROGRESS_FILE"] = str(PROGRESS_FILE)

        status.update(label="🚀 Запускаю инференс…")

    # --- Запускаем инференс в отдельном потоке ---
    err_box = st.empty()
    prog_bar = st.progress(0, text="Старт...")
    stage_text = st.empty()

    def _worker():
        try:
            pipeline_infer(TMP_IN, TMP_OUT)
        except Exception as e:
            try:
                with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                    json.dump({"stage": "error", "current": 0, "total": 1, "note": str(e)}, f)
            except Exception:
                pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    # --- Опрос файла прогресса, пока работает поток ---
    while t.is_alive():
        info = _read_progress()
        if info.get("stage") == "error":
            err_box.error(f"Во время инференса произошла ошибка: {info.get('note')}")
            st.stop()

        stage = info.get("stage", "работа")
        cur = int(info.get("current", 0))
        tot = max(int(info.get("total", 1)), 1)
        note = info.get("note", "")

        frac = _calc_total_progress(stage, cur, tot)
        prog_bar.progress(int(frac * 100), text=f"{stage} • {cur}/{tot} {note}")
        stage_text.write(f"Текущая стадия: **{stage}** &nbsp;&nbsp; {cur}/{tot} {note}")
        time.sleep(0.6)

    # финальный апдейт
    info = _read_progress()
    if info.get("stage") == "error":
        err_box.error(f"Во время инференса произошла ошибка: {info.get('note')}")
        st.stop()
    prog_bar.progress(100, text="Готово ✅")

    # Постобработка и выдача результата
    duration = int(time.time() - start_time)
    try:
        df_res = _read_any_csv(TMP_OUT)
    except Exception as e:
        st.error(f"Не удалось прочитать результат `{TMP_OUT}`: {e}")
        st.stop()

    # --- Автосохранение результата на диск Space ---
    ts = time.strftime("%Y%m%d-%H%M%S")
    saved_path = OUTPUTS_DIR / f"predicted-{ts}.csv"
    saved_path.write_bytes(TMP_OUT.read_bytes())

    # --- Отображение и скачивание ---
    st.success(f"Готово за {duration} сек. Ниже первые строки результата:")
    st.dataframe(df_res.head(20), use_container_width=True)

    # Кнопка для текущего результата
    st.download_button(
        "⬇️ Скачать текущий результат",
        data=saved_path.read_bytes(),
        file_name=saved_path.name,
        mime="text/csv",
        type="primary"
    )

    # История сохранений
    with st.expander("🗂 История сохранённых результатов (последние)"):
        files = _list_outputs()
        if not files:
            st.caption("Пока нет сохранённых результатов.")
        else:
            for f in files:
                cols = st.columns([3, 1])
                cols[0].markdown(
                    f"**{f.name}** — {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(f.stat().st_mtime))} · {(f.stat().st_size/1024):.1f} KB"
                )
                cols[1].download_button(
                    "Скачать", data=f.read_bytes(), file_name=f.name, mime="text/csv", key=f"dl_{f.name}"
                )

    with st.expander("ℹ️ Подсказки по ускорению"):
        st.markdown(
            "- Для предварительной проверки используйте **быстрый прогон** — быстрее убедитесь, что всё работает.\n"
            "- Полная обработка больших CSV может занимать время на бесплатном железе.\n"
            "- Если нужно ещё быстрее: заранее нарежьте входной CSV на части и прогоняйте их отдельно."
        )
