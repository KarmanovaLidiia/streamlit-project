import os
import json
import time
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

uploaded = st.file_uploader("Загрузите CSV", type=["csv"])
run = st.button("Оценить", type="primary", disabled=uploaded is None)

# --------------------------- RUNTIME PATHS ---------------------------
TMP_DIR_IN = Path("data/raw")
TMP_DIR_OUT = Path("data/processed")
TMP_DIR_IN.mkdir(parents=True, exist_ok=True)
TMP_DIR_OUT.mkdir(parents=True, exist_ok=True)

TMP_IN = TMP_DIR_IN / "tmp_input.csv"
TMP_OUT = TMP_DIR_OUT / "tmp_output.csv"

# Файл прогресса, его будет писать src/predict.py
PROGRESS_FILE = Path("/tmp/progress.json")
os.environ["PROGRESS_FILE"] = str(PROGRESS_FILE)

# Веса стадий для общего прогресса (сумма ≈ 100)
STAGE_WEIGHTS = {
    "загрузка CSV": 2,
    "очистка": 6,
    "базовые фичи": 6,
    "семантика": 55,        # самый тяжелый шаг
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
    """Переводим локальный прогресс стадии в общий % с учетом весов."""
    # сумма весов предыдущих стадий
    stages = list(STAGE_WEIGHTS.keys())
    done_weight = 0.0
    for s in stages:
        if s == cur_stage:
            break
        done_weight += STAGE_WEIGHTS.get(s, 0)
    stage_weight = STAGE_WEIGHTS.get(cur_stage, 0)
    frac = 0.0
    if total and total > 0:
        frac = min(max(current / float(total), 0.0), 1.0)
    return (done_weight + stage_weight * frac) / sum(STAGE_WEIGHTS.values())

def _read_progress():
    """Безопасное чтение прогресса из файла."""
    try:
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"stage": "подготовка", "current": 0, "total": 1, "note": ""}

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
        st.write("Сохраняю загруженный файл…")
        raw_bytes = uploaded.read()
        _save_bytes_to(TMP_IN, raw_bytes)

        # Быстрый прогон — перезапишем TMP_IN первыми N строками
        if fast_mode:
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
            # складываем текст исключения в файл прогресса, чтобы основной поток отобразил
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump({"stage": "error", "current": 0, "total": 1, "note": str(e)}, f)

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
        time.sleep(0.8)

    # финальный апдейт (на случай, если поток завершился между итерациями)
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

    st.success(f"Готово за {duration} сек. Ниже первые строки результата:")
    st.dataframe(df_res.head(20), use_container_width=True)

    st.download_button(
        "⬇️ Скачать результат",
        data=TMP_OUT.read_bytes(),
        file_name="predicted.csv",
        mime="text/csv",
        type="primary"
    )

    with st.expander("ℹ️ Подсказки по ускорению"):
        st.markdown(
            "- Для предварительной проверки используйте **быстрый прогон** — быстрее убедитесь, что всё работает.\n"
            "- Полная обработка больших CSV может занимать время на бесплатном железе.\n"
            "- Если нужно ещё быстрее: заранее нарежьте входной CSV на части и прогоняйте их отдельно."
        )
