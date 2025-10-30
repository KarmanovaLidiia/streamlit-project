# tools/merge_truth_and_eval.py
from __future__ import annotations
import csv
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score

PRED_PATH = Path("predicted_big_final.csv")
TRUTH_PATH = Path("data/tmp_input.csv")
OUT_PATH = Path("predicted_with_truth.csv")
DIAG_DIR = Path("reports/diag_merge")
DIAG_DIR.mkdir(parents=True, exist_ok=True)

def read_csv_smart(path: Path) -> pd.DataFrame:
    """
    Надёжное чтение CSV:
    1) sep=';' + engine='python'
    2) sep=None (автоопределение)
    3) если всё равно одна колонка — читаем шапку вручную и переоткрываем с именами
    """
    tried = []

    def _try(**kw):
        tried.append(kw)
        return pd.read_csv(path, **kw)

    try:
        df = _try(sep=";", encoding="utf-8-sig", engine="python")
        if len(df.columns) > 1:
            return df
    except Exception:
        pass

    try:
        df = _try(sep=None, encoding="utf-8-sig", engine="python")
        if len(df.columns) > 1:
            return df
    except Exception:
        pass

    # Вручную парсим заголовок (первую строку) и перечитываем с явными именами
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline().rstrip("\n\r")
    # Пробуем разделить по ';'
    header = [h.strip() for h in first_line.split(";")]
    # перечитываем, пропуская первую строку, с этими именами
    try:
        df = _try(sep=";", encoding="utf-8-sig", engine="python",
                  names=header, header=None, skiprows=1, quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
        return df
    except Exception:
        # последний шанс: csv.reader
        rows = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rdr = csv.reader(f, delimiter=";", quotechar='"')
            for row in rdr:
                rows.append(row)
        df = pd.DataFrame(rows[1:], columns=rows[0] if rows else None)
        return df

def normalize_num(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    s = s.str.replace(r"[^\d\-]+", "", regex=True)
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def normalize_txt(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\ufeff", "", regex=False).str.replace('"', "", regex=False).str.strip()

def load_pred(path: Path) -> pd.DataFrame:
    df = read_csv_smart(path)
    need = ["Id экзамена", "Id вопроса", "№ вопроса", "pred_score"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"[pred] Нет колонок {miss}. Найдены: {list(df.columns)}")

    df["Id экзамена"] = normalize_num(df["Id экзамена"])
    df["Id вопроса"]  = normalize_num(df["Id вопроса"])
    df["№ вопроса"]   = normalize_num(df["№ вопроса"])
    df["pred_score"]  = pd.to_numeric(df["pred_score"], errors="coerce")

    for c in ["Текст вопроса", "Транскрибация ответа", "Картинка из вопроса", "Ссылка на оригинальный файл запис"]:
        if c in df.columns:
            df[c] = normalize_txt(df[c])

    if "Оценка экзаменатора" in df.columns:
        df["Оценка экзаменатора"] = pd.to_numeric(df["Оценка экзаменатора"], errors="coerce")

    return df

def load_truth(path: Path) -> pd.DataFrame:
    df = read_csv_smart(path)

    # пытаемся найти колонку с истиной
    truth_candidates = ["Оценка экзаменатора", "Оценка_экзаменатора", "Оценка", "label", "score"]
    truth_col = next((c for c in truth_candidates if c in df.columns), None)
    if truth_col is None:
        raise RuntimeError(f"[truth] Не найдена колонка истины среди {truth_candidates}. Найдены: {list(df.columns)}")

    for c in ["Id экзамена", "Id вопроса", "№ вопроса"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["Id экзамена"] = normalize_num(df["Id экзамена"])
    df["Id вопроса"]  = normalize_num(df["Id вопроса"])
    df["№ вопроса"]   = normalize_num(df["№ вопроса"])
    df["true_score"]  = pd.to_numeric(df[truth_col], errors="coerce")

    keep = ["Id экзамена", "Id вопроса", "№ вопроса", truth_col, "true_score"]
    return df[[c for c in keep if c in df.columns]]

def try_merge(pred: pd.DataFrame, truth: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    l = pred.dropna(subset=[k for k in keys if k in pred.columns])
    r = truth.dropna(subset=[k for k in keys if k in truth.columns])
    return l.merge(r, on=keys, how="inner", suffixes=("", "_truth"))

def dump_diag(pred, truth, keys, merged, tag):
    # ключи, отсутствующие у партнёра
    merged_keys = merged[keys].drop_duplicates()
    pred_keys = pred[keys].drop_duplicates()
    truth_keys = truth[keys].drop_duplicates()

    not_in_truth = pred_keys.merge(merged_keys, on=keys, how="left", indicator=True)
    not_in_truth = not_in_truth[not_in_truth["_merge"] == "left_only"].drop(columns="_merge")
    if not not_in_truth.empty:
        not_in_truth.to_csv(DIAG_DIR / f"not_in_truth__{tag}.csv", sep=";", index=False)

    not_in_pred = truth_keys.merge(merged_keys, on=keys, how="left", indicator=True)
    not_in_pred = not_in_pred[not_in_pred["_merge"] == "left_only"].drop(columns="_merge")
    if not not_in_pred.empty:
        not_in_pred.to_csv(DIAG_DIR / f"not_in_pred__{tag}.csv", sep=";", index=False)

def main():
    pred = load_pred(PRED_PATH)
    truth = load_truth(TRUTH_PATH)

    print(f"[pred] rows={len(pred)}  cols={list(pred.columns)}")
    print(f"[truth] rows={len(truth)} cols={list(truth.columns)}")

    strategies = [
        (["Id вопроса"],                            "question_only"),
        (["Id экзамена", "Id вопроса", "№ вопроса"], "exam+question+qno"),
        (["Id экзамена", "№ вопроса"],              "exam+qno"),
    ]

    merged = pd.DataFrame()
    used = None
    for keys, tag in strategies:
        if not set(keys).issubset(pred.columns) or not set(keys).issubset(truth.columns):
            continue
        m = try_merge(pred, truth, keys)
        print(f"→ try keys={keys}: matched={len(m)}")
        if len(m) > 0:
            merged = m
            used = (keys, tag)
            dump_diag(pred, truth, keys, merged, tag)
            break

    if merged.empty:
        # быстрые пересечения по ключам — для понимания что сломано
        for keys, tag in strategies:
            inter = []
            for k in keys:
                inter.append((k,
                              len(set(pred[k].dropna()) & set(truth[k].dropna()))))
            print(f"[DIAG] {keys} intersections: {inter}")
        print("❌ Совпадений нет. Смотри reports/diag_merge/*.csv и проверь формат ключей.")
        return

    out = merged.copy()
    if "Оценка экзаменатора" in out.columns:
        out["Оценка экзаменатора"] = out["Оценка экзаменатора"].fillna(out["true_score"])

    valid = out.dropna(subset=["true_score", "pred_score"])
    print(f"valid pairs: {len(valid)} of {len(out)}")
    if len(valid) > 0:
        mae = mean_absolute_error(valid["true_score"], valid["pred_score"])
        acc = accuracy_score(
            valid["true_score"].round().clip(0, 2),
            valid["pred_score"].round().clip(0, 2)
        )
        print(f"MAE={mae:.4f} | ACC(round)={acc:.4f}")

    cols = ["Id экзамена","Id вопроса","№ вопроса",
            "Текст вопроса","Картинка из вопроса",
            "Оценка экзаменатора","Транскрибация ответа",
            "Ссылка на оригинальный файл запис","pred_score","true_score"]
    cols = [c for c in cols if c in out.columns]
    out[cols].to_csv(OUT_PATH, sep=";", index=False, encoding="utf-8-sig")
    print(f"OK -> {OUT_PATH} | rows={len(out)} | cols={len(cols)}")
    if used:
        print(f"Used keys: {used[0]} ({used[1]})")
    print("Диагностика — reports/diag_merge")

if __name__ == "__main__":
    main()
