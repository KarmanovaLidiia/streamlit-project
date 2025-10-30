# src/analyze_results.py
from __future__ import annotations
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _read_csv_safely(path: Path) -> pd.DataFrame:
    """
    Устойчивое чтение CSV: пробуем ; , и авто. Выбираем тот, где получилось >= 5 колонок.
    """
    tries = [
        ("utf-8-sig", ","),   # СНАЧАЛА КОММА — чаще для наших predicted.csv
        ("utf-8-sig", ";"),
        ("utf-8", ","),
        ("utf-8", ";"),
        ("utf-8-sig", None),
        ("utf-8", None),
    ]
    last_err = None
    best_df = None
    best_info = None

    for enc, sep in tries:
        try:
            if sep is None:
                df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
                got = f"auto"
            else:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                got = sep
            # эвристика: нормальные файлы имеют >= 5 колонок
            if df.shape[1] >= 5:
                print(f"[i] CSV прочитан с encoding='{enc}', sep='{got}'")
                return df
            # запомним самый «лучший» (по числу колонок), если ни один не пройдёт порог
            if best_df is None or df.shape[1] > best_df.shape[1]:
                best_df, best_info = df, (enc, got)
        except Exception as e:
            last_err = e

    if best_df is not None:
        enc, got = best_info
        print(f"[!] Не удалось надёжно определить разделитель, взят лучший вариант encoding='{enc}', sep='{got}' (cols={best_df.shape[1]})")
        return best_df

    raise last_err if last_err else RuntimeError("Не удалось прочитать CSV")

def _resolve_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Не удалось найти колонку из вариантов: {candidates}\nИмеющиеся колонки: {list(df.columns)}")

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/processed/predicted.csv", help="Путь к CSV с колонкой pred_score")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"❌ Не найден файл с предсказаниями: {in_path}")
        sys.exit(1)

    df = _read_csv_safely(in_path)

    # Колонки
    y_col = _resolve_col(df, ["Оценка экзаменатора", "оценка экзаменатора", "score", "y", "target"])
    p_col = _resolve_col(df, ["pred_score", "pred", "prediction"])

    # Колонка номера вопроса: поддержим оба варианта
    if "№ вопроса" in df.columns:
        q_col = "№ вопроса"
    else:
        q_col = _resolve_col(df, ["question_number", "q", "номер вопроса", "№  вопроса"])

    # Числовые массивы
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    p = pd.to_numeric(df[p_col], errors="coerce").to_numpy()

    # Общая MAE
    m_all = mae(y, p)
    print(f"MAE (вся выборка): {m_all:.3f}\n")

    # MAE по вопросам
    g = (df[[q_col]].copy())
    g["y"] = y
    g["p"] = p
    mae_by_q = (
        g.groupby(q_col, as_index=True)
         .apply(lambda s: mae(s["y"].to_numpy(), s["p"].to_numpy()))
         .to_frame("MAE")
    )
    print("MAE по вопросам:")
    print(mae_by_q)
    out_dir = Path("reports"); out_dir.mkdir(parents=True, exist_ok=True)
    mae_by_q.to_csv(out_dir / "metrics_summary.csv", encoding="utf-8-sig")
    print(f"\n✅ Сохранено: {out_dir / 'metrics_summary.csv'}")

    # Графики
    # 1) гистограмма ошибок
    err = np.abs(y - p)
    plt.figure()
    plt.hist(err[~np.isnan(err)], bins=30)
    plt.title("Absolute Error Histogram")
    plt.xlabel("|y - pred|"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(out_dir / "error_hist.png"); plt.close()
    print(f"📊 Гистограмма: {out_dir / 'error_hist.png'}")

    # 2) mae_by_q барплот
    plt.figure()
    mae_by_q["MAE"].plot(kind="bar")
    plt.ylabel("MAE"); plt.title("MAE by question")
    plt.tight_layout(); plt.savefig(out_dir / "mae_by_q.png"); plt.close()
    print(f"📊 MAE по вопросам: {out_dir / 'mae_by_q.png'}")

    # 3) scatter
    plt.figure()
    plt.scatter(y, p, alpha=0.3)
    plt.xlabel("true"); plt.ylabel("pred"); plt.title("Pred vs True")
    plt.tight_layout(); plt.savefig(out_dir / "pred_vs_true.png"); plt.close()
    print(f"📊 Scatter: {out_dir / 'pred_vs_true.png'}")

if __name__ == "__main__":
    main()
