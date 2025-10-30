# tools/check_predicted.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def read_csv_safely(path: Path) -> pd.DataFrame:
    # сначала пробуем авто-детект, потом явные варианты
    tries = [
        ("utf-8-sig", None),
        ("utf-8", None),
        ("utf-8-sig", ";"),
        ("utf-8-sig", ","),
        ("utf-8", ";"),
        ("utf-8", ","),
    ]
    last_err = None
    for enc, sep in tries:
        try:
            if sep is None:
                df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
                print(f"[i] CSV прочитан с encoding='{enc}', sep='auto'")
            else:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                print(f"[i] CSV прочитан с encoding='{enc}', sep='{sep}'")
            # если «одна колонка» и в ней видны разделители — попробуем противоположный
            if df.shape[1] == 1:
                head = str(df.columns[0])
                if ("," in head) or (";" in head):
                    alt = "," if sep == ";" else ";"  # переключаем
                    print(f"[i] Похоже, файл не тем разделителем. Пробую sep='{alt}'...")
                    df = pd.read_csv(path, encoding=enc, sep=alt)
                    print(f"[i] Повторно прочитано: sep='{alt}', колонки={len(df.columns)}")
            return df
        except Exception as e:
            last_err = e
    raise last_err


def resolve_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    norm = {c: "".join(str(c).lower().split()) for c in df.columns}
    for cand in candidates:
        ckey = "".join(cand.lower().split())
        for real, rkey in norm.items():
            if rkey == ckey:
                return real
    raise KeyError(f"Не удалось найти колонку из: {candidates}\nЕсть: {list(df.columns)}")


def mae(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(float)
    p = p.astype(float)
    return float(np.mean(np.abs(y - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default="data/processed/predicted.csv")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"❌ Не найден файл: {path}")

    df = read_csv_safely(path)
    print("Колонки:", list(df.columns))
    print("Размер:", df.shape)

    pred_col = resolve_col(df, ["pred_score", "pred"])
    y_col    = resolve_col(df, ["Оценка экзаменатора", "оценка", "score", "Score"])
    q_col    = resolve_col(df, ["№ вопроса", "№  вопроса", "номер вопроса", "question_number", "q"])

    y = pd.to_numeric(df[y_col], errors="coerce")
    p = pd.to_numeric(df[pred_col], errors="coerce")
    if y.isna().any():
        print(f"⚠️ Внимание: {y.isna().sum()} NaN в '{y_col}' — отброшу при MAE.")
    m = ~(y.isna() | p.isna())
    df = df.loc[m].copy()
    y = y.loc[m].to_numpy()
    p = p.loc[m].to_numpy()

    print("\nДиапазоны предсказаний по вопросам:")
    for q,(lo,hi) in {1:(0,1), 2:(0,2), 3:(0,1), 4:(0,2)}.items():
        mq = df[q_col] == q
        if mq.any():
            pq = pd.to_numeric(df.loc[mq, pred_col], errors="coerce")
            print(f"  Q{q}: min={pq.min():.3f}  max={pq.max():.3f}  ожидается [{lo},{hi}]  (n={mq.sum()})")
        else:
            print(f"  Q{q}: нет строк.")

    print(f"\nMAE(вся выборка): {mae(y, p):.3f}")

    print("\nMAE по вопросам:")
    for q in [1,2,3,4]:
        mq = df[q_col] == q
        if mq.any():
            mae_q = mae(pd.to_numeric(df.loc[mq, y_col]).to_numpy(),
                        pd.to_numeric(df.loc[mq, pred_col]).to_numpy())
            print(f"  Q{q}: {mae_q:.3f}")

    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    print(f"\nДиапазон истинных меток '{y_col}': [{y_min:.3f}, {y_max:.3f}]")
    uniq = np.unique(y)
    if len(uniq) <= 6:
        print(f"Уникальные значения меток: {uniq}")


if __name__ == "__main__":
    main()
