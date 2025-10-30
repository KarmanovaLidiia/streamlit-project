# tools/split_holdout.py
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd

RAW = pathlib.Path("data/raw/Данные для кейса.csv")
TRAIN_OUT = pathlib.Path("data/raw/train_split.csv")
HOLD_OUT = pathlib.Path("data/raw/holdout_split.csv")

def main():
    df = pd.read_csv(RAW, encoding="utf-8-sig", sep=";")
    rng = np.random.default_rng(42)
    msk = rng.random(len(df)) < 0.8  # 80% train, 20% holdout
    df_train = df[msk].copy()
    df_hold  = df[~msk].copy()
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig", sep=";")
    df_hold.to_csv(HOLD_OUT, index=False, encoding="utf-8-sig", sep=";")
    print(f"[done] train={df_train.shape}, holdout={df_hold.shape}")
    print(f"train -> {TRAIN_OUT}")
    print(f"holdout -> {HOLD_OUT}")

if __name__ == "__main__":
    main()
