import argparse
import pandas as pd
import numpy as np
import sys

def safe_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred-col", default="predicted_score")
    ap.add_argument("--score-col", default="examiner_score")
    ap.add_argument("--question-col", default="question_number")
    ap.add_argument("--key", default="")
    args = ap.parse_args()

    p = pd.read_csv(args.pred)
    g = pd.read_csv(args.gold)

    if args.pred_col not in p.columns:
        print(f"ERROR: нет {args.pred_col} в {args.pred}"); sys.exit(1)
    if args.score_col not in g.columns:
        print(f"ERROR: нет {args.score_col} в {args.gold}"); sys.exit(1)

    keys = [k.strip() for k in args.key.split(",") if k.strip()]
    if keys:
        for miss in [k for k in keys if k not in p.columns]:
            print(f"ERROR: нет ключа {miss} в pred"); sys.exit(1)
        for miss in [k for k in keys if k not in g.columns]:
            print(f"ERROR: нет ключа {miss} в gold"); sys.exit(1)
        merged = p[keys + [args.pred_col]].merge(
            g[keys + [args.score_col]], on=keys, how="inner", validate="one_to_one"
        )
    else:
        if len(p) != len(g):
            print("ERROR: разные размеры pred/gold и нет ключа --key"); sys.exit(1)
        merged = pd.DataFrame({
            args.pred_col: p[args.pred_col].values,
            args.score_col: g[args.score_col].values
        })

    y_pred = merged[args.pred_col].map(safe_float)
    y_true = merged[args.score_col].map(safe_float)
    mask = (~y_pred.isna()) & (~y_true.isna())
    mae = np.mean(np.abs(y_pred[mask] - y_true[mask]))
    print(f"MAE (общий): {mae:.4f} | N={mask.sum()}")

    # по вопросам, если есть
    try:
        qp = p.loc[mask, args.question_col] if args.question_col in p.columns else g.loc[mask, args.question_col]
        df = pd.DataFrame({"qn": qp.values, "pred": y_pred[mask].values, "true": y_true[mask].values})
        for q, v in df.groupby("qn").apply(lambda d: np.mean(np.abs(d["pred"] - d["true"]))).sort_index().items():
            print(f"  Q{int(q)} MAE: {v:.4f}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
