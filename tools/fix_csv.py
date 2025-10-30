# tools/fix_csv.py
from pathlib import Path
import pandas as pd
import csv

inp = Path("predicted_big.csv")
out = Path("predicted_big_clean.csv")

def parse_mixed_header(header_raw: str) -> list[str]:
    """
    Первая строка в predicted_big.csv у тебя такая:
    'Id экзамена,Id вопроса,...,pred_score;Оценка экзаменатора;pred_score'
    Т.е. слева имена через запятую, потом через ';'.
    Восстанавливаем единый список имён столбцов.
    """
    header_raw = header_raw.strip().lstrip("\ufeff")  # убрать BOM на всякий
    if ";" in header_raw:
        left, right = header_raw.split(";", 1)
    else:
        left, right = header_raw, ""

    cols = [c.strip() for c in left.split(",") if c.strip()]
    if right:
        cols.extend([c.strip() for c in right.split(";") if c.strip()])

    # Уберём дубликаты имён (оставим последнее вхождение)
    seen = set()
    dedup = []
    for c in cols:
        if c in seen:
            continue
        seen.add(c)
        dedup.append(c)
    return dedup

def main():
    # 1) читаем первую строку (шапку) «как есть»
    with inp.open("r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()
    if not lines:
        raise RuntimeError("Файл пустой")

    header_raw = lines[0]
    cols = parse_mixed_header(header_raw)

    # 2) читаем остальные строки как нормальный ;-CSV без шапки
    #    и принудительно задаём имена колонок
    from io import StringIO
    body = "".join(lines[1:])

    df = pd.read_csv(
        StringIO(body),
        delimiter=";",           # именно ; — как в нормальном CSV
        header=None,             # шапку мы уже восстановили вручную
        names=cols,              # задаём имена колонок
        engine="python",
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE,  # кавычки не считаем спецсимволами
        escapechar="\\",
        on_bad_lines="skip",
    )

    # 3) финальная косметика
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    # 4) сохраняем обратно в нормальном виде
    df.to_csv(out, sep=";", index=False, encoding="utf-8-sig")
    print(f"OK -> {out} | rows: {len(df)} | cols: {len(df.columns)}")
    print("columns:", "; ".join(df.columns))

if __name__ == "__main__":
    main()
