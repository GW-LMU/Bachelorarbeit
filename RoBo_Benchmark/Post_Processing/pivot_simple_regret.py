"""Erstellt ergebnisse_wide.xlsx: simple_regret im Wide-Format.

Die normalen results.csv-Dateien sind im Long-Format (eine Zeile pro
BO-Iteration UND Tensorblock). Dieses Skript pivotiert sie so, dass jede
Zeile genau einer (Kombination, Iteration) entspricht - die Tensorbloecke
stehen dann nebeneinander in den Spalten Var_1..Var_n (statt untereinander
in eigenen Zeilen), analog zu den Var_n-Spalten in ROBO/Subprocess.py.

    python Post_Processing/pivot_simple_regret.py --pattern "data/instances/instance_*/results/results.csv" --out data/ergebnisse_wide.xlsx
    (aus der ROBO_Benchmark-Wurzel)
"""

import argparse
import glob

import pandas as pd

ID_COLS = [
    "combination_id", "funktion", "dimension", "surrogate_model",
    "acquisition_model", "sample_size_initial", "max_iteration", "iteration",
]


def main(pattern, out_path):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"Keine Dateien gefunden fuer Pattern: {pattern}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    n_error = (df["status"] == "error").sum()
    df = df[df["status"] == "ok"].copy()

    df_wide = df.pivot_table(
        index=ID_COLS,
        columns="tensor_block",
        values="simple_regret",
    )
    df_wide.columns = [f"Var_{int(c) + 1}" for c in df_wide.columns]
    df_wide = df_wide.reset_index().sort_values(["combination_id", "iteration"])

    if out_path.endswith(".xlsx"):
        df_wide.to_excel(out_path, index=False)
    else:
        df_wide.to_csv(out_path, index=False)

    print(f"Zeilen (Kombination x Iteration): {len(df_wide)} | Var-Spalten: {len(df_wide.columns) - len(ID_COLS)}")
    print(f"Ausgelassene Fehler-Tasks: {n_error}")
    print(f"Geschrieben nach: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--out", default="data/ergebnisse_wide.xlsx")
    args = parser.parse_args()
    main(args.pattern, args.out)
