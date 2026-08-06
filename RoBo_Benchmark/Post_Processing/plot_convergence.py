"""Erstellt einen Konvergenzgeschwindigkeits-Plot (Mittelwert + Konfidenzband
ueber die Iteration) aus der Ausgabe von compute_confidence_interval.py.

Einfache Filterlogik: ueber --funktion/--dimension/--surrogate-model/
--acquisition-model wird auf eine oder mehrere Kombinationen eingegrenzt.
Bleiben nach dem Filtern mehrere Kombinationen uebrig (z.B. verschiedene
surrogate_model fuer dieselbe Funktion/Dimension), wird pro Kombination eine
eigene Linie + Konfidenzband gezeichnet - so laesst sich die
Konvergenzgeschwindigkeit mehrerer Varianten direkt vergleichen.

    # eine einzelne Kombination (aus der ROBO_Benchmark-Wurzel)
    python Post_Processing/plot_convergence.py --in data/ergebnisse_konfidenzintervall.xlsx --funktion 1 --dimension 2 --surrogate-model GausianProzess --acquisition-model EI --out plots/f1_d2_gp.png

    # mehrere Varianten fuer dieselbe Funktion/Dimension zum Vergleich
    python Post_Processing/plot_convergence.py --in data/ergebnisse_konfidenzintervall.xlsx --funktion 1 --dimension 2 --out plots/f1_d2_vergleich.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # funktioniert ohne Display (z.B. auf der LRZ-Instanz/SSH)
import matplotlib.pyplot as plt
import pandas as pd

GROUP_COLS = ["funktion", "dimension", "surrogate_model", "acquisition_model"]


def load_data(in_path):
    if in_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(in_path)
    return pd.read_csv(in_path)


def filter_data(df, funktion=None, dimension=None, surrogate_model=None, acquisition_model=None):
    """Einfache Filterlogik: jeder angegebene Parameter grenzt df weiter ein,
    nicht angegebene Parameter (None) bleiben ungefiltert."""
    mask = pd.Series(True, index=df.index)
    if funktion is not None:
        mask &= df["funktion"] == funktion
    if dimension is not None:
        mask &= df["dimension"] == dimension
    if surrogate_model is not None:
        mask &= df["surrogate_model"] == surrogate_model
    if acquisition_model is not None:
        mask &= df["acquisition_model"] == acquisition_model
    return df[mask].copy()


def plot_convergence(df, out_path=None, title=None, ylabel="simple_regret"):
    """Zeichnet fuer jede in df verbleibende (funktion, dimension, surrogate_model,
    acquisition_model)-Kombination eine Linie (mean) + Konfidenzband
    (ci_lower/ci_upper) ueber der Iteration."""
    if df.empty:
        raise SystemExit(
            "Keine Zeilen nach dem Filtern uebrig - Filterwerte pruefen "
            "(funktion/dimension/surrogate_model/acquisition_model)."
        )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.get_cmap("tab10").colors

    for i, (key, gdf) in enumerate(df.groupby(GROUP_COLS, sort=True)):
        gdf = gdf.sort_values("iteration")
        funktion, dimension, surrogate_model, acquisition_model = key
        label = f"{surrogate_model} / {acquisition_model} (f{funktion}, d{dimension})"
        color = colors[i % len(colors)]

        ax.plot(gdf["iteration"], gdf["mean"], color=color, label=label, linewidth=1.8)

        # Konfidenzband nur zeichnen, wo Grenzen vorhanden sind - bei n<2
        # Tensorbloecken sind ci_lower/ci_upper NaN (siehe compute_confidence_interval.py)
        has_ci = gdf["ci_lower"].notna() & gdf["ci_upper"].notna()
        if has_ci.any():
            ax.fill_between(
                gdf.loc[has_ci, "iteration"],
                gdf.loc[has_ci, "ci_lower"],
                gdf.loc[has_ci, "ci_upper"],
                color=color, alpha=0.18, linewidth=0,
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"{ylabel} (Mittelwert ± Konfidenzintervall)")
    ax.set_title(title or "Konvergenzgeschwindigkeit")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Plot gespeichert: {out_path}")
    else:
        fig.savefig("convergence_plot.png", dpi=150)
        print("Kein --out angegeben, gespeichert als: convergence_plot.png")

    return fig, ax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True,
                         help="Ausgabe von compute_confidence_interval.py (mean/ci_lower/ci_upper je Zeile)")
    parser.add_argument("--funktion", type=int, default=None)
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument("--surrogate-model", default=None)
    parser.add_argument("--acquisition-model", default=None)
    parser.add_argument("--out", default=None,
                         help="Pfad fuer die PNG-Datei, z.B. plots/f1_d2.png (Default: convergence_plot.png)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    df = load_data(args.in_path)
    df_filtered = filter_data(
        df, funktion=args.funktion, dimension=args.dimension,
        surrogate_model=args.surrogate_model, acquisition_model=args.acquisition_model,
    )
    n_groups = df_filtered.groupby(GROUP_COLS).ngroups if not df_filtered.empty else 0
    print(f"{len(df_filtered)} von {len(df)} Zeilen nach Filter, {n_groups} Kombination(en)")

    plot_convergence(df_filtered, out_path=args.out, title=args.title)


if __name__ == "__main__":
    main()
