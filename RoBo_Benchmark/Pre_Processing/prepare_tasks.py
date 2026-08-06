"""Erzeugt EINMALIG zentral die vollstaendige Task-Liste + Sample-Tensoren.

Ausfuehren, bevor die Tasks auf die Instanzen aufgeteilt werden (split_tasks.py).

    python Pre_Processing/prepare_tasks.py   (aus der ROBO_Benchmark-Wurzel)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401 - fuegt Main_Prozess/, BO_Verfahren/, Post_Processing/ zu sys.path hinzu

import itertools
import json

import cocoex
import numpy as np
import pandas as pd

import config
import core_bo


def build_kombinationen():
    rows = []
    combination_id = 0
    for funktion, dimension, (surrogate, akquisition), n_init, n_iter in itertools.product(
        config.FUNKTIONEN,
        config.DIMENSIONEN,
        config.SURROGATE_AKQUISITION_PAARE,
        config.N_SAMPLE_INIT,
        config.N_ITERATION,
    ):
        rows.append({
            "combination_id": combination_id,
            "funktion": funktion,
            "dimension": dimension,
            "surrogate_model": surrogate,
            "acquisition_model": akquisition,
            "sample_size_initial": n_init,
            "max_iteration": n_iter,
        })
        combination_id += 1
    return pd.DataFrame(rows)


def add_optimum_estimates(df_gesamt):
    """Schaetzt je (Funktion, Dimension) das Maximum von f(x) im Suchraum
    EINMAL und traegt es allen betroffenen Kombinationen ein, statt es
    (teuer) pro Task neu zu berechnen."""
    suite = cocoex.Suite("bbob", "", "")

    lookup = {}
    for funktion, dimension in df_gesamt[["funktion", "dimension"]].drop_duplicates().itertuples(index=False):
        x_opt, f_opt = core_bo.estimate_bbob_maximum(
            funktion, dimension, suite,
            config.MINUS_AREA, config.PLUS_AREA,
            n_samples=config.N_OPTIMUM_ESTIMATE_SAMPLES,
            seed=config.OPTIMUM_ESTIMATE_SEED,
        )
        lookup[(funktion, dimension)] = (json.dumps(x_opt.tolist()), f_opt)
        print(f"  Funktion {funktion}, Dimension {dimension}: f_opt_approx = {f_opt:.4f}")

    df_gesamt["x_opt_approx"] = df_gesamt.apply(
        lambda r: lookup[(r["funktion"], r["dimension"])][0], axis=1
    )
    df_gesamt["f_opt_approx"] = df_gesamt.apply(
        lambda r: lookup[(r["funktion"], r["dimension"])][1], axis=1
    )
    return df_gesamt


def build_tensor_noise(df_gesamt):
    max_dim = int(df_gesamt["dimension"].max())
    max_samples = int(df_gesamt["sample_size_initial"].max())

    tensor = np.empty((config.N_SAMPLE_STAT, max_samples, max_dim), dtype=np.float32)
    for i in range(config.N_SAMPLE_STAT):
        rng = np.random.default_rng(config.SEEDS[i])
        tensor[i] = rng.uniform(
            low=config.MINUS_AREA, high=config.PLUS_AREA,
            size=(max_samples, max_dim),
        ).astype(np.float32)

    # Rauschen hinzufuegen, analog add_noise_stratified() in ROBO/Subprocess.py
    rng_auswahl = np.random.default_rng(config.NOISE_SEED_AUSWAHL)
    rng_rauschen = np.random.default_rng(config.NOISE_SEED_RAUSCHEN)
    ergebnis = tensor.copy()
    for i in range(tensor.shape[0]):
        schicht = ergebnis[i]
        n_total = schicht.size
        n_auswahl = int(round(n_total * config.NOISE_ANTEIL))
        flat_indices = rng_auswahl.choice(n_total, size=n_auswahl, replace=False)
        idx = np.unravel_index(flat_indices, schicht.shape)
        rauschen = rng_rauschen.normal(loc=0.0, scale=config.NOISE_SIGMA, size=n_auswahl)
        schicht[idx] = schicht[idx] + rauschen

    return ergebnis


def build_tasks(df_gesamt):
    rows = []
    task_id = 0
    for combo in df_gesamt.itertuples():
        for tensor_block in range(config.N_SAMPLE_STAT):
            rows.append({
                "task_id": task_id,
                "combination_id": combo.combination_id,
                "tensor_block": tensor_block,
                "funktion": combo.funktion,
                "dimension": combo.dimension,
                "surrogate_model": combo.surrogate_model,
                "acquisition_model": combo.acquisition_model,
                "sample_size_initial": combo.sample_size_initial,
                "max_iteration": combo.max_iteration,
                "x_opt_approx": combo.x_opt_approx,
                "f_opt_approx": combo.f_opt_approx,
            })
            task_id += 1
    return pd.DataFrame(rows)


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_gesamt = build_kombinationen()
    print("Schaetze Maximum je (Funktion, Dimension)...")
    df_gesamt = add_optimum_estimates(df_gesamt)
    tensor_noise = build_tensor_noise(df_gesamt)
    df_tasks = build_tasks(df_gesamt)

    df_gesamt.to_csv(config.DATA_DIR / "kombinationen.csv", index=False)
    df_tasks.to_csv(config.DATA_DIR / "tasks.csv", index=False)
    np.save(config.DATA_DIR / "tensor_noise.npy", tensor_noise)

    print(f"Kombinationen: {len(df_gesamt)}")
    print(f"Tasks gesamt (Kombinationen x {config.N_SAMPLE_STAT} Tensorbloecke): {len(df_tasks)}")
    print(f"Tensor-Shape: {tensor_noise.shape}")
    print(f"Geschrieben nach: {config.DATA_DIR}")


if __name__ == "__main__":
    main()
