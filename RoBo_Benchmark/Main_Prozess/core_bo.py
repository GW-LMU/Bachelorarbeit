"""BO-Kernfunktionen fuer die verteilte Pipeline.

Eigenstaendige Kopie/Anpassung der Logik aus ROBO/coco_benchmark.py und
ROBO/Standard_PY.py (dort nicht veraendert), damit ROBO_NEU ohne Abhaengigkeit
zum alten Ordner auf jede LRZ-Instanz kopiert werden kann. Die Standard-GP-
Variante liegt in gaussian_process.py, die weiteren robusten Varianten
(Imprecise GP, Relevance Pursuit, STABLEOPT, DRBO) sind analog angepasste
Kopien aus ROBO/, AIRBO ist eine schlanke Neuimplementierung nach Yang et al.
2023 (ROBO/AIRBO.py selbst war leer, siehe airbo.py fuer Details/Annahmen).
Alle sechs werden unten in select_and_run_variant() angebunden.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401 - fuegt Pre_Processing/ (config) und BO_Verfahren/ zu sys.path hinzu

import json

import numpy as np
import torch

import config


def evaluate_bbob_function(x, funktion, dimension, suite, instance=1):# ✅
    """Negierter BBOB-Funktionswert: -f(x). BBOB-Probleme sind als
    Minimierung von f(x) definiert, mit dem bekannten (analytischen)
    Optimum im Inneren des Suchraums. Da hier durchgaengig MAXIMIERT wird
    (BoTorch-Konvention), wird die Standard-BBOB-Konvention hergestellt,
    indem -f(x) statt des rohen f(x) zurueckgegeben wird - das Maximum von
    -f(x) entspricht dann exakt dem Minimum von f(x)."""
    problem = suite.get_problem_by_function_dimension_instance(
        function=funktion, dimension=dimension, instance=instance
    )
    x_numpy = np.asarray(x, dtype=float).reshape(-1)
    y_val = problem(x_numpy)
    return -float(y_val)


def compute_regret(X_history, Y_history, x_opt, f_opt):# ✅
    y_values = np.asarray(Y_history, dtype=float)
    x_opt = np.asarray(x_opt, dtype=float)

    x_distances = np.array([np.linalg.norm(np.asarray(x, dtype=float).reshape(-1) - x_opt) for x in X_history])
    best_so_far = np.maximum.accumulate(y_values)
    simple_regret = f_opt - best_so_far
    immediate_regret = f_opt - y_values

    return best_so_far, simple_regret, immediate_regret, x_distances


def select_and_run_variant(surrogate_model, acquisition_func, train_X, train_Y, dim,
                            lower_bound, upper_bound):# ✅
    """Analog zu waehle_und_berechne_variante() in ROBO/coco_benchmark.py.

    Alle Varianten haben dieselbe Signatur (train_X, train_Y, dim, acquisition_func,
    lower_bound, upper_bound) und geben (next_x, acq_value) zurueck, siehe die
    jeweiligen run_*()-Funktionen in gaussian_process.py / imprecise_gp.py /
    relevance_pursuit.py / stable_opt.py / drbo.py / airbo.py fuer Details und
    modellspezifische Einschraenkungen (Imprecise GP: nur dim=1; STABLEOPT: nur
    acquisition_func="UCB"; DRBO: nur acquisition_func="distributionally
    robust UCB-Akquisition", nutzt die letzte Spalte von dim als
    Kontextvariable; AIRBO: nur acquisition_func="UCB", nimmt eine simulierte
    Gauss-Eingabeunsicherheit um jeden Punkt an, siehe airbo.py).
    """
    if surrogate_model == "GausianProzess":
        from gaussian_process import run_gaussian_process
        return run_gaussian_process(train_X, train_Y, dim, acquisition_func, lower_bound, upper_bound)
    elif surrogate_model == "Imprecise_GausianProzess_Rodemann":
        from imprecise_gp import run_imprecise_gausian_prozess
        return run_imprecise_gausian_prozess(train_X, train_Y, dim, acquisition_func, lower_bound, upper_bound)
    elif surrogate_model == "AIRBO":
        from airbo import run_airbo
        return run_airbo(train_X, train_Y, dim, acquisition_func, lower_bound, upper_bound)
    elif surrogate_model == "STABLEOPT":
        from stable_opt import run_stable_opt
        return run_stable_opt(train_X, train_Y, dim, acquisition_func, lower_bound, upper_bound)
    elif surrogate_model == "DRBO":
        from drbo import run_drbo
        return run_drbo(train_X, train_Y, dim, acquisition_func, lower_bound, upper_bound)
    elif surrogate_model == "Relevance Pursuit":
        # eigener Import-Block, siehe Hinweis oben: schlaegt mit aelterem botorch
        # gezielt erst HIER fehl, statt schon beim Modul-Import von core_bo.py
        try:
            from relevance_pursuit import run_relevance_pursuit
        except ImportError as e:
            raise ImportError(
                "Variante 'Relevance Pursuit' braucht RobustRelevancePursuitSingleTaskGP "
                "(botorch.models.robust_relevance_pursuit_model bzw. bei aelteren Versionen "
                "botorch.models.relevance_pursuit_model) - das gibt es erst ab botorch >= 0.11, "
                "was wiederum Python >= 3.10 voraussetzt. Mit der hier installierten botorch-/"
                "Python-Version nicht verfuegbar - siehe RoBo_Benchmark/.venv311 (Python 3.11 + "
                "aktuelles botorch) fuer eine Umgebung, in der diese Variante laeuft."
            ) from e
        return run_relevance_pursuit(train_X, train_Y, dim, acquisition_func, lower_bound, upper_bound)
    else:
        raise ValueError(f"Unbekanntes Surrogatmodell: {surrogate_model}")


def _make_y_noise_fn(task):# ✅
    """Baut eine Funktion, die -f(x)-Werte optional mit Rauschen versieht.

    Nur aktiv, wenn config.NOISE_Y_ENABLED True ist (siehe die drei Faelle
    NOISE_X_ENABLED/NOISE_Y_ENABLED in config.py). Der RNG wird deterministisch
    aus (combination_id, tensor_block) geseedet, damit derselbe Task bei einem
    Rerun (z.B. nach Abbruch) exakt dasselbe Rauschen bekommt, verschiedene
    Tasks/Tensorbloecke aber unabhaengiges Rauschen.
    """
    if not config.NOISE_Y_ENABLED:
        return lambda y: y

    seed = int(config.NOISE_Y_SEED) + int(task["combination_id"]) * 100_000 + int(task["tensor_block"])
    rng = np.random.default_rng(seed)

    def add_noise(y):
        if rng.random() < config.NOISE_Y_ANTEIL:
            return y + rng.normal(loc=0.0, scale=config.NOISE_Y_SIGMA)
        return y

    return add_noise


def run_task(task, suite, tensor_noise, bounds, progress_callback=None):# ✅ 
    """Fuehrt EINE Task komplett aus: ein (Kombination, Tensorblock)-Paar.

    Gibt eine Liste von Zeilen zurueck, eine pro BO-Iteration (Long-Format),
    analog zu den Var_n-Spalten in df_gesamt_iteration im alten Skript.

    progress_callback: optional, wird nach jeder abgeschlossenen Iteration mit
    der aktuellen Iterationsnummer j aufgerufen (fuer Live-Fortschrittsanzeige,
    siehe run_instance.py). Standardmaessig None (kein Overhead, z.B. wenn
    run_task ausserhalb der Pipeline direkt aufgerufen wird).
    """
    funktion = int(task["funktion"])
    dimension = int(task["dimension"])
    surrogate_model = task["surrogate_model"]
    acquisition_model = task["acquisition_model"]
    sample_size_initial = int(task["sample_size_initial"])
    max_iteration = int(task["max_iteration"])
    tensor_block = int(task["tensor_block"])

    lower_bound, upper_bound = bounds

    # bereits in prepare_tasks.py einmalig pro (Funktion, Dimension) geschaetzt
    f_opt = float(task["f_opt_approx"])
    x_opt = np.asarray(json.loads(task["x_opt_approx"]), dtype=float)

    add_y_noise = _make_y_noise_fn(task)

    block = tensor_noise[tensor_block]
    train_X_np = block[:sample_size_initial, :dimension]
    train_Y_np = np.array([
        add_y_noise(evaluate_bbob_function(x, funktion, dimension, suite)) for x in train_X_np
    ])

    train_X = torch.as_tensor(train_X_np, dtype=torch.double)
    train_Y = torch.as_tensor(train_Y_np, dtype=torch.double).unsqueeze(-1)

    rows = []
    for j in range(1, max_iteration + 1):
        #Hier wird das nächste x vorhergesagt, bassiern dauf den 
        next_x, _ = select_and_run_variant(
            surrogate_model, acquisition_model, train_X, train_Y, dimension,
            lower_bound, upper_bound,
        )
        next_x_np = next_x.numpy()
        next_y_np = add_y_noise(evaluate_bbob_function(next_x_np, funktion, dimension, suite))

        train_X_np = np.vstack([train_X_np, next_x_np.reshape(1, -1)])
        train_Y_np = np.append(train_Y_np, next_y_np)
        train_X = torch.as_tensor(train_X_np, dtype=torch.double)
        train_Y = torch.as_tensor(train_Y_np, dtype=torch.double).unsqueeze(-1)

        best_so_far, simple_regret, immediate_regret, x_distances = compute_regret(
            train_X_np, train_Y_np, x_opt, f_opt
        )

        rows.append({
            **task,
            "iteration": j,
            "best_so_far": float(best_so_far[-1]),
            "simple_regret": float(simple_regret[-1]),
            "immediate_regret": float(immediate_regret[-1]),
            "x_distance": float(x_distances[-1]),
            "status": "ok",
            "error": "",
        })

        if progress_callback is not None:
            progress_callback(j)

        # Fruehzeitiger Abbruch: simple_regret (im Betrag, da er wegen der nur
        # approximierten f_opt auch negativ werden kann - siehe compute_regret())
        # ist bereits nahe 0 -> weitere Iterationen bringen kaum noch etwas,
        # Worker kann sofort mit der naechsten Task weitermachen. Schwelle
        # relativ zu |f_opt|, damit das auf allen Dimensionen gleich gut
        # greift (siehe Kommentar bei EARLY_STOP_* in config.py).
        if config.EARLY_STOP_ENABLED:
            early_stop_threshold = max(
                config.EARLY_STOP_MIN_EPSILON,
                config.EARLY_STOP_RELATIVE_EPSILON * abs(f_opt),
            )
            if abs(simple_regret[-1]) < early_stop_threshold:
                break

    return rows
