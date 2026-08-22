"""BO-Kernfunktionen fuer die verteilte Pipeline.

Eigenstaendige Kopie/Anpassung der Logik aus ROBO/coco_benchmark.py und
ROBO/Standard_PY.py (dort nicht veraendert), damit ROBO_NEU ohne Abhaengigkeit
zum alten Ordner auf jede LRZ-Instanz kopiert werden kann. Die weiteren
robusten Varianten (Imprecise GP, Relevance Pursuit, STABLEOPT, DRBO) sind
analog angepasste Kopien aus ROBO/, AIRBO ist eine schlanke Neuimplementierung
nach Yang et al. 2023 (ROBO/AIRBO.py selbst war leer, siehe airbo.py fuer
Details/Annahmen). Alle fuenf werden unten in select_and_run_variant()
angebunden.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401 - fuegt Pre_Processing/ (config) und BO_Verfahren/ zu sys.path hinzu

import glob
import json
import re
import shutil
import uuid

import cocoex
import numpy as np
import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

import config

# Hinweis: die Varianten-Module (imprecise_gp/relevance_pursuit/stable_opt/drbo)
# werden bewusst NICHT hier oben importiert, sondern erst lazy innerhalb der
# jeweiligen Verzweigung in select_and_run_variant(). Grund: relevance_pursuit.py
# braucht botorch.models.relevance_pursuit_model, das erst ab einer neueren
# botorch-Version existiert. Ein Top-Level-Import wuerde bei aelterem botorch
# den kompletten core_bo-Import zum Absturz bringen - auch fuer Varianten wie
# "GausianProzess", die davon gar nicht betroffen sind.


def evaluate_bbob_function(x, funktion, dimension, suite, instance=1):
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


def estimate_bbob_argmax(funktion, dimension, suite, lower_bound, upper_bound,
                          n_samples, seed, instance=1):
    """Schaetzt NUR den Punkt x_opt, an dem -f(x) im Suchraum (ungefaehr)
    maximal wird, per Zufallsstichprobe. Wird ausschliesslich fuer die
    Nebenmetrik x_distance gebraucht (siehe compute_regret()) - der zugehoerige
    Funktionswert f_opt kommt NICHT mehr von hier, sondern exakt aus
    get_bbob_fopt_exact(), da eine Zufallsstichprobe den wahren Maximalwert
    besonders bei hohen Dimensionen und stark zerkluefteten Funktionen (z.B.
    BBOB-Funktion 24) deutlich unterschaetzt (siehe Vergleich im Chat: bei
    d=20 teils >200% Abweichung vom wahren Wert). Der zugehoerige x_opt bleibt
    aus demselben Grund zwangslaeufig ebenfalls nur eine grobe Naeherung -
    x_distance ist entsprechend nur zur groben Orientierung zu verwenden,
    kein belastbares Genauigkeitsmass. Wird EINMAL pro (Funktion, Dimension)-
    Paar zentral in prepare_tasks.py aufgerufen, nicht pro Task, um die
    Kosten bei sehr vielen Kombinationen gering zu halten.
    """
    problem = suite.get_problem_by_function_dimension_instance(
        function=funktion, dimension=dimension, instance=instance
    )
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(lower_bound, upper_bound, size=(n_samples, dimension))
    values = np.array([-problem(x) for x in candidates])
    best_idx = int(np.argmax(values))
    return candidates[best_idx]


def get_bbob_fopt_exact(funktion, dimension, suite, instance=1):
    """Liefert das EXAKTE f_opt (Maximum von -f(x)) direkt aus dem COCO-
    Observer/Logger - keine Schaetzung noetig.

    COCO kennt fuer jede BBOB-Instanz intern den wahren Minimalwert von f(x)
    (Rohkonvention, Precision 1e-8), gibt ihn ueber die normale Problem-API
    aber nicht heraus (Sinn eines Black-Box-Benchmarks). Haengt man jedoch
    einen 'bbob'-Observer an und wertet die Funktion EINMAL beliebig aus
    (z.B. am Nullpunkt), schreibt der Logger eine .dat-Datei, deren erste
    Zeile den Wert als Kommentar enthaelt: "... - Fopt (7.948...e+01) ...".
    Das wird hier ausgelesen und negiert (-Fopt), um dieselbe -f(x)-
    Vorzeichenkonvention wie evaluate_bbob_function() zu erhalten.

    Der Log-Ordner wird in einen eindeutig benannten Unterordner unter
    ./exdata/ geschrieben und danach wieder geloescht, damit nichts vom
    Repo-Arbeitsverzeichnis uebrig bleibt (auch bei mehreren parallelen
    Aufrufen unkritisch, da jeder Aufruf einen eigenen Unterordner bekommt).
    """
    problem = suite.get_problem_by_function_dimension_instance(
        function=funktion, dimension=dimension, instance=instance
    )
    result_folder = f"fopt_probe_{funktion}_{dimension}_{instance}_{uuid.uuid4().hex[:8]}"
    observer = cocoex.Observer("bbob", f"result_folder: {result_folder}")
    try:
        problem.observe_with(observer)
        problem(np.zeros(dimension))  # ein einziger Dummy-Aufruf reicht, um das Log zu erzeugen
    finally:
        problem.free()

    dat_files = glob.glob(f"exdata/{result_folder}/data_f{funktion}/*.dat")
    if not dat_files:
        raise RuntimeError(
            f"COCO-Observer hat keine .dat-Datei fuer Funktion {funktion}, "
            f"Dimension {dimension} erzeugt (erwartet unter exdata/{result_folder}/)."
        )
    first_line = open(dat_files[0]).readline()
    match = re.search(r"Fopt \(([-\d.eE+]+)\)", first_line)
    if match is None:
        raise RuntimeError(
            f"Konnte 'Fopt (...)' nicht aus der .dat-Kopfzeile lesen: {first_line!r}"
        )
    fopt_raw = float(match.group(1))

    shutil.rmtree(f"exdata/{result_folder}", ignore_errors=True)

    return -fopt_raw


def compute_regret(X_history, Y_history, x_opt, f_opt):
    y_values = np.asarray(Y_history, dtype=float)
    x_opt = np.asarray(x_opt, dtype=float)

    x_distances = np.array([
        np.linalg.norm(np.asarray(x, dtype=float).reshape(-1) - x_opt) for x in X_history
    ])

    best_so_far = np.maximum.accumulate(y_values)
    simple_regret = f_opt - best_so_far
    immediate_regret = f_opt - y_values

    return best_so_far, simple_regret, immediate_regret, x_distances


def run_gaussian_process(train_X, train_Y, dim, acquisition_func, lower_bound, upper_bound):
    # Input auf [0,1] normalisieren + Output standardisieren: bei einem
    # Suchraum wie [-1000, 1000] ist der GP-Fit auf Rohwerten numerisch
    # instabil und schlaegt haeufig fehl ("not contained to the unit cube").
    # BoTorch macht das De-/Normalisieren intern transparent, d.h. next_x
    # kommt weiterhin in den originalen Bounds zurueck.
    gp_bounds = torch.tensor([[lower_bound] * dim, [upper_bound] * dim], dtype=torch.double)
    model = SingleTaskGP(
        train_X, train_Y,
        input_transform=Normalize(d=dim, bounds=gp_bounds),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    if acquisition_func == "EI":
        acqf = LogExpectedImprovement(model, best_f=train_Y.max())
    elif acquisition_func == "UCB":
        acqf = UpperConfidenceBound(model, beta=2.0, maximize=True)
    elif acquisition_func == "LCB":
        acqf = UpperConfidenceBound(model, beta=2.0, maximize=False)
    else:
        raise ValueError(f"Unbekannte Akquisitionsfunktion fuer GausianProzess: {acquisition_func}")

    bounds = torch.tensor([[lower_bound] * dim, [upper_bound] * dim], dtype=torch.double)
    next_x, acq_value = optimize_acqf(
        acqf, bounds=bounds, q=1, num_restarts=10, raw_samples=512,
    )
    return next_x.squeeze(0), acq_value


def select_and_run_variant(surrogate_model, acquisition_func, train_X, train_Y, dim,
                            lower_bound, upper_bound):
    """Analog zu waehle_und_berechne_variante() in ROBO/coco_benchmark.py.

    Alle Varianten haben dieselbe Signatur (train_X, train_Y, dim, acquisition_func,
    lower_bound, upper_bound) und geben (next_x, acq_value) zurueck, siehe die
    jeweiligen run_*()-Funktionen in imprecise_gp.py / relevance_pursuit.py /
    stable_opt.py / drbo.py / airbo.py fuer Details und modellspezifische
    Einschraenkungen (Imprecise GP: nur dim=1; STABLEOPT: nur
    acquisition_func="UCB"; DRBO: nur acquisition_func="distributionally
    robust UCB-Akquisition", nutzt die letzte Spalte von dim als
    Kontextvariable; AIRBO: nur acquisition_func="UCB", nimmt eine simulierte
    Gauss-Eingabeunsicherheit um jeden Punkt an, siehe airbo.py).
    """
    if surrogate_model == "GausianProzess":
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


def _make_y_noise_fn(task):
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


def run_task(task, suite, tensor_noise, bounds, progress_callback=None):
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
