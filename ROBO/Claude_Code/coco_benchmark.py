import numpy as np
import cocoex


def evaluate_bbob_function(x, funktion, dimension, instance=1, suite=None, maximize=False):
    """
    Berechnet den Funktionswert eines COCO/BBOB-Benchmarkproblems.

    x         : multidimensionaler Vektor (Liste, np.ndarray oder torch.Tensor) der Länge `dimension`
    funktion  : int, welche der 24 BBOB-Funktionen verwendet wird (1-24)
    dimension : int, Dimension des Suchraums (muss len(x) entsprechen)
    instance  : int, welche Instanz der Funktion verwendet wird (Standard: 1)
    suite     : optional bereits geladene cocoex.Suite, spart wiederholtes Laden
    maximize  : bool, falls True wird -f(x) zurückgegeben, damit BO (die maximiert)
                dasselbe Problem als Maximierung behandeln kann
    """
    if suite is None:
        suite = cocoex.Suite("bbob", "", "")

    problem = suite.get_problem_by_function_dimension_instance(
        function=funktion,
        dimension=dimension,
        instance=instance
    )

    x_numpy = np.asarray(x, dtype=float).reshape(-1)

    if x_numpy.shape[0] != dimension:
        raise ValueError(
            f"Länge von x ({x_numpy.shape[0]}) stimmt nicht mit dimension ({dimension}) überein"
        )

    y_val = problem(x_numpy)
    if maximize:
        y_val = -y_val

    print(f"Funktion {funktion}, Dimension {dimension}: f({x_numpy}) = {y_val:.6f}")

    return y_val


def get_bbob_optimum(funktion, dimension, instance=1, suite=None, maximize=False):
    """
    Gibt das bekannte Optimum (x_opt, f_opt) eines COCO/BBOB-Benchmarkproblems zurück.

    Gleiche Input-Parameter wie `evaluate_bbob_function` (ohne x, da das Optimum
    nicht von einem konkreten x abhängt, sondern eine feste Eigenschaft des
    gewählten funktion/dimension/instance-Tripels ist). Dient als Referenzwert
    für den späteren Vergleich mit den vom Optimierer gefundenen Ergebnissen.

    funktion  : int, welche der 24 BBOB-Funktionen verwendet wird (1-24)
    dimension : int, Dimension des Suchraums
    instance  : int, welche Instanz der Funktion verwendet wird (Standard: 1)
    suite     : optional bereits geladene cocoex.Suite, spart wiederholtes Laden
    maximize  : bool, falls True wird f_opt als -f_min zurückgegeben, passend zu
                `evaluate_bbob_function(..., maximize=True)`. x_opt bleibt dabei
                unverändert (die Optimalstelle verschiebt sich durch Negieren nicht)
    """
    if suite is None:
        suite = cocoex.Suite("bbob", "", "")

    problem = suite.get_problem_by_function_dimension_instance(
        function=funktion,
        dimension=dimension,
        instance=instance
    )

    problem._best_parameter("print")
    x_opt = np.loadtxt("._bbob_problem_best_parameter.txt")
    f_min = problem(x_opt)

    f_opt = -f_min if maximize else f_min

    print(f"Funktion {funktion}, Dimension {dimension}: Optimum bei x_opt={x_opt}, f_opt={f_opt:.6f}")

    return x_opt, f_opt


def compute_regret(X_history, Y_history, x_opt, f_opt):
    """
    Berechnet den Regret-Verlauf für eine Historie von BO-vorgeschlagenen Punkten.

    X_history : Liste/Array von Punkten x_1, ..., x_T, in der Reihenfolge, in der sie
                von der BO vorgeschlagen wurden
    Y_history : bereits berechnete Funktionswerte y_1, ..., y_T zu X_history (gleiche
                Reihenfolge, gleiches Vorzeichen-Konvention wie x_opt/f_opt)
    x_opt     : bekannte wahre Optimalstelle, z.B. aus get_bbob_optimum(...)
    f_opt     : bekannter wahrer Optimalwert, z.B. aus get_bbob_optimum(...)

    Gibt direkt einzelne Arrays (je ein Wert pro Iteration) zurück, in dieser Reihenfolge:
        best_so_far       : bestes bisher beobachtetes y (laufendes Optimum)
        simple_regret     : f_opt - best_so_far (Standardmetrik, monoton fallend)
        immediate_regret  : f_opt - y (Regret des aktuellen Punkts, kann wieder steigen)
        x_distance        : ||x_t - x_opt|| (Distanz im Suchraum zur wahren Optimalstelle)
    """
    y_values = np.asarray(Y_history, dtype=float)
    x_opt = np.asarray(x_opt, dtype=float)

    x_distances = np.array([
        np.linalg.norm(np.asarray(x, dtype=float).reshape(-1) - x_opt) for x in X_history
    ])

    best_so_far = np.maximum.accumulate(y_values)
    simple_regret = f_opt - best_so_far
    immediate_regret = f_opt - y_values

    return best_so_far, simple_regret, immediate_regret, x_distances


if __name__ == "__main__":
    x = [0.5, -1.2, 3.0]

    # Minimierung (Standard)
    y = evaluate_bbob_function(x, funktion=1, dimension=3)
    x_opt, f_min = get_bbob_optimum(funktion=1, dimension=3)

    # Maximierung (z.B. für Bayesian Optimization)
    y_max = evaluate_bbob_function(x, funktion=1, dimension=3, maximize=True)
    x_opt, f_max = get_bbob_optimum(funktion=1, dimension=3, maximize=True)

    # Regret-Verlauf über eine (Beispiel-)Historie vorgeschlagener BO-Punkte
    X_history = [
        [0.5, -1.2, 3.0],
        [0.1, -0.5, 1.2],
        [0.0, 0.0, 0.1],
    ]
    Y_history = [
        evaluate_bbob_function(x, funktion=1, dimension=3, maximize=True) for x in X_history
    ]
    best_so_far, simple_regret, immediate_regret, x_distances = compute_regret(
        X_history, Y_history, x_opt, f_max
    )
    print("simple_regret:", simple_regret)


def waehle_und_berechne_variante(surrogate_model, acquisition_func, train_X, train_Y, dim,
                                  lower_bound=minus_area, upper_bound=plus_area):
    """
    Nimmt eine Zeile aus df_suggo_aqui (surrogate_model, acquisition_func) und führt
    genau die dazu passende Variante aus: Surrogatmodell fitten, passende Akquisitions-
    funktion bauen, nächsten Punkt vorschlagen.

    train_X : torch.double, shape [n, dim]
    train_Y : torch.double, shape [n, 1] (Maximierungs-Konvention, siehe coco_benchmark.py)

    Rückgabe: next_x (torch.double, shape [dim]), acq_value (torch.double, Skalar)
    """
    bounds = torch.tensor(
        [[lower_bound] * dim, [upper_bound] * dim], dtype=torch.double
    )

    if surrogate_model == "GausianProzess":
        model = _gp_fitten(train_X, train_Y)
        acqf = _gp_acquisition_waehlen(acquisition_func, model, train_Y)

    elif surrogate_model == "Imprecise_GausianProzess_Rodemann":
        raise NotImplementedError(
            "Variante 'Imprecise_GausianProzess_Rodemann' ist noch nicht implementiert."
        )

    elif surrogate_model == "AIRBO":
        raise NotImplementedError("Variante 'AIRBO' ist noch nicht implementiert.")

    elif surrogate_model == "STABLEOPT":
        raise NotImplementedError("Variante 'STABLEOPT' ist noch nicht implementiert.")

    elif surrogate_model == "DRBO":
        raise NotImplementedError("Variante 'DRBO' ist noch nicht implementiert.")

    elif surrogate_model == "Relevance Pursuit":
        raise NotImplementedError("Variante 'Relevance Pursuit' ist noch nicht implementiert.")

    else:
        raise ValueError(f"Unbekanntes Surrogatmodell: {surrogate_model}")

    next_x, acq_value = optimize_acqf(
        acqf,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
    )

    return next_x.squeeze(0), acq_value
