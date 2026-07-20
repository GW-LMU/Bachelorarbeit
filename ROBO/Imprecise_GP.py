import numpy as np
import torch
from uncertainty_aware_bo import BayesOptimizer, matern


minus_area = -1000
plus_area = 1000


def run_imprecise_gausian_prozess(train_X, train_Y, dim, acquisition_func,
                                   lower_bound=minus_area, upper_bound=plus_area,
                                   kernel=matern, c=50.0, length_scale=1.0, outputscale=1.0,
                                   n_candidates=500):
    """
    Führt die Imprecise-Gaussian-Process-Variante (Rodemann) für eine Iteration aus:
    1) BayesOptimizer aus uncertainty_aware_bo.py mit den bisherigen Beobachtungen bauen
    2) passende Akquisitionsfunktion anhand von acquisition_func wählen ("LCB", "GLCB")
    3) nächsten Punkt über ein Kandidatenraster im Suchraum vorschlagen

    Hinweis: BayesOptimizer arbeitet mit skalaren Kernel-Distanzen (np.subtract.outer)
    und unterstützt daher nur den 1D-Fall (dim=1). Für dim>1 wird bewusst ein
    NotImplementedError geworfen, statt mathematisch falsche Ergebnisse zu liefern.

    train_X : torch.double, shape [n, dim]
    train_Y : torch.double, shape [n, 1] (Maximierungs-Konvention, siehe coco_benchmark.py)
    dim     : int, Dimension des Suchraums

    Rückgabe: next_x (torch.double, shape [dim]), acq_value (torch.double, Skalar)
    """
    if dim != 1:
        raise NotImplementedError(
            "Imprecise_GausianProzess_Rodemann (BayesOptimizer) unterstützt aktuell nur "
            f"dim=1, da die Kernelfunktionen (matern/rbf) auf skalaren Distanzen basieren. "
            f"Angefragt wurde dim={dim}."
        )

    x_obs = train_X.squeeze(-1).detach().cpu().numpy()
    y_obs = train_Y.squeeze(-1).detach().cpu().numpy()

    # 1) Modell mit den bisherigen Beobachtungen aufbauen
    bo = BayesOptimizer(
        x_obs=x_obs,
        y_obs=y_obs,
        kernel=kernel,
        c=c,
        length_scale=length_scale,
        outputscale=outputscale,
    )

    # 2) Akquisitionsfunktion abfragen und passend wählen
    if acquisition_func == "LCB":
        af = bo.UCB
    elif acquisition_func == "GLCB":
        af = bo.PROBO
    else:
        raise ValueError(
            f"Unbekannte Akquisitionsfunktion für Imprecise_GausianProzess_Rodemann: {acquisition_func}"
        )

    # 3) Kandidatenraster über den gesamten Suchraum, bestes x auswählen
    x_candidates = np.linspace(lower_bound, upper_bound, n_candidates)
    acquisition_values = np.array([af(x) for x in x_candidates])

    best_idx = np.argmax(acquisition_values)
    next_x = torch.tensor([x_candidates[best_idx]], dtype=torch.double)
    acq_value = torch.tensor(acquisition_values[best_idx], dtype=torch.double)

    return next_x, acq_value
