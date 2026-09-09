
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401 - fuegt Pre_Processing/ (config) zu sys.path hinzu

import numpy as np
import torch

import config
from uncertainty_aware_bo import BayesOptimizer, matern


def run_imprecise_gausian_prozess(train_X, train_Y, dim, acquisition_func,
                                   lower_bound=config.MINUS_AREA, upper_bound=config.PLUS_AREA,
                                   kernel=matern, c=50.0, length_scale=1.0, outputscale=1.0,
                                   n_candidates=500):
    """
    Fuehrt die Imprecise-Gaussian-Process-Variante (Rodemann) fuer eine Iteration aus:
    1) BayesOptimizer aus uncertainty_aware_bo.py mit den bisherigen Beobachtungen bauen
    2) passende Akquisitionsfunktion anhand von acquisition_func waehlen ("LCB", "GLCB")
    3) naechsten Punkt ueber Kandidaten im Suchraum vorschlagen

    Zwei Faelle je nach dim (siehe BayesOptimizer in uncertainty_aware_bo.py):
    - dim == 1: unveraendert wie bisher - skalare Kernel-Distanzen (np.subtract.outer),
      Kandidaten als aequidistantes 1D-Gitter (np.linspace).
    - dim > 1: neue Erweiterung aus homo_bo_compare_af.ipynb (Zelle 19) - vektorisierter
      Kernel ueber R^d (matern_nd/rbf_nd), Kandidaten als zufaellige Punkte in der
      d-dimensionalen Box (np.random.uniform).

    train_X : torch.double, shape [n, dim]
    train_Y : torch.double, shape [n, 1] (Maximierungs-Konvention, siehe core_bo.py)
    dim     : int, Dimension des Suchraums

    Rueckgabe: next_x (torch.double, shape [dim]), acq_value (torch.double, Skalar)
    """
    y_obs = train_Y.squeeze(-1).detach().cpu().numpy()

    if dim == 1:
        # --- Fall dim == 1: unveraendert wie bisher ---
        x_obs = train_X.squeeze(-1).detach().cpu().numpy()
        x_candidates = np.linspace(lower_bound, upper_bound, n_candidates)
    else:
        # --- Fall dim > 1: neue, vektorisierte Erweiterung ---
        x_obs = train_X.detach().cpu().numpy()  # shape (n, dim)
        x_candidates = np.random.uniform(lower_bound, upper_bound, size=(n_candidates, dim))

    # 1) Modell mit den bisherigen Beobachtungen aufbauen
    bo = BayesOptimizer(
        x_obs=x_obs,
        y_obs=y_obs,
        kernel=kernel,
        c=c,
        length_scale=length_scale,
        outputscale=outputscale,
        dim=dim,
    )

    # 2) Akquisitionsfunktion abfragen und passend waehlen
    if acquisition_func == "LCB":
        af = bo.UCB
    elif acquisition_func == "GLCB":
        af = bo.PROBO
        # Hier mit festen tau=1.0, rho=1.0
    else:
        raise ValueError(
            f"Unbekannte Akquisitionsfunktion fuer Imprecise_GausianProzess_Rodemann: {acquisition_func}"
        )

    # 3) Kandidaten auswerten, bestes x auswaehlen
    acquisition_values = np.array([af(x) for x in x_candidates])
    best_idx = np.argmax(acquisition_values)

    if dim == 1:
        next_x = torch.tensor([x_candidates[best_idx]], dtype=torch.double)
    else:
        next_x = torch.tensor(x_candidates[best_idx], dtype=torch.double)
    acq_value = torch.tensor(acquisition_values[best_idx], dtype=torch.double)

    return next_x, acq_value
