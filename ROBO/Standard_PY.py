import cocoex
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch


minus_area = -1000
plus_area = 1000


def run_gausian_prozess(train_X, train_Y, dim, acquisition_func,
                         lower_bound=minus_area, upper_bound=plus_area):
    """
    Führt die komplette Standard-GP-Variante für eine Iteration aus:
    1) SingleTaskGP fitten
    2) passende Akquisitionsfunktion anhand von acquisition_func wählen ("EI", "UCB", "LCB")
    3) nächsten Punkt per optimize_acqf vorschlagen

    train_X : torch.double, shape [n, dim]
    train_Y : torch.double, shape [n, 1] (Maximierungs-Konvention, siehe coco_benchmark.py)
    dim     : int, Dimension des Suchraums

    Rückgabe: next_x (torch.double, shape [dim]), acq_value (torch.double, Skalar)
    """
    # 1) Modell fitten
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # 2) Akquisitionsfunktion abfragen und passend wählen
    if acquisition_func == "EI":
        acqf = LogExpectedImprovement(model, best_f=train_Y.max())
    elif acquisition_func == "UCB":
        acqf = UpperConfidenceBound(model, beta=2.0, maximize=True)
    elif acquisition_func == "LCB":
        acqf = UpperConfidenceBound(model, beta=2.0, maximize=False)
    else:
        raise ValueError(f"Unbekannte Akquisitionsfunktion für GausianProzess: {acquisition_func}")

    # 3) nächsten Punkt vorschlagen
    bounds = torch.tensor(
        [[lower_bound] * dim, [upper_bound] * dim], dtype=torch.double
    )
    next_x, acq_value = optimize_acqf(
        acqf,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
    )

    return next_x.squeeze(0), acq_value