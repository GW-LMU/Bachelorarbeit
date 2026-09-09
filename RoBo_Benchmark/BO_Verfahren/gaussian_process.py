
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401 - fuegt Pre_Processing/ (config) zu sys.path hinzu

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


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
