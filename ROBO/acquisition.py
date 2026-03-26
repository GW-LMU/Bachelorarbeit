import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf



def train_acquisition(model_acquisition, trained_model, train_Y, value_range, dimension):
    if model_acquisition == "ei":
        best_f = train_Y.max()

        ei = ExpectedImprovement(trained_model, best_f=best_f)

        lower = [value_range[0]] * dimension
        upper = [value_range[1]] * dimension
        bounds = torch.tensor([lower, upper], dtype=torch.double)

        candidate, acq_value = optimize_acqf(
            ei,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=50,
        )

        return candidate, acq_value

    raise ValueError(f"Unbekanntes Acquisition-Modell: {model_acquisition}")