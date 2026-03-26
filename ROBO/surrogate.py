import pandas as pd
import itertools
import numpy as np
from IPython.display import display
from Traindata import train_data_problem
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

def train_surrogatmodell(model , train_X_tensor, train_Y_tensor):
    if model == "GP":
        model = SingleTaskGP(train_X_tensor, train_Y_tensor)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        posterior = model.posterior(train_X_tensor)

        mean = posterior.mean
        variance = posterior.variance
    else:
        print("Hello")
    return(mean, variance)