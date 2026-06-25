import pandas as pd
import itertools
import numpy as np
from IPython.display import display
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood



def train_surrogat(model, tensor_X, tensor_Y):
    if model == "GP":
        gp_model = train_surrogat_GP(tensor_X, tensor_Y)
    elif model == "IGP":
        gp_model = train_surrogat_IGP(tensor_X, tensor_Y)
    else:
        raise ValueError(f"Unknown model: {model}")
    return gp_model




def train_surrogat_GP(tensor_X, tensor_Y):
        gp_model = SingleTaskGP(tensor_X, tensor_Y)
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        fit_gpytorch_model(mll)   # legacy API; fit_gpytorch_mll is the newer recommended helper
        return gp_model




def train_surrogat_IGP(tensor_X, tensor_Y):
        gp_model = SingleTaskGP(tensor_X, tensor_Y)
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        fit_gpytorch_model(mll)   # legacy API; fit_gpytorch_mll is the newer recommended helper
        return gp_model