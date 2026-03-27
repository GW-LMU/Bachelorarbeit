from Traindata import train_data_problem, evaluate_candidate
from surrogate import train_surrogatmodell
from acquisition import train_acquisition
import torch
import cocoex
import random
import numpy as np
import pandas as pd



import pandas as pd
import torch

def iteration_sample_cal(
    n_iteration,
    n_samples,
    train_X_tensor,
    train_Y_tensor,
    fun,
    dimension,
    model,
    model_acquisition,
    value_range
):
    # Name erzeugen
    name = f"df_{fun}_{dimension}_{model}_{model_acquisition}"

    # Leeres DataFrame
    df = pd.DataFrame(columns=[
        "Funktion",
        "Dimension",
        "Surrogate_Model",
        "Acquisition_Model",
        "Sample_Size",
        "Iteration",
        "MSE"
    ])

    train_X = train_X_tensor.clone()
    train_Y = train_Y_tensor.clone()

    for n in range(n_samples):
        for i in range(n_iteration):

            trained_model = train_surrogatmodell(model, train_X, train_Y)

            x_next = train_acquisition(
                model_acquisition,
                trained_model,
                train_Y,
                value_range,
                dimension
            )

            y_next = evaluate_candidate(
                x_next[0],
                fun,
                n_samples,
                value_range,
                dimension
            )

            train_X = torch.cat([train_X, x_next], dim=0)
            train_Y = torch.cat([train_Y, y_next], dim=0)

            mse = 23 * 3  # später ersetzen

            df.loc[len(df)] = [
                fun,
                dimension,
                model,
                model_acquisition,
                n,
                i,
                mse
            ]

    return df, name

