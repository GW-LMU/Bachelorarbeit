import pandas as pd
import itertools
import numpy as np
from IPython.display import display
from Traindata import train_data_problem

n_sample = 50
iteration = 10
# [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
vec_fun = [0,1,2]
vec_dim = [1,2,5,10, 15, 25, 50]
vec_var = ["GP", "IGP"]
acq = ["IP", "EP", "NL"]
vec_sample = list(range(1, n_sample + 1 ))
vec_iter = list(range(1, iteration + 1 ))

def df_kombination_function(vec_fun, vec_dim, vec_var, acq):
    kombination = list(itertools.product(vec_fun, vec_dim, vec_var, acq))
    
    # ❗ nur 4 Spalten hier
    df_kombination = pd.DataFrame(
    kombination,
    columns=["Funktion", "Dimension", "Surrogate_Model", "Acquisitions_Model"]
    )

    df_kombination["Sample_Size"] = np.nan
    df_kombination["Iteration"] = np.nan
    df_kombination["MSE"] = np.nan
    
    return df_kombination


def df_kombination_function_number(vec_fun, vec_dim, vec_var, acq,vec_sample,vec_iter):
    kombination = list(itertools.product(vec_fun,vec_dim, vec_var, acq, vec_sample, vec_iter))
    df_kombination = pd.DataFrame(kombination, columns=["Funktion", "Dimension", "Surrogate_Model", "Acquisitions_Model", "Samle_Size", "Iteration", "MSE"])
    df_kombination['MSE'] = np.nan
    print(f"Anzahl Kombinationen :{len(df_kombination)}")
    

def df_count_kombination(vec_fun, vec_dim, vec_var, acq,vec_sample,vec_iter):
    kombinations = len(vec_fun)*len(vec_dim)*len(vec_var)*len(acq)*len(vec_sample)*len(vec_iter)
    print(f"Anzahl Kombinationen :{kombinations}")


