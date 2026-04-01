import torch
import cocoex
import random   
import itertools
import numpy as np
import pandas as pd


value_range = (-100,100)
dim = 1000
n_sample = 60
n_iteration = 10
# [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
vec_fun = [1,2]
vec_dim = [1,2,5,10, 15, 25, 50]
vec_var = ["GP", "IGP"]
acq = ["IP", "EP", "NL"]
vec_sample = list(range(1, n_sample + 1 ))






#======== Loading Suite Cocoex =======#

def loading_load_suite_coco():
    suite = cocoex.Suite("bbbob", "","")
    return(suite)


#======== Trainings Daten X erzeugen =======#

def draw_n_train_samples(n_samples, seed, dim , value_range):
    random.seed(seed)e
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_X = torch.rand(n_samples, dim) * (value_range[1] - value_range[0])+ value_range[0]
    print(train_X)

    return(train_X)


#======== Erstellugn Dataframe Komninationen=======#


def df_kombination_function(vec_fun, dim, vec_var, acq):
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







df_new = pd.DataFrame(columns=[
        "Funktion",
        "Dimension",
        "Surrogate_Model",
        "Acquisitions_Model",
        "Sample_Size",
        "Iteration",
        "MSE"
    ])


df_kombination = df_kombination_function(vec_fun, dim, vec_var, acq)


train_X = draw_n_train_samples(1000, 42, dim, value_range)
df = pd.DataFrame(train_X.numpy() )
df.to_excel("output.xlsx", index=False, header=False)

a = 0
for row in df_kombination.itertuples():
    for n in range(1, n_sample):
        tensor = torch.tensor(df.values[:n, :row.Dimension], dtype=torch.float32)
        for i in range(n_iteration):
            print(row,n,i)
            df_new.loc[len(df_new)] = [row.Funktion,row.Dimension,row.Surrogate_Model,row.Acquisitions_Model,n,i,np.nan]


df_new.to_excel("output_Kombi.xlsx", index=False, header=False)