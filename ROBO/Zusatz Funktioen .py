import torch
import cocoex
import random   
import itertools
import numpy as np
import pandas as pd
from surrogate import train_surrogat
from acquisition import train_acquisition


seed = 42
n_samples = 20
dim = 40
dim_coco = [2, 3, 5, 10, 20, 40]
n_iteration = 20

value_range = (-100,100)
# [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
vec_fun = [1,2,3,4]
vec_var = ["GP"]
# ["GP", "IGP"]
acq = ["ei"]

#["ei", "acqf"]
vec_sample = list(range(1, n_samples + 1 ))



#==================================================================================#


#======== Loading Suite Cocoex =======#

def loading_load_suite_coco():
    suite = cocoex.Suite("bbob", "","")
    return(suite)

#======== Trainings Daten X erzeugen =======#

def draw_n_train_samples(n_samples, seed, dim , value_range):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_X = torch.rand(n_samples, dim) * (value_range[1] - value_range[0])+ value_range[0]
    print(train_X)
    return(train_X)

#======== Trainings Daten Y berechnen =======#

def calculate_train_samples(train_X, suite,fun, dim):
    problem = suite.get_problem_by_function_dimension_instance(
        function = fun,
        dimension = dim,
        instance = 1
    )

    train_Y = []
    for x in train_X:
        x_numpy = x.numpy()
        y_val = problem(x_numpy)
        train_Y.append(y_val)
        print(f"f({x_numpy}) = {y_val:.6f}")


        # train_Y = np.array([problem(x) for x in X]) Vektorisierte Version oft schneller

    return(train_Y)

#======== Erstellugn Dataframe Komninationen=======#

def df_kombination_function(vec_fun, dim_coco, vec_var, acq):
    kombination = list(itertools.product(vec_fun, dim_coco, vec_var, acq))
    
    # ❗ nur 4 Spalten hier
    df_kombination = pd.DataFrame(
    kombination,
    columns=["Funktion", "Dimension", "Surrogate_Model", "Acquisitions_Model"]
    )

    df_kombination["Sample_Size"] = np.nan
    df_kombination["Iteration"] = np.nan
    df_kombination["MSE"] = np.nan
    
    return df_kombination

def getMinimumWithArgmin(suite, funktion, dimension, instance=1):
    problem = suite.get_problem_by_function_dimension_instance(
        funktion, dimension, instance
    )

    problem._best_parameter("print")
    x_opt = np.loadtxt("._bbob_problem_best_parameter.txt")

    f_min = problem(x_opt)

    return (
        torch.tensor(x_opt, dtype=torch.double),
        torch.tensor(f_min, dtype=torch.double)
    )


#==================================================================================#

#Suite laden 
suite = loading_load_suite_coco()

# Zieldateframe inizialieren 
df_new = pd.DataFrame(columns=[
        "Funktion",
        "Dimension",
        "Surrogate_Model",
        "Acquisitions_Model",
        "Sample_Size",
        "Iteration",
        "MSE",
        "max_value"
    ])

#Datenframe mit Kombinationen initialiersiern
df_kombination = df_kombination_function(vec_fun, dim_coco, vec_var, acq)
df_kombination.to_excel("df_kombiantion.xlsx", index=False, header=False)

#Dateframe mit allen Sampels erstellen 
train_X = draw_n_train_samples(n_samples, seed, dim, value_range)
df_sample = pd.DataFrame(train_X.numpy() )

df_sample.to_excel("output.xlsx", index=False, header=False)

# Hauptalgorithmus 
# Iterriern über df_kombiniation und nimmt sich je nach Dimension und Samplegröße die richtigeb Daren aus df
for row in df_kombination.itertuples():
    min_value = getMinimumWithArgmin(suite, row.Funktion, row.Dimension, instance=1)
    for n in range(1, n_samples + 1):
        tensor_X = torch.tensor(df_sample.values[:n, :row.Dimension], dtype = torch.double)  # Fehel :n und row.Dimension umdrehen 
        tensor_Y = calculate_train_samples(tensor_X, suite,row.Funktion, row.Dimension)
        tensor_Y = torch.tensor(tensor_Y, dtype = torch.double).reshape(-1, 1)
        print("X shape:", tensor_X.shape)
        print("Y shape:", tensor_Y.shape)   
        for i in range(n_iteration):
            gp_model = train_surrogat(row.Surrogate_Model, tensor_X, tensor_Y)
            new_candidate = train_acquisition(row.Acquisitions_Model, gp_model, tensor_Y, value_range, row.Dimension)
            print(new_candidate)

            new_candidate_value = calculate_train_samples(new_candidate[0], suite,row.Funktion, row.Dimension)
            new_val = new_candidate_value[0]
        
            dist = torch.norm(new_val - min_value[0], p=2)
            
            df_new.loc[len(df_new)] = [row.Funktion,row.Dimension,row.Surrogate_Model,row.Acquisitions_Model,n,i,
                                       dist.item() if torch.is_tensor(dist) else dist,
                                       min_value.item() if torch.is_tensor(min_value) else min_value]

df_new.to_excel("output_Kombi.xlsx", index=False, header=False) # nach jeder Zeile an Kombination für zwischenstandsspeicherung 



# - GPU Auslagerung 
# - Berechungsstand