import torch
import cocoex
import random   
import itertools
import numpy as np
import pandas as pd


seed = 42
n_samples = 60
dim = 40
dim_coco = [2, 3, 5, 10, 20, 40]
n_iteration = 10

value_range = (-100,100)
# [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
vec_fun = [1,2]
vec_var = ["GP", "IGP"]
acq = ["IP", "EP", "NL"]
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
        "MSE"
        "max_value"
    ])

#Datenframe mit Kombinationen initialiersiern
df_kombination = df_kombination_function(vec_fun, dim_coco, vec_var, acq)

#Datefraem mit allen Sampels erstellen 
train_X = draw_n_train_samples(n_samples, seed, dim, value_range)
df_sampel = pd.DataFrame(train_X.numpy() )
df_sampel.to_excel("output.xlsx", index=False, header=False)

# Hauptalgorithmus 
# Iterriern über df_kombiniation und nimmt sich je nach Dimension und Samplegröße die richtigeb Daren aus df
for row in df_kombination.itertuples():
    min_value = getMinimum(suite.row.Funktion, row.Dimension)
    for n in range(1, n_samples + 1):
        tensor_X = torch.tensor(df_sampel.values[:n, :row.Dimension], dtype=torch.float32)
        tensor_Y = calculate_train_samples(tensor_X, suite,row.Funktion, row.Dimension)
        for i in range(n_iteration):
        #Erstmaliges Fitten des Modell 
        # if row_SuggorateModel == "EP":
        #   fitten der Gauprozess
        # elif row.surrpgateModel =="GP":
        # fitten des Imprecise Gauprozss
        # Rückgabe 
        # Qquisitionsfunktion fitten 
        # Neuen Kandiaten auswählen 
        # Neuen Kandiaten evaluieren 
        # Messung durch fühühenen 
        # Alle werte dem Datafraem hinzuügen 
            df_new.loc[len(df_new)] = [row.Funktion,row.Dimension,row.Surrogate_Model,row.Acquisitions_Model,n,i,999,min_value]

df_new.to_excel("output_Kombi.xlsx", index=False, header=False)