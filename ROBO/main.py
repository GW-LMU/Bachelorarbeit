from Traindata import train_data_problem, evaluate_candidate
from surrogate import train_surrogatmodell
from acquisition import train_acquisition
from iteration_sample import iteration_sample_cal
from Dataframe import df_kombination_function, df_count_kombination
import torch
import cocoex
import random
import numpy as np
import pandas as pd


# Input Iterationschleifen
n_iteration = 50
n_samples = 10 

# Input für df_kombination 
#[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

funktionen = [ 1,2,3]
model = ["GP", "LP"]
model_acquisition = ["ei"]
fun = 1                             
value_range = (-5, 5)      
dimension = [2, 3]
vec_sample = list(range(1, n_samples + 1 ))
vec_iter = list(range(1, n_iteration + 1 ))

dfs  = {}

df_count_kombination(funktionen, dimension, model, model_acquisition,vec_sample, vec_iter) # hier nummerische Berechung schlauer 2*3*4*5
df_kombination = df_kombination_function(funktionen, dimension, model, model_acquisition)               
print(df_kombination.head(5))
print("...")
print(df_kombination.tail(5))

for row in df_kombination.itertuples():

    iteration_sample_cal(n_iteration, n_samples, row.Funktion, row.Dimension,
                         row.Surrogate_Model, row.Acquisitions_Model, value_range)
    name = iteration_sample_cal[1]
    dfs[name] = iteration_sample_cal(...)[0]



df_gesamt = pd.concat(dfs.values(), ignore_index=True)
    
    
    



