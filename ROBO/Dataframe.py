import pandas as pd
import itertools
import numpy as np
from IPython.display import display
from Traindata import train_data_problem

n_sample = 50
iteration = 10

vec_fun = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
vec_dim = [1,2,5,10, 15, 25,50]
vec_var = ["GP", "IGP"]
acq = ["IP", "EP", "NL"]
vec_sample = list(range(1, n_sample + 1 ))
vec_iter = list(range(1, iteration + 1 ))

kombination = list(itertools.product(vec_fun,vec_dim, vec_var, acq, vec_iter, vec_sample))

df_kombination = pd.DataFrame(kombination, columns=['Funktion','Dimension','Varianten', 'Acqisition', 'Sample', 'Iteration'])

df_kombination['MSE'] = np.nan

print(f"Anzahl Kombinationen :{len(df_kombination)}")
print(df_kombination.head(10))
print("...")
print(df_kombination.tail(10))