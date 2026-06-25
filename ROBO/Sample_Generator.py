import pandas as pd
import itertools
import numpy as np
from IPython.display import display
from Traindata import train_data_problem

# Anzahl an Sample mit dem Inizial der Prozess gestartet wird( wie viel vorwissen mitgeben wird)
n_sample_init =[1,2,5,10,25,50,100]


# Anzahl der Wiedrholten Verusche

n_sample_stat = 90

# Anzahl der Iteration die in einen Prozess geben bestimmter Paramter berehcnet wird
iteration = 200

# Die Folgenden Funktion stehen zu Auswahl und können abgefragt werdern 
# [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
vec_fun = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

# Dimensionen an denen der Algorithmus getetest wird
vec_dim = [1,2,5,10, 15, 25, 50]


vec_var = ["GP", "IGP"]
acq = ["UP", "EP", "NL"]
#vec_sample = list(range(1, n_sample + 1 ))
#vec_iter = list(range(1, iteration + 1 ))

df_suggo_aqui = pd.DataFrame({
    "Surrogate_Model": ["GausianProzess", "GausianProzess", "GausianProzess", "Imprecise_GausianProzess_Rodemann", "AIRBO", "AIRBO", "STABLEOPT","DRBO", "Relevance Pursuit"],
    "Acquisitions_Model": ["LCB", "UCB", "EI", "GLCB", "UCB","EI" "UCB", "distributionally robust UCB-Akquisition", "qLogNEI"]
})





########### Seeds für Statistische Unsicherheit #############

seeds = [48291, 730184, 15937, 904622, 318450, 67219, 845301, 290776, 513908, 76402,
         991245, 406833, 128594, 557019, 683741, 230465, 719008, 364952, 875116, 492607,
         105938, 638214, 781590, 249731, 930856, 572084, 316729, 864015, 451298, 697430,
         82476, 209583, 746901, 385214, 918673, 540127, 167892, 802436, 624019, 973251,
         438706, 715894, 296318, 851027, 604735, 139486, 768250, 325971, 947608, 512364]


########### Grid Boundies #############

minus_area = -1000
plus_area = 1000



########### Tenssor für Initzial Sample #############

def tensor_generator_1(vec_dim, n_sample_init, n_sample_stat, seeds):
    tensor = np.empty(
        (n_sample_stat, max(n_sample_init), max(vec_dim)),
        dtype=np.float32
    )

    for i in range(n_sample_stat):
        rng = np.random.default_rng(seeds[i])

        tensor[i] = rng.uniform(
            low=minus_area,
            high=plus_area,
            size=(max(n_sample_init), max(vec_dim))
        ).astype(np.float32)

    return tensor


def tensor_info(tensor):
    print("Tensor-Eigenschaften")
    print("--------------------")
    print("Typ:", type(tensor))
    print("Shape:", tensor.shape)
    print("Anzahl Achsen / Dimensionen:", tensor.ndim)
    print("Anzahl aller Werte:", tensor.size)
    print("Datentyp:", tensor.dtype)
    print("Minimum:", tensor.min())
    print("Maximum:", tensor.max())
    print("Mittelwert:", tensor.mean())
    print("Standardabweichung:", tensor.std())

    
#te = tensor_generator(vec_dim, n_sample_init,n_sample_stat, seeds[0]).astype(np.float32)

te_1 = tensor_generator_1(vec_dim, n_sample_init,n_sample_stat, seeds)

display(te_1)
tensor_info(te_1)

