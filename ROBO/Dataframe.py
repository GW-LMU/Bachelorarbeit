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
iteration = [100]
n_iterations = 100

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
    "Surrogate_Model": ["GausianProzess", "GausianProzess", "GausianProzess", "Imprecise_GausianProzess_Rodemann", "Imprecise_GausianProzess_Andere"],
    "Acquisitions_Model": ["EI", "PI", "UCB", "EI", "EIL"]
})

#"EI"  = Expected Improvement, EI
#"PI" = Probability of Improvement, PI
#"UCB" = Upper Confidence Bound, UCB




# Erstellung Dataframe - Kombinationen 


def df_kombination_function(vec_fun, vec_dim, surrogate_model, acq_model, n_sample_init, iteration):
    kombination = list(itertools.product(
        vec_fun,
        vec_dim,
        [surrogate_model],
        [acq_model],
        n_sample_init,
        iteration
    ))

    df_kombination = pd.DataFrame(
        kombination,
        columns=[
            "Funktion",
            "Dimension",
            "Surrogate_Model",
            "Acquisitions_Model",
            "Sample_Size_Initial",
            "Iteration"
        ]
    )

    return df_kombination


dfs = {}

for i in range(len(df_suggo_aqui)):
    surrogate_model = df_suggo_aqui.iloc[i, 0]
    acq_model = df_suggo_aqui.iloc[i, 1]

    name = f"df_{surrogate_model}_{acq_model}"

    df_temp = df_kombination_function(
        vec_fun,
        vec_dim,
        surrogate_model,
        acq_model,
        n_sample_init,
        iteration
    )

    dfs[name] = df_temp


df_gesamt = pd.concat(dfs.values(), ignore_index=True)



df_gesamt_iterationen = (
    df_gesamt
    .loc[df_gesamt.index.repeat(n_iterations)]
    .reset_index(drop=True)
)

df_gesamt_iterationen["iteration"] = (
    list(range(1, n_iterations + 1)) * len(df_gesamt)
)

df_gesamt.to_excel("df_gesamt.xlsx", index=False)
df_gesamt_iterationen.to_excel("df_gesamt_iteration.xlsx", index=False)


display(df_gesamt)
display(df_gesamt_iterationen)


