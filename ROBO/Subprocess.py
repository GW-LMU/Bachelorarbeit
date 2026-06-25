import pandas as pd
import itertools
import numpy as np
from IPython.display import display

########### Laden der Daten ##############
#df_gesamt = pd.read_excel(r"C:\Users\gabri\Download - ICH\Bachelorarbeit\DF_GESAMT.xlsx")
#df_gesamt_iteration = pd.read_excel(r"C:\Users\gabri\Download - ICH\Bachelorarbeit\df_gesamt_iteration.xlsx")



#########
# [1,2,5,10,25,50,100]
n_sample_init =[1,2,5]


# Anzahl der Wiedrholten Verusche

n_sample_stat = 10

# Anzahl der Iteration die in einen Prozess geben bestimmter Paramter berehcnet wird
n_iteration = [20]
iteration = 20

# Die Folgenden Funktion stehen zu Auswahl und können abgefragt werdern 
# [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
vec_fun = [1,2,3]

# Dimensionen an denen der Algorithmus getetest wird
# [1,2,5,10, 15, 25, 50]
vec_dim = [1,2,5]


vec_var = ["GP", "IGP"]
acq = ["UP", "EP", "NL"]
#vec_sample = list(range(1, n_sample + 1 ))
#vec_iter = list(range(1, iteration + 1 ))

df_suggo_aqui = pd.DataFrame({
    "Surrogate_Model": ["GausianProzess", "GausianProzess", "GausianProzess", "Imprecise_GausianProzess_Rodemann", "AIRBO", "AIRBO", "STABLEOPT","DRBO", "Relevance Pursuit"],
    "Acquisitions_Model": ["LCB", "UCB", "EI", "GLCB", "UCB","EI","UCB", "distributionally robust UCB-Akquisition", "qLogNEI"]
})

##########################################################################################
##########################################################################################
##########################################################################################


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
        n_iteration
    )

    dfs[name] = df_temp


df_gesamt = pd.concat(dfs.values(), ignore_index=True)



df_gesamt_iteration = (
    df_gesamt
    .loc[df_gesamt.index.repeat(iteration)]
    .reset_index(drop=True)
)

df_gesamt_iteration["iteration"] = (
    list(range(1, iteration + 1)) * len(df_gesamt)
)

df_gesamt.to_excel("df_gesamt.xlsx", index=False)
df_gesamt_iteration.to_excel("df_gesamt_iteration.xlsx", index=False)


display(df_gesamt)
display(df_gesamt_iteration)


##########################################################################################
##########################################################################################
##########################################################################################


########### Seeds für Statistische Unsicherheit #############

seeds = [48291, 730184, 15937, 904622, 318450, 67219, 845301, 290776, 513908, 76402,
         991245, 406833, 128594, 557019, 683741, 230465, 719008, 364952, 875116, 492607,
         105938, 638214, 781590, 249731, 930856, 572084, 316729, 864015, 451298, 697430,
         82476, 209583, 746901, 385214, 918673, 540127, 167892, 802436, 624019, 973251,
         438706, 715894, 296318, 851027, 604735, 139486, 768250, 325971, 947608, 512364]


########### Grid Boundies #############

minus_area = -1000
plus_area = 1000


##########################################################################################
########### Tenssor für Initzial Sample #############
##########################################################################################

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
    print("------------------")
    print("Typ:", type(tensor))
    print("Shape:", tensor.shape)
    print("Anzahl Achsen / Dimensionen:", tensor.ndim)
    print("Anzahl aller Werte:", tensor.size)
    print("Datentyp:", tensor.dtype)
    print("Minimum:", tensor.min())
    print("Maximum:", tensor.max())
    print("Mittelwert:", tensor.mean())
    print("Standardabweichung:", tensor.std())

tensor = tensor_generator_1(vec_dim, n_sample_init,n_sample_stat, seeds)






##########################################################################################
######### Zusatz Funktionen ###########
##########################################################################################


def choose_model(model):
    models = {
        "GausianProzess": None,  # TODO: durch echte Klasse/Funktion ersetzen
        "Imprecise_GausianProzess_Rodemann": None,
        "AIRBO": None,
        "STABLEOPT": None,
        "DRBO": None,
        "Relevance Pursuit": None,
    }

    if model not in models:
        raise ValueError(f"Unbekanntes Surrogate-Modell: {model}")

    return models[model]


def choose_acquisition(acquisition):
    acquisitions = {
        "LCB": None,  # TODO: durch echte Klasse/Funktion ersetzen
        "UCB": None,
        "EI": None,
        "GLCB": None,
        "distributionally robust UCB-Akquisition": None,
        "qLogNEI": None,
    }

    if acquisition not in acquisitions:
        raise ValueError(f"Unbekannte Acquisition-Funktion: {acquisition}")

    return acquisitions[acquisition]
     
def berechne_iteration(
    func,
    dim,
    sample,
    surrogate_model,
    acquisition_func,
    initial_sample_size,
    iteration
):
    sample_value = float(np.mean(sample))

    return (
        func * 10000
        + dim * 1000
        + initial_sample_size * 100
        + sample_value
        + iteration
    )



########### Erstellen Sub - Dataframe ##############


########### Erstellen Main - Dataframe ##############

for i in range(1, n_sample_stat +1):
    df_gesamt_iteration[f"Var_{i}"] = None

#df_gesamt_iteration.to_excel("df_gesamt_iteration.xlsx", index=False)

########### Subprozess ##############

parameter_cols = [
    "Funktion",
    "Dimension",
    "Surrogate_Model",
    "Acquisitions_Model",
    "Sample_Size_Initial",
    "Iteration"
]

for param_idx, param_row in df_gesamt.iterrows():

    ### Parameter aus df_gesamt laden ###
    func = param_row["Funktion"]
    dim = param_row["Dimension"]
    surrogate_model = param_row["Surrogate_Model"]
    acquisition_func = param_row["Acquisitions_Model"]
    initial_sample_size = param_row["Sample_Size_Initial"]
    max_iteration = int(param_row["Iteration"])

    ### Modell und Acquisition auswählen ###
    current_surrogate_model = choose_model(surrogate_model)
    current_acquisition_func = choose_acquisition(acquisition_func)

    ### passende Zeilen in df_gesamt_iteration finden ###
    mask_parameter = (
        (df_gesamt_iteration["Funktion"] == func) &
        (df_gesamt_iteration["Dimension"] == dim) &
        (df_gesamt_iteration["Surrogate_Model"] == surrogate_model) &
        (df_gesamt_iteration["Acquisitions_Model"] == acquisition_func) &
        (df_gesamt_iteration["Sample_Size_Initial"] == initial_sample_size) &
        (df_gesamt_iteration["Iteration"] == max_iteration)
    )

    ### Tensor-Blöcke durchgehen: Var_1 bis Var_40 ###
    for n in range(n_sample_stat):

        # wichtig: nicht tensor[n-1], weil bei n=0 sonst der letzte Block genommen wird
        tensor_block = tensor[n]

        var_col = f"Var_{n + 1}"

        ### Iterationen durchgehen ###
        for j in range(1, max_iteration + 1):

            # Hier kommt deine eigentliche Berechnung rein
            # Beispiel-Platzhalter:
            ergebnis = berechne_iteration(
                func=func,
                dim=dim,
                sample=tensor_block,
                surrogate_model=current_surrogate_model,
                acquisition_func=current_acquisition_func,
                initial_sample_size=initial_sample_size,
                iteration=j
            )

            # passende Zeile: gleiche Parameter + konkrete iteration
            mask_iteration = mask_parameter & (df_gesamt_iteration["iteration"] == j)

            # Ergebnis speichern
            df_gesamt_iteration.loc[mask_iteration, var_col] = ergebnis

    
df_gesamt_iteration.to_excel("df_gesamt_iteration.xlsx", index=False)  







