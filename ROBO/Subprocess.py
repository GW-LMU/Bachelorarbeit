import pandas as pd
import itertools
import numpy as np
import openpyxl
from IPython.display import display
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from coco_benchmark import evaluate_bbob_function
from coco_benchmark import get_bbob_optimum
from coco_benchmark import compute_regret
from coco_benchmark import waehle_und_berechne_variante
from Standard_PY import run_gausian_prozess
import cocoex
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch

########### Laden der Daten ##############
#df_gesamt = pd.read_excel(r"C:\Users\gabri\Download - ICH\Bachelorarbeit\DF_GESAMT.xlsx")
#df_gesamt_iteration = pd.read_excel(r"C:\Users\gabri\Download - ICH\Bachelorarbeit\df_gesamt_iteration.xlsx")



#########
# [1,2,5,10,25,50,100]
n_sample_init =[1,
                2,
                #5,
                #20
                ]


# Anzahl der Wiedrholten Verusche

n_sample_stat = 5

# Anzahl der Iteration die in einen Prozess geben bestimmter Paramter berehcnet wird
n_iteration = [10]
iteration = 10

# Die Folgenden Funktion stehen zu Auswahl und können abgefragt werdern 
# [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
vec_fun = [1,2]

# Dimensionen an denen der Algorithmus getetest wird
# [1,2,5,10, 15, 25, 50]
vec_dim = [2,
           5,
           #10
           ]


vec_var = ["GP", "IGP"]
acq = ["UP", "EP", "NL"]
#vec_sample = list(range(1, n_sample + 1 ))
#vec_iter = list(range(1, iteration + 1 ))


speicherpfad = r"C:\Users\gabri\Download - ICH\Bachelorarbeit\Bachelorarbeit\ROBO"

df_suggo_aqui = pd.DataFrame({
    "Surrogate_Model": ["GausianProzess", 
                        #"GausianProzess", 
                        #"GausianProzess", 
                        #"Imprecise_GausianProzess_Rodemann" 
                        #"AIRBO", 
                        #"AIRBO", 
                        #"STABLEOPT",
                        #"DRBO", 
                        #"Relevance Pursuit"
                        ],

    "Acquisitions_Model": ["LCB", 
                           #"UCB", 
                           #"EI", 
                           #"GLCB", 
                           #"UCB",
                           #"EI",
                           #"UCB", 
                           #"distributionally robust UCB-Akquisition", 
                           #"qLogNEI"
                           ]
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

for i in range(len(df_suggo_aqui)):                     # Erstell pro Suggorat und aquisiktiosnfunktion Datafram
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

df_gesamt = pd.concat(dfs.values(), ignore_index=True) # Schreibt alle Dataframs zusammen 

### Schreibt zu df_gesamt die Iterationszeilen 

df_gesamt_iteration = (
    df_gesamt
    .loc[df_gesamt.index.repeat(iteration)]
    .reset_index(drop=True)
)

df_gesamt_iteration["iteration"] = (
    list(range(1, iteration + 1)) * len(df_gesamt)
)

# Dataframes werden gespeichert 

df_gesamt.to_excel("df_gesamt.xlsx", index=False)
df_gesamt_iteration.to_excel("df_gesamt_iteration.xlsx", index=False)

#  Datadrames werden Testweise angezeigt 
display(df_gesamt)
display(df_gesamt_iteration)


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
tensor_2d = tensor[1]
pd.DataFrame(tensor_2d).to_excel(
    "tensor_block_1.xlsx",
    index=False,
    header=False
)


def add_noise_stratified(
    tensor: np.ndarray,
    prozent: float,
    sigma: float,
    achse: int = 0,
    seed_auswahl: int = 42,
    seed_rauschen: int = 123,
    mu: float = 0.0,
) -> np.ndarray:
    """
    Variante: in JEDER Scheibe entlang einer Achse wird derselbe Prozentsatz
    verändert - garantiert räumlich gleichmäßige Streuung.
    """
    assert tensor.ndim == 3, "Erwartet einen 3D-Tensor"
    assert 0 < prozent <= 1

    ergebnis = tensor.copy()
    rng_auswahl = np.random.default_rng(seed_auswahl)
    rng_rauschen = np.random.default_rng(seed_rauschen)

    n_schichten = tensor.shape[achse]
    for i in range(n_schichten):
        schicht_idx = [slice(None)] * 3
        schicht_idx[achse] = i
        schicht = ergebnis[tuple(schicht_idx)]

        n_total = schicht.size
        n_auswahl = int(round(n_total * prozent))
        flat_indices = rng_auswahl.choice(n_total, size=n_auswahl, replace=False)
        idx = np.unravel_index(flat_indices, schicht.shape)

        rauschen = rng_rauschen.normal(loc=mu, scale=sigma, size=n_auswahl)
        schicht[idx] = schicht[idx] + rauschen

    return ergebnis




tensor_noise = add_noise_stratified(tensor, 1, 4, 1)
tensor_noise_2d = tensor_noise[1]
pd.DataFrame(tensor_noise_2d).to_excel(
    "tensor_block_1_noise.xlsx",
    index=False,
    header=False
)


# def plot_diff_histogram(
#     tensor1: np.ndarray,
#     tensor2: np.ndarray,
#     bins: int = 50,
#     title: str = "Histogramm der Differenz",
#     speicherpfad: str | None = None,
# ):
#     assert tensor1.shape == tensor2.shape, "Tensoren müssen dieselbe Form haben"

#     # --- Differenz berechnen ---
#     diff = tensor1 - tensor2
#     werte = diff.flatten()
#     n_werte = werte.size

#     # --- Statistik für die Normalverteilung schätzen ---
#     mu = werte.mean()
#     sigma = werte.std()

#     # --- Histogramm plotten ---
#     fig, ax = plt.subplots(figsize=(8, 5))
#     counts, bin_edges, _ = ax.hist(
#         werte, bins=bins, color="steelblue", edgecolor="black",
#         alpha=0.7, label="Häufigkeit der Differenzwerte"
#     )

#     # --- Normalverteilung passend zur Histogramm-Skala (Counts) überlagern ---
#     bin_width = bin_edges[1] - bin_edges[0]
#     x = np.linspace(werte.min(), werte.max(), 500)
#     pdf = norm.pdf(x, loc=mu, scale=sigma)
#     pdf_skaliert = pdf * n_werte * bin_width  # von Dichte auf Counts skaliert

#     ax.plot(x, pdf_skaliert, color="firebrick", linewidth=2,
#             label=f"N(μ={mu:.3f}, σ={sigma:.3f})")

#     ax.set_xlabel("Differenzwert")
#     ax.set_ylabel("Anzahl")
#     ax.set_title(f"{title}\n(Anzahl Werte: N = {n_werte})")
#     ax.legend()
#     fig.tight_layout()

#     if speicherpfad:
#         fig.savefig(speicherpfad, dpi=150)
#         print(f"Plot gespeichert unter: {speicherpfad}")
#     else:
#         plt.show()

#     plt.close(fig)
#     return diff




#diff = plot_diff_histogram(
#        tensor, tensor_noise,
#        bins=40,
#        title="Beispiel: Tensor A - Tensor B",
#        speicherpfad="diff_histogram.png",  # oder None für direkte Anzeige
#    )
#print("Form der Differenz:", diff.shape)




####################################################################
########### Erstellen Main - Dataframe  mit Testgröße ##############
####################################################################

for i in range(1, n_sample_stat +1):
    df_gesamt_iteration[f"Var_{i}"] = None

#df_gesamt_iteration.to_excel("df_gesamt_iteration.xlsx", index=False)

####################################################################
####################### Main Algorithmus  ##########################
####################################################################

parameter_cols = [
    "Funktion",
    "Dimension",
    "Surrogate_Model",
    "Acquisitions_Model",
    "Sample_Size_Initial",
    "Iteration"
]

suite = cocoex.Suite("bbob", "", "")

for param_idx, param_row in df_gesamt.iterrows():

    ### Parameter aus df_gesamt laden ###
    funktion = param_row["Funktion"]
    dimension = param_row["Dimension"]
    surrogate_model = param_row["Surrogate_Model"]
    acquisition_funktion = param_row["Acquisitions_Model"]
    initial_sample_size = param_row["Sample_Size_Initial"]
    max_iteration = int(param_row["Iteration"])

 

    current_function_dim = suite.get_problem_by_function_dimension_instance(funktion, dimension, 1 )

    ### passende Zeilen in df_gesamt_iteration finden ###
    mask_parameter = (
        (df_gesamt_iteration["Funktion"] == funktion) &
        (df_gesamt_iteration["Dimension"] == dimension) &
        (df_gesamt_iteration["Surrogate_Model"] == surrogate_model) &
        (df_gesamt_iteration["Acquisitions_Model"] == acquisition_funktion) &
        (df_gesamt_iteration["Sample_Size_Initial"] == initial_sample_size) &
        (df_gesamt_iteration["Iteration"] == max_iteration)
    )

    # Holt sich das Optimum für gegebene Paramter Cofuguatation

    x_opt, f_opt = get_bbob_optimum(funktion, dimension, instance=1, suite=suite, maximize = False)

    ### Tensor-Blöcke durchgehen: Var_1 bis Var_n ###
    for n in range(n_sample_stat):


        # Sampels vorberieten und X und Y Werte berehcen 

        tensor_block = tensor_noise[n]
        tensor_block_torch = torch.as_tensor(tensor_block, dtype=torch.double)

        var_col = f"Var_{n + 1}"

        # Berechung Inital Sample
        # X-Werte als NumPy ziehen
        train_X_np = tensor_block[:initial_sample_size, :dimension]

        # Y-Werte mit der Funktion berechnen (pro Zeile einzeln, da evaluate_bbob_function
        # nur einen einzelnen Punkt auf einmal auswertet)
        train_Y_np = np.array([
            evaluate_bbob_function(x, funktion, dimension) for x in train_X_np
        ])

        train_X = torch.as_tensor(train_X_np, dtype=torch.double)
        train_Y = torch.as_tensor(train_Y_np, dtype=torch.double).unsqueeze(-1)

        ### Iterationen durchgehen ###
        for j in range(1, max_iteration + 1):

            ####################

            next_x, acq_value = waehle_und_berechne_variante(surrogate_model, acquisition_funktion, train_X, train_Y, dimension,
                                                    lower_bound=minus_area, upper_bound=plus_area)

            ###Punkt hinzufügen#####

            # next_point von Torch zu NumPy
            next_point_np = next_x.numpy()

            # neuen Y-Wert für den vorgeschlagenen Punkt berechnen
            next_value_np = evaluate_bbob_function(next_point_np, funktion, dimension)

            # neuen Punkt an train_X_np / train_Y_np anhängen
            train_X_np = np.vstack([train_X_np, next_point_np.reshape(1, -1)])
            train_Y_np = np.append(train_Y_np, next_value_np)

            # Torch-Tensoren nachziehen, damit das GP im nächsten Schritt
            # auch die neu vorgeschlagenen Punkte sieht
            train_X = torch.as_tensor(train_X_np, dtype=torch.double)
            train_Y = torch.as_tensor(train_Y_np, dtype=torch.double).unsqueeze(-1)


            ### eta-reget berechen 
                        
                        
            best_so_far, simple_regret, immediate_regret, x_distances = compute_regret(train_X_np, train_Y_np, x_opt, f_opt)

            ergebnis = simple_regret[-1]


            print(f"Funktion {funktion}, Dimension {dimension}: Modell:={surrogate_model}, Aqui={acquisition_funktion},Tensorblock={n} Iteration={j}")
            #################################################################

            # passende Zeile: gleiche Parameter + konkrete iteration
            mask_iteration = mask_parameter & (df_gesamt_iteration["iteration"] == j)

            # Ergebnis speichern
            df_gesamt_iteration.loc[mask_iteration, var_col] = ergebnis

    
df_gesamt_iteration.to_excel("df_gesamt_iteration.xlsx", index=False)  







#ergebnis = berechne_iteration(
#                func=func,
#                dim=dim,
#                sample=tensor_block,
#                surrogate_model=current_surrogate_model,
#                acquisition_func=current_acquisition_func,
#                initial_sample_size=initial_sample_size,
#                iteration=j
#            )