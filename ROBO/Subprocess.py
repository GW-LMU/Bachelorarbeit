import pandas as pd
import itertools
import numpy as np
import openpyxl
from IPython.display import display
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
#import cocoex
#from botorch.models import SingleTaskGP
#from botorch.fit import fit_gpytorch_mll
#from botorch.acquisition.analytic import LogExpectedImprovement
#from botorch.optim import optimize_acqf
#from gpytorch.mlls import ExactMarginalLogLikelihood

########### Laden der Daten ##############
#df_gesamt = pd.read_excel(r"C:\Users\gabri\Download - ICH\Bachelorarbeit\DF_GESAMT.xlsx")
#df_gesamt_iteration = pd.read_excel(r"C:\Users\gabri\Download - ICH\Bachelorarbeit\df_gesamt_iteration.xlsx")



#########
# [1,2,5,10,25,50,100]
n_sample_init =[1,2,5,20]


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
vec_dim = [1,2,5,10]


vec_var = ["GP", "IGP"]
acq = ["UP", "EP", "NL"]
#vec_sample = list(range(1, n_sample + 1 ))
#vec_iter = list(range(1, iteration + 1 ))

df_suggo_aqui = pd.DataFrame({
    "Surrogate_Model": ["GausianProzess", "GausianProzess", "GausianProzess", "Imprecise_GausianProzess_Rodemann", "AIRBO", "AIRBO", "STABLEOPT","DRBO", "Relevance Pursuit"],
    "Acquisitions_Model": ["LCB", "UCB", "EI", "GLCB", "UCB","EI","UCB", "distributionally robust UCB-Akquisition", "qLogNEI"]
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


def plot_diff_histogram(
    tensor1: np.ndarray,
    tensor2: np.ndarray,
    bins: int = 50,
    title: str = "Histogramm der Differenz",
    speicherpfad: str | None = None,
):
    assert tensor1.shape == tensor2.shape, "Tensoren müssen dieselbe Form haben"

    # --- Differenz berechnen ---
    diff = tensor1 - tensor2
    werte = diff.flatten()
    n_werte = werte.size

    # --- Statistik für die Normalverteilung schätzen ---
    mu = werte.mean()
    sigma = werte.std()

    # --- Histogramm plotten ---
    fig, ax = plt.subplots(figsize=(8, 5))
    counts, bin_edges, _ = ax.hist(
        werte, bins=bins, color="steelblue", edgecolor="black",
        alpha=0.7, label="Häufigkeit der Differenzwerte"
    )

    # --- Normalverteilung passend zur Histogramm-Skala (Counts) überlagern ---
    bin_width = bin_edges[1] - bin_edges[0]
    x = np.linspace(werte.min(), werte.max(), 500)
    pdf = norm.pdf(x, loc=mu, scale=sigma)
    pdf_skaliert = pdf * n_werte * bin_width  # von Dichte auf Counts skaliert

    ax.plot(x, pdf_skaliert, color="firebrick", linewidth=2,
            label=f"N(μ={mu:.3f}, σ={sigma:.3f})")

    ax.set_xlabel("Differenzwert")
    ax.set_ylabel("Anzahl")
    ax.set_title(f"{title}\n(Anzahl Werte: N = {n_werte})")
    ax.legend()
    fig.tight_layout()

    if speicherpfad:
        fig.savefig(speicherpfad, dpi=150)
        print(f"Plot gespeichert unter: {speicherpfad}")
    else:
        plt.show()

    plt.close(fig)
    return diff




diff = plot_diff_histogram(
        tensor, tensor_noise,
        bins=40,
        title="Beispiel: Tensor A - Tensor B",
        speicherpfad="diff_histogram.png",  # oder None für direkte Anzeige
    )
print("Form der Differenz:", diff.shape)


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
     

# Kleine Testfunktion für Speicherfunktion der Datframes 
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


def berechne_train_Y(train_X_np, current_function_dim, minimize=False):
    train_Y_np = np.array(
        [
            [-current_function_dim(x) if minimize else current_function_dim(x)]
            for x in train_X_np
        ],
        dtype=np.float64
    )

    return train_Y_np

def test_GP(train_X, train_Y, optimal_Y):
    # 1) Gaussian Process Modell
    model = SingleTaskGP(train_X=train_X, train_Y=train_Y)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # 2) Aktuell bester beobachteter Funktionswert
    best_f = train_Y.max()

    # 3) EI-Akquisitionsfunktion
    ei = LogExpectedImprovement(model=model, best_f=best_f)

    # 4) Nächsten Punkt finden
    bounds = torch.tensor([[0.0], [1.0]], dtype=train_X.dtype, device=train_X.device)

    candidate, acq_value = optimize_acqf(
        acq_function=ei,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=128,
    )

    print("Nächster Punkt:", candidate)
    print("Akquisitionswert:", acq_value)

    # 5) Zielfunktion am neuen Punkt auswerten
    new_Y = berechne_train_Y(candidate, current_function_dim, minimize=False)

    if new_Y.ndim == 1:
        new_Y = new_Y.reshape(-1, 1)

    # 6) Daten erweitern
    train_X = torch.cat([train_X, candidate], dim=0)
    train_Y = torch.cat([train_Y, new_Y], dim=0)

    # 7) Regret berechnen
    best_observed_Y = train_Y.max()
    regret = optimal_Y - best_observed_Y

    print("Bester bisheriger Wert:", best_observed_Y.item())
    print("Optimum:", optimal_Y)
    print("Regret:", regret.item())

    return train_X, train_Y, regret
            



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
    func = param_row["Funktion"]
    dim = param_row["Dimension"]
    surrogate_model = param_row["Surrogate_Model"]
    acquisition_func = param_row["Acquisitions_Model"]
    initial_sample_size = param_row["Sample_Size_Initial"]
    max_iteration = int(param_row["Iteration"])

    ### Modell und Acquisition auswählen ###
    current_surrogate_model = choose_model(surrogate_model)
    current_acquisition_func = choose_acquisition(acquisition_func)

    current_function_dim = suite.get_problem_by_function_dimension_instance(func, dim, 1 )

    ### passende Zeilen in df_gesamt_iteration finden ###
    mask_parameter = (
        (df_gesamt_iteration["Funktion"] == func) &
        (df_gesamt_iteration["Dimension"] == dim) &
        (df_gesamt_iteration["Surrogate_Model"] == surrogate_model) &
        (df_gesamt_iteration["Acquisitions_Model"] == acquisition_func) &
        (df_gesamt_iteration["Sample_Size_Initial"] == initial_sample_size) &
        (df_gesamt_iteration["Iteration"] == max_iteration)
    )

    ### Tensor-Blöcke durchgehen: Var_1 bis Var_n ###
    for n in range(n_sample_stat):

        tensor_block = tensor[n]
        tensor_block_torch = torch.as_tensor(tensor_block, dtype=torch.double)

        var_col = f"Var_{n + 1}"

        # Berechung Inital Sample 
        # X-Werte als NumPy ziehen
        train_X_np = tensor_block[:initial_sample_size, :dim]

        # Y-Werte mit der Funktion berechnen
        train_Y_np = berechne_train_Y(train_X_np, current_function_dim, minimize=False)
        
        train_X = torch.as_tensor(train_X_np, dtype=torch.double)
        train_Y = torch.as_tensor(train_Y_np, dtype=torch.double)

        ### Iterationen durchgehen ###
        for j in range(1, max_iteration + 1):

            #################################################################
            ergebnis = test_GP(train_X, train_Y )














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