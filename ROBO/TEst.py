import pandas as pd
import itertools
import numpy as np
import openpyxl
from IPython.display import display
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from uncertainty_aware_bo import BayesOptimizer
import cocoex
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

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


# Nachdem der Tensor mit Noise versehen worden ist Übergabe an BO ALGo




### Standard Ansatz ###

def run_bayes_opt_botorch_nd(x_obs, y_obs, n_iterations: int, f, dim: int,
                              lower_bound=-5.0, upper_bound=5.0,
                              af="EI", minimize=True, **af_kwargs):
    """
    Führt Standard-Bayesian-Optimization mit BoTorch für mehrdimensionale
    (COCO-)Zielfunktionen durch, basierend auf einer vorhandenen Stichprobe.

    Parameter
    ---------
    x_obs : array-like, shape [n, d]
        Bereits beobachtete Parametervektoren.
    y_obs : array-like, shape [n]
        Dazugehörige beobachtete Zielwerte (Skalare).
    n_iterations : int
        Anzahl der Optimierungsschritte (z.B. 6).
    f : callable
        Zielfunktion, z.B. aus cocoex. Nimmt einen Vektor der Länge `dim`
        entgegen und gibt einen Skalar zurück: f(x) -> float
    dim : int
        Dimensionalität des Suchraums (muss zur COCO-Funktion passen).
    lower_bound, upper_bound : float oder array-like
        Grenzen des Suchraums, entweder skalar (gilt für alle Dimensionen)
        oder als Vektor der Länge `dim`.
    af : str
        "EI" (Expected Improvement) oder "UCB".
    minimize : bool
        True, falls f minimiert werden soll (COCO-Standardfall).
        BoTorch maximiert intern -> wir arbeiten daher mit -f.
    af_kwargs : dict
        Zusätzliche Parameter für die Akquisitionsfunktion (z.B. beta für UCB).

    Returns
    -------
    x_obs, y_obs : np.ndarray
        Alle beobachteten Punkte (shape [n, d]) und Zielwerte (shape [n]),
        inkl. der neu hinzugefügten. y_obs bleibt in Original-Skala
        (nicht negiert), auch wenn intern minimiert wurde.
    """

    x_obs = np.atleast_2d(np.array(x_obs, dtype=float))
    y_obs = np.array(y_obs, dtype=float).reshape(-1)

    assert x_obs.shape[1] == dim, f"x_obs hat {x_obs.shape[1]} Spalten, erwartet dim={dim}"
    assert x_obs.shape[0] == y_obs.shape[0], "x_obs und y_obs müssen gleich viele Punkte haben"

    dtype = torch.double

    lb = np.full(dim, lower_bound) if np.isscalar(lower_bound) else np.array(lower_bound, dtype=float)
    ub = np.full(dim, upper_bound) if np.isscalar(upper_bound) else np.array(upper_bound, dtype=float)
    bounds = torch.tensor(np.vstack([lb, ub]), dtype=dtype)  # shape [2, d]

    sign = -1.0 if minimize else 1.0  # BoTorch maximiert, COCO wird meist minimiert

    for i in range(n_iterations):
        # ---- Daten als Torch-Tensoren vorbereiten ----
        train_X = torch.tensor(x_obs, dtype=dtype)                       # [n, d]
        train_Y = torch.tensor(sign * y_obs, dtype=dtype).unsqueeze(-1)  # [n, 1], intern maximiert

        # ---- GP-Modell fitten ----
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # ---- Akquisitionsfunktion wählen ----
        if af == "EI":
            best_f = train_Y.max()
            acq_func = ExpectedImprovement(model=model, best_f=best_f)
        elif af == "UCB":
            beta = af_kwargs.get("beta", 2.0)
            acq_func = UpperConfidenceBound(model=model, beta=beta)
        else:
            raise ValueError(f"Unbekannte Akquisitionsfunktion: {af}")

        # ---- nächsten Punkt vorschlagen ----
        candidate, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,
        )
        next_x = candidate.squeeze(0).numpy()  # shape [d]

        # ---- Zielfunktion auswerten ----
        # Hier wird f übergeben, z.B. deine cocoex-Funktion:
        #   next_y = f(next_x)
        next_y = f(next_x)
        # ---------------------------------

        # ---- neue Beobachtung anhängen ----
        x_obs = np.vstack([x_obs, next_x])
        y_obs = np.append(y_obs, next_y)

        print(f"Iteration {i+1}/{n_iterations}: x = {np.round(next_x, 4)}, y = {next_y:.4f}")

    return x_obs, y_obs

### Imprcsie Baysian Ansatz ###

def run_bayes_opt(x_obs, y_obs, n_iterations: int, f, bound=5.0, af="PROBO", **bo_kwargs):

    """
    Führt Bayesian Optimization basierend auf einer bereits vorhandenen
    Stichprobe (x_obs, y_obs) für n_iterations Schritte durch.

    Parameter
    ---------
    x_obs : array-like
        Bereits beobachtete Parameterwerte (Startstichprobe).
    y_obs : array-like
        Dazugehörige beobachtete Zielwerte.
    n_iterations : int
        Anzahl der Optimierungsschritte (z.B. 6).
    f : callable
        Die zu optimierende Zielfunktion, z.B. aus cocoex.
        Wird hier bewusst offen gelassen -> f(x) muss einen Skalar zurückgeben.
    bound : float
        Suchraum-Grenze für die Vorschläge (Parameterwerte in [-bound, bound]).
    af : str
        Name der Akquisitionsfunktion ("PROBO", "UCB", "MEAN", "RAHBO").
    bo_kwargs : dict
        Weitere Parameter für BayesOptimizer (kernel, c, length_scale, outputscale, ...).

    Returns
    -------
    opt.x_obs, opt.y_obs : np.ndarray
        Alle beobachteten Punkte inkl. der neu hinzugefügten.
    """

    x_obs = np.array(x_obs, dtype=float)
    y_obs = np.array(y_obs, dtype=float)

    # Optimizer initialisieren
    opt = BayesOptimizer(x_obs, y_obs, **bo_kwargs)

    # Akquisitionsfunktion setzen
    af_map = {
        "PROBO": opt.PROBO,
        "UCB": opt.UCB,
        "MEAN": opt.MEAN,
        "RAHBO": opt.RAHBO,
    }
    if af not in af_map:
        raise ValueError(f"Unbekannte Akquisitionsfunktion: {af}")
    opt._AF = af_map[af]

    for i in range(n_iterations):
        # nächsten vielversprechenden Punkt vorschlagen
        next_x = opt.select_next_x(bound=bound)

        # ---- Zielfunktion auswerten ----
        # Hier wird f übergeben, z.B. deine cocoex-Funktion:
        #   next_y = f(next_x)
        next_y = f(next_x)
        # ---------------------------------

        # neue Beobachtung anhängen
        opt.x_obs = np.append(opt.x_obs, next_x)
        opt.y_obs = np.append(opt.y_obs, next_y)

        # Modell (C, C_inv, s_k, S_k) mit neuer Beobachtung aktualisieren
        opt.update()

        print(f"Iteration {i+1}/{n_iterations}: x = {next_x:.4f}, y = {next_y:.4f}")

    return opt.x_obs, opt.y_obs


### AIRBO Ansatz ###


##########################################################################################
########### Dispatch: pro Zeile (Surrogat + Akquisition) die richtige Variante fahren ####
##########################################################################################

def run_gausian_prozess(train_X, train_Y, dim, acquisition_func,
                         lower_bound=minus_area, upper_bound=plus_area):
    """
    Führt die komplette Standard-GP-Variante für eine Iteration aus:
    1) SingleTaskGP fitten
    2) passende Akquisitionsfunktion anhand von acquisition_func wählen ("EI", "UCB", "LCB")
    3) nächsten Punkt per optimize_acqf vorschlagen

    train_X : torch.double, shape [n, dim]
    train_Y : torch.double, shape [n, 1] (Maximierungs-Konvention, siehe coco_benchmark.py)
    dim     : int, Dimension des Suchraums

    Rückgabe: next_x (torch.double, shape [dim]), acq_value (torch.double, Skalar)
    """
    # 1) Modell fitten
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # 2) Akquisitionsfunktion abfragen und passend wählen
    if acquisition_func == "EI":
        acqf = LogExpectedImprovement(model, best_f=train_Y.max())
    elif acquisition_func == "UCB":
        acqf = UpperConfidenceBound(model, beta=2.0, maximize=True)
    elif acquisition_func == "LCB":
        acqf = UpperConfidenceBound(model, beta=2.0, maximize=False)
    else:
        raise ValueError(f"Unbekannte Akquisitionsfunktion für GausianProzess: {acquisition_func}")

    # 3) nächsten Punkt vorschlagen
    bounds = torch.tensor(
        [[lower_bound] * dim, [upper_bound] * dim], dtype=torch.double
    )
    next_x, acq_value = optimize_acqf(
        acqf,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
    )

    return next_x.squeeze(0), acq_value


def waehle_und_berechne_variante(surrogate_model, acquisition_func, train_X, train_Y, dim,
                                  lower_bound=minus_area, upper_bound=plus_area):
    """
    Nimmt eine Zeile aus df_suggo_aqui (surrogate_model, acquisition_func) und führt
    genau die dazu passende Variante aus: Surrogatmodell fitten, passende Akquisitions-
    funktion bauen, nächsten Punkt vorschlagen.

    train_X : torch.double, shape [n, dim]
    train_Y : torch.double, shape [n, 1] (Maximierungs-Konvention, siehe coco_benchmark.py)

    Rückgabe: next_x (torch.double, shape [dim]), acq_value (torch.double, Skalar)
    """
    if surrogate_model == "GausianProzess":
        return run_gausian_prozess(train_X, train_Y, dim, acquisition_func,
                                    lower_bound=lower_bound, upper_bound=upper_bound)

    elif surrogate_model == "Imprecise_GausianProzess_Rodemann":
        raise NotImplementedError(
            "Variante 'Imprecise_GausianProzess_Rodemann' ist noch nicht implementiert."
        )

    elif surrogate_model == "AIRBO":
        raise NotImplementedError("Variante 'AIRBO' ist noch nicht implementiert.")

    elif surrogate_model == "STABLEOPT":
        raise NotImplementedError("Variante 'STABLEOPT' ist noch nicht implementiert.")

    elif surrogate_model == "DRBO":
        raise NotImplementedError("Variante 'DRBO' ist noch nicht implementiert.")

    elif surrogate_model == "Relevance Pursuit":
        raise NotImplementedError("Variante 'Relevance Pursuit' ist noch nicht implementiert.")

    else:
        raise ValueError(f"Unbekanntes Surrogatmodell: {surrogate_model}")