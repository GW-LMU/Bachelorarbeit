"""Zentrale Konfiguration fuer die verteilte BO-Benchmark-Pipeline (ROBO_Benchmark).

Wird von prepare_tasks.py (Erzeugung der Task-Liste) und run_instance.py
(Ausfuehrung auf den einzelnen LRZ-Instanzen) verwendet.
"""

from pathlib import Path

# --- Verzeichnisse -------------------------------------------------------
# config.py liegt in Pre_Processing/, ROBO_Benchmark-Wurzel (mit data/) ist
# daher eine Ebene hoeher.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# --- BO-Parameter (Kreuzprodukt ergibt die Kombinationen) ----------------
# TEST-WERTE fuer einen lokalen Laufzeit-Messlauf auf diesem PC - fuer den
# echten Lauf auf dem LRZ hier wieder auf die vollen Listen hochskalieren.
FUNKTIONEN = [1]    
# [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]          # BBOB-Funktionen 1-24
DIMENSIONEN = [2]
# [2, 3, 5, 10, 20, 40]                   # Suchraum-Dimensionen

# je (Surrogatmodell, Akquisitionsfunktion)-Paar eine Zeile.
# Implementiert (core_bo.py -> select_and_run_variant()), analog zu
# waehle_und_berechne_variante() in ROBO/coco_benchmark.py:
#   "GausianProzess"                    -> "EI" | "UCB" | "LCB"
#   "Imprecise_GausianProzess_Rodemann" -> "LCB" | "GLCB"   (nur dim=1, siehe imprecise_gp.py)
#   "Relevance Pursuit"                 -> "EI" | "UCB" | "LCB"
#   "STABLEOPT"                         -> nur "UCB"        (siehe stable_opt.py)
#   "DRBO"                              -> nur "distributionally robust UCB-Akquisition"
#                                          (letzte Spalte von dim = Kontextvariable, siehe drbo.py)
#   "AIRBO"                             -> nur "UCB" (siehe airbo.py)
#                                          ACHTUNG: aktuell quadratische Laufzeit
#                                          (~5s/Iteration schon bei 50 Beobachtungen) -
#                                          vor grossen Laeufen n_x_candidates/
#                                          n_cloud_samples/sub_samp_size in airbo.py
#                                          reduzieren oder Kernel vektorisieren, siehe README.md.
SURROGATE_AKQUISITION_PAARE = [
    ("GausianProzess", "EI"),
    ("Imprecise_GausianProzess_Rodemann", "GLCB"),
    #("Relevance Pursuit", "EI"),
    #("STABLEOPT", "UCB"),
    #("DRBO", "distributionally robust UCB-Akquisition"),
    #("AIRBO", "UCB"),
]
N_SAMPLE_INIT = [2]                 # Groesse des initialen Samples
#               [2,5,10,20]   
N_ITERATION = [100]                 # Anzahl BO-Iterationen pro Lauf (Messlauf: Zielwert)

# --- Statistische Wiederholungen (Tensorbloecke) -------------------------
N_SAMPLE_STAT = 10                  # Messlauf: klein gehalten, nur fuer Zeitmessung
SEEDS = SEEDS = [
    48291, 730184, 15937, 904622, 318450, 67219, 845301, 290776, 513908, 76402,
    #5898, 130626, 186734, 204177, 213143, 227336, 229135, 248246, 266314, 316758,
    #332814, 391109, 441839, 567503, 598308, 613474, 642683, 667237, 672755, 676254,
    #699214, 706155, 732433, 788646, 797364, 839424, 914911, 941801, 948880, 988455,
]

# --- Suchraum --------------------------------------------------------------
# BBOB/COCO-Standard-Region-of-Interest: das bekannte Optimum jeder der
# 24 Funktionen liegt garantiert innerhalb von [-5, 5]^Dimension.
MINUS_AREA = -5
PLUS_AREA = 5

# --- Schaetzung des Maximums je (Funktion, Dimension) fuer den Regret ------
# Ziel ist Maximierung von f(x); cocoex kennt nur das Minimum, daher wird
# das Maximum per Zufallsstichprobe angenaehert (einmal pro Funktion/
# Dimension, nicht pro Task - siehe prepare_tasks.py).
N_OPTIMUM_ESTIMATE_SAMPLES = 2000
OPTIMUM_ESTIMATE_SEED = 999

# --- Rauschen auf den initialen Samples (analog add_noise_stratified) ------
NOISE_ANTEIL = 1.0
NOISE_SIGMA = 0.2                   # ~2% der Suchraumbreite bei [-5,5]
NOISE_SEED_AUSWAHL = 42
NOISE_SEED_RAUSCHEN = 123

# --- Verteilung auf die LRZ-Instanzen ---------------------------------------
# Fuer den lokalen Test auf diesem PC (12 Kerne verfuegbar): 1 "Instanz",
# 4 Worker-Prozesse. Fuer den echten LRZ-Lauf: N_INSTANZEN=4, WORKERS=10.
N_INSTANZEN = 1
WORKERS_PRO_INSTANZ = 2
