import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.special import expit  # Sigmoid
import random
import uncertainty_aware_bo
from uncertainty_aware_bo import BayesOptimizer


def imprecise_gp(train_X, train_Y, kernel, c, length_scale, outputscale):

    # Falls train_x und train_y schon existieren:
    train_X = np.asarray(train_X).ravel()
    train_Y = np.asarray(train_Y).ravel()
    


    ## Inizalisieren der BaysianOptimizer 

    bo = BayesOptimizer(x_obs=train_X,
                        y_obs=train_Y,
                        kernel=kernel,
                        c=c,
                        length_scale=length_scale,
                        outputscale=outputscale)
    
    # Kandidatenbereich definieren
    x_min = train_X.min()
    x_max = train_Y.max()

    x_candidates = np.linspace(x_min, x_max, 500)

    # Surrogatemodell berechnen
    means = []
    variances = []

    for x in x_candidates:
        mu, var = bo.predict_posterior(x)
        means.append(mu)
        variances.append(var)

    means = np.array(means)
    variances = np.maximum(np.array(variances), 0)

    # Acquisition Function berechnen
    acquisition_values = np.array([bo.PROBO(x) for x in x_candidates])

    # Bestes nächstes x auswählen
    next_x = x_candidates[np.argmax(acquisition_values)]

    





