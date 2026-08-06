"""Kern des Imprecise-Gaussian-Process-Modells (Rodemann) - schlanke Kopie.

Angepasste/gekuerzte Version von ROBO/uncertainty_aware_bo.py: enthaelt nur
die Teile, die imprecise_gp.py fuer eine BO-Iteration tatsaechlich braucht
(Kernel, BayesOptimizer mit UCB/PROBO-Akquisition). Die Diagnose-/Plot-Methode
`plot_motiv` sowie ungenutzte Akquisitionsvarianten (RAHBO, CredalAF, ...) und
die matplotlib-Abhaengigkeit wurden bewusst entfernt, damit ROBO_NEU ohne
Abhaengigkeit zum alten Ordner auf jede LRZ-Instanz kopiert werden kann.
"""

import numpy as np


def matern(x, y, length_scale=1.0, outputscale=1.0):
    dists = np.abs(np.subtract.outer(x, y))
    sqrt3_dists = np.sqrt(3) * dists / length_scale
    return outputscale * (1.0 + sqrt3_dists) * np.exp(-sqrt3_dists)


def rbf(x, y, length_scale=1.0, outputscale=1.0):
    sqdist = np.subtract.outer(x, y) ** 2
    return outputscale * np.exp(-0.5 * sqdist / length_scale ** 2)


class BayesOptimizer:
    """1D Imprecise-GP-Modell (Rodemann): robuste Akquisition ueber eine
    Menge unsicherer Priori-Mittelwerte M*h statt eines einzelnen Priors.
    """

    def __init__(self, x_obs, y_obs, y_obs_vars=None, kernel=matern, c=50.0,
                 length_scale=1.0, outputscale=1.0):
        """
        x_obs: array der beobachteten x-Werte
        y_obs: array der beobachteten y-Werte
        y_obs_vars: array der beobachteten y-Varianz (aleatorische Unsicherheit),
                    None im Fall homoskedastischen Rauschens
        kernel: Kernel-Funktion (matern/rbf)
        c: Imprecision-Parameter
        length_scale, outputscale: Kernel-Hyperparameter
        """
        self.x_obs = np.array(x_obs)
        self.y_obs = np.array(y_obs)
        self.y_obs_vars = np.array(y_obs_vars) if y_obs_vars is not None else None

        self.kernel = lambda X, Y: kernel(X, Y, length_scale=length_scale, outputscale=outputscale)
        self.c = c

        # C, C_inv, s_k, S_k vorab berechnen
        self.C = self.kernel(self.x_obs, self.x_obs)
        self.C_inv = self.robust_inverse(self.C, self.x_obs)
        self.s_k = self.C_inv @ np.ones(len(self.x_obs))
        self.S_k = np.ones(len(self.x_obs)) @ self.C_inv @ np.ones(len(self.x_obs))

        # Prior-Mittelwert-Kandidaten
        rng = np.random.default_rng(42)
        self.Ms = np.sort(rng.uniform(0, 5, 250))
        self.Ms_ = np.concatenate([self.Ms, -self.Ms])

        # worst-case Prior
        self.worst_M, self.worst_h = self.compute_worst_case_prior()

        # most-likely Prior (MLL)
        self.mlls = None
        self.most_likely_M, self.most_likely_h = self.compute_most_likely_prior()

    def compute_worst_case_prior(self):
        """M und h, die den Prior am weitesten vom beobachteten Mittel entfernen."""
        worst_dist = -np.inf
        worst_M, worst_h = None, None
        y_mean = np.mean(self.y_obs)

        for M in self.Ms:
            for h in [-1, 1]:
                dist_to_obs = np.abs(M * h - y_mean)
                if dist_to_obs > worst_dist:
                    worst_dist = dist_to_obs
                    worst_M, worst_h = M, h
        return worst_M, worst_h

    def compute_most_likely_prior(self):
        """M und h mit der groessten Marginal-Log-Likelihood (MLL)."""
        mlls = [self.compute_mll_for_prior_mean(M) for M in self.Ms_]
        self.mlls = mlls
        idx = np.argmax(mlls)
        M_ml = float(np.abs(self.Ms_[idx]))
        h_ml = float(np.sign(self.Ms_[idx]))
        return M_ml, h_ml

    def k_vec(self, x):
        """k(x) als Vektor zwischen x und allen beobachteten Punkten."""
        return np.array([self.kernel(np.array([x]), np.array([xi]))[0, 0] for xi in self.x_obs])

    def k_scalar(self, x):
        """k(x,x), skalare Selbst-Kovarianz."""
        return self.kernel(np.array([x]), np.array([x]))[0, 0]

    def mu_bounds_diff(self, x):
        """Differenz zwischen den mu-Grenzen an x (Breite des Credal-Sets)."""
        kx = self.kernel(np.array([x]), self.x_obs).flatten()
        term = self.s_k @ self.y_obs / self.S_k
        if np.abs(term) > 1 + self.c / self.S_k:
            diff = (1 - kx @ self.s_k) * (term + self.c / self.S_k - term / (self.c + self.S_k))
        else:
            diff = 2 * self.c * np.abs(1 - kx @ self.s_k) / self.S_k
        return diff

    def update(self):
        """C, C_inv, s_k, S_k nach einem Daten-Update neu berechnen."""
        self.C = self.kernel(np.array(self.x_obs), np.array(self.x_obs))
        self.C_inv = self.robust_inverse(self.C, np.array(self.x_obs))
        self.s_k = self.C_inv @ np.ones(len(self.x_obs))
        self.S_k = np.ones(len(self.x_obs)) @ self.C_inv @ np.ones(len(self.x_obs))

    def robust_inverse(self, C, x_obs):
        if self.y_obs_vars is None:
            return np.linalg.inv(C + 1e-8 * np.eye(len(x_obs)))
        else:
            return np.linalg.inv(C + self.y_obs_vars * np.eye(len(x_obs)))

    def predict_posterior(self, x_new, M=0.0, h=1):
        """GP-Posterior-Mittelwert/-Varianz fuer einen neuen Punkt.

        M: Prior-Mittelwert-Parameter
        h: +/- 1
        """
        num_obs = len(self.y_obs)

        k = np.array([self.kernel(np.array([x_new]), np.array([xi]))[0, 0] for xi in self.x_obs])
        K_xx = self.kernel(np.array([x_new]), np.array([x_new]))[0, 0]

        y_hat = ((M + 1) * self.s_k.T @ self.y_obs + self.c * M * h) / (self.c + (M + 1) * self.S_k)

        mu_pred = k @ self.C_inv @ (self.y_obs - y_hat * np.ones(num_obs)) + y_hat

        prior_term = (M + 1) * (1 - k @ self.s_k) * (1 - k.T @ self.s_k) / (self.c + (M + 1) * self.S_k)
        sigma2_pred = K_xx - k @ self.C_inv @ k.T + prior_term

        return mu_pred, sigma2_pred

    def compute_mll_for_prior_mean(self, M):
        """Marginal-Log-Likelihood fuer einen gegebenen konstanten Prior-Mittelwert M."""
        y_centered = self.y_obs - M
        return -0.5 * y_centered.T @ self.C_inv @ y_centered

    def UCB(self, x, tau=1.0):
        """(Un-)Lower-Confidence-Bound am worst-/most-likely-Prior (=Standardposterior, M=0)."""
        mu, sigma = self.predict_posterior(x)
        return -mu + tau * sigma

    def PROBO(self, x, tau=1.0, rho=1.0):
        """GLCB nach Rodemann: Standard-UCB-Term + rho * Breite des Credal-Sets."""
        kx = self.k_vec(x)
        mean_term = -kx @ self.C_inv @ np.array(self.y_obs)
        var_term = tau * (self.k_scalar(x) - kx @ self.C_inv @ kx.T)
        return mean_term + var_term + rho * self.mu_bounds_diff(x)
