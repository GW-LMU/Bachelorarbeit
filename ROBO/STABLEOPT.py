import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


minus_area = -1000
plus_area = 1000


def _sample_ball(center, epsilon, n_samples, lower_bound, upper_bound, rng):
    """
    Zieht n_samples Punkte aus der euklidischen Kugel mit Radius epsilon um `center`
    (Delta_epsilon(x) aus Bogunovic et al. 2018), beschraenkt auf die Suchraumgrenzen.
    """
    dim = center.shape[0]
    directions = rng.normal(size=(n_samples, dim))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.clip(norms, 1e-12, None)
    radii = epsilon * rng.uniform(0.0, 1.0, size=(n_samples, 1)) ** (1.0 / dim)
    points = center[None, :] + directions * radii
    return np.clip(points, lower_bound, upper_bound)


def run_stable_opt(train_X, train_Y, dim, acquisition_func,
                    lower_bound=minus_area, upper_bound=plus_area,
                    epsilon=None, beta=2.0, n_x_candidates=256, n_delta_candidates=32):
    """
    Führt eine Iteration des STABLE OPT-Algorithmus aus (Bogunovic et al. 2018,
    "Adversarially Robust Optimization with Gaussian Processes", Algorithm 1):

        x~_t = argmax_x  min_{delta in Delta_eps(x)}  ucb_{t-1}(x + delta)      (13)
        delta_t = argmin_{delta in Delta_eps(x~_t)}  lcb_{t-1}(x~_t + delta)     (Zeile 3)
        query: x~_t + delta_t

    Delta_eps(x) ist die euklidische epsilon-Kugel um x (d(x,x')=||x-x'||_2), geschnitten
    mit dem Suchraum. Da die verschachtelte Min-Max-Optimierung auf kontinuierlichem X
    im Allgemeinen nicht exakt lösbar ist (siehe Paper, Abschnitt 3), wird sie hier per
    Kandidaten-Sampling approximiert, analog zum num_restarts/raw_samples-Vorgehen der
    übrigen Surrogatmodelle in diesem Projekt.

    train_X : torch.double, shape [n, dim]
    train_Y : torch.double, shape [n, 1] (Maximierungs-Konvention, siehe coco_benchmark.py)
    dim     : int, Dimension des Suchraums
    epsilon : Stabilitätsradius (Standard: 1% der Suchraumbreite)

    Rückgabe: next_x (torch.double, shape [dim]), acq_value (torch.double, Skalar; ucb-Wert von x~_t)
    """
    if acquisition_func != "UCB":
        raise ValueError(
            f"STABLE OPT verwendet intern feste (ucb,lcb)-Konfidenzbänder; "
            f"unbekannte/inkompatible Akquisitionsfunktion: {acquisition_func}"
        )

    if epsilon is None:
        epsilon = 0.01 * (upper_bound - lower_bound)

    bounds = torch.tensor(
        [[lower_bound] * dim, [upper_bound] * dim], dtype=torch.double
    )

    # 1) Modell fitten (Inputs auf [0,1]^d normalisiert, Outputs standardisiert)
    model = SingleTaskGP(
        train_X, train_Y,
        input_transform=Normalize(d=dim, bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    rng = np.random.default_rng()

    def ucb_lcb(points_np):
        points = torch.as_tensor(points_np, dtype=torch.double)
        with torch.no_grad():
            posterior = model.posterior(points)
            mean = posterior.mean.squeeze(-1)
            std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)
        beta_sqrt = beta ** 0.5
        ucb = (mean + beta_sqrt * std).numpy()
        lcb = (mean - beta_sqrt * std).numpy()
        return ucb, lcb

    # 2) x~_t = argmax_x min_{delta in Delta_eps(x)} ucb_{t-1}(x+delta)  (Kandidaten-Sampling)
    x_candidates = rng.uniform(lower_bound, upper_bound, size=(n_x_candidates, dim))

    best_stable_ucb = -np.inf
    x_tilde = None
    for x in x_candidates:
        perturbed = _sample_ball(x, epsilon, n_delta_candidates, lower_bound, upper_bound, rng)
        ucb_vals, _ = ucb_lcb(perturbed)
        stable_ucb = ucb_vals.min()
        if stable_ucb > best_stable_ucb:
            best_stable_ucb = stable_ucb
            x_tilde = x

    # 3) delta_t = argmin_{delta in Delta_eps(x~_t)} lcb_{t-1}(x~_t + delta)
    perturbed_final = _sample_ball(x_tilde, epsilon, n_delta_candidates, lower_bound, upper_bound, rng)
    _, lcb_vals = ucb_lcb(perturbed_final)
    delta_idx = np.argmin(lcb_vals)
    query_point = perturbed_final[delta_idx]

    next_x = torch.as_tensor(query_point, dtype=torch.double)
    acq_value = torch.as_tensor(best_stable_ucb, dtype=torch.double)

    return next_x, acq_value
