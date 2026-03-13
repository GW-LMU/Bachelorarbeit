import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.special import expit  # Sigmoid
import random

def matern(x, y, length_scale=1.0, outputscale=1.0):
    dists = np.abs(np.subtract.outer(x, y))
    sqrt3_dists = np.sqrt(3) * dists / length_scale
    return outputscale * (1.0 + sqrt3_dists) * np.exp(-sqrt3_dists)

def rbf(x, y, length_scale=1.0, outputscale=1.0):
    sqdist = np.subtract.outer(x, y)**2
    return outputscale * np.exp(-0.5 * sqdist / length_scale**2)


class BayesOptimizer:
    def __init__(self, x_obs, y_obs, y_obs_vars=None, kernel=matern, c=50.0, length_scale=1.0, outputscale=1.0):
        """
        x_obs: array of observed x-values
        y_obs: array of observed y-values
        y_obs_vars: array of observed y-variance (aleatoric uncertainty), is None in case of homoscedastic noise
        kernel: callable kernel function
        c: imprecision parameter
        length_scale, outputscale: kernel hyperparameters
        """
        self.x_obs = np.array(x_obs)
        self.y_obs = np.array(y_obs)
        self.y_obs_vars = np.array(y_obs_vars) if y_obs_vars is not None else None

        self.kernel = lambda X, Y: kernel(X, Y, length_scale=length_scale, outputscale=outputscale)
        self.c = c

        # Precompute C, C_inv, s_k, S_k
        self.C = self.kernel(self.x_obs, self.x_obs)
        self.C_inv = self.robust_inverse(self.C, self.x_obs)
        self.s_k = self.C_inv @ np.ones(len(self.x_obs))
        self.S_k = np.ones(len(self.x_obs)) @ self.C_inv @ np.ones(len(self.x_obs))

        # prior means
        np.random.seed(42)
        self.Ms = np.random.uniform(0, 5, 250)  # 2001 random numbers in [-1000, 1000]
        self.Ms = np.sort(self.Ms)
        self.Ms_ = np.concatenate([self.Ms, -self.Ms])


        # compute worst-case prior
        self.worst_M, self.worst_h = self.compute_worst_case_prior()
        print(f"Worst-case prior is {self.worst_M} with h {self.worst_h}")


        # compute most-likely prior (MLL)
        self.mlls = None # will be set in the following function
        self.most_likely_M, self.most_likely_h = self.compute_most_likely_prior()
        print(f"Most likely prior is {self.most_likely_M} with h {self.most_likely_h}")

        # default AF
        self._AF = self.PROBO


        # default collection of posterior means and variances
        self.post_means = None
        self.post_vars = None


    def compute_worst_case_prior(self):
        """Return M and h that produce the prior farthest from the observed mean."""
        worst_dist = -np.inf
        worst_M, worst_h = None, None
        y_mean = np.mean(self.y_obs)

        for M in self.Ms:
            for h in [-1, 1]:
                dist_to_obs = np.abs(M * h - y_mean)  # or np.max(np.abs(M*h - self.y_obs)) if desired
                if dist_to_obs > worst_dist:
                    worst_dist = dist_to_obs
                    worst_M, worst_h = M, h
        return worst_M, worst_h

    def compute_most_likely_prior(self):
        """Return M and h with the largest marginal log-likelihood (MLL)."""
        mlls = [self.compute_mll_for_prior_mean(M) for M in self.Ms_]
        self.mlls = mlls
        idx = np.argmax(mlls)
        M_ml = float(np.abs(self.Ms_[idx]))
        h_ml = float(np.sign(self.Ms_[idx]))
        return M_ml, h_ml

    def k_vec(self, x):
        """Return k(x) as vector between x and all observed points."""
        return np.array([self.kernel(np.array([x]), np.array([xi]))[0, 0] for xi in self.x_obs])

    def k_scalar(self, x):
        """Return k(x,x) scalar self-covariance."""
        return self.kernel(np.array([x]), np.array([x]))[0, 0]

    def mu_bounds_diff(self, x):
        """Compute difference between mu bounds at a given x."""
        kx = self.kernel(np.array([x]), self.x_obs).flatten()  # make sure it's 1D
        term = self.s_k @ self.y_obs / self.S_k
        if np.abs(term) > 1 + self.c / self.S_k:
            diff = (1 - kx @ self.s_k) * (term + self.c / self.S_k - term / (self.c + self.S_k))
        else:
            diff = 2 * self.c * np.abs(1 - kx @ self.s_k) / self.S_k
        return diff

    def update(self):
        """Recompute C, C_inv, s_k, S_k after data update."""
        self.C = self.kernel(np.array(self.x_obs), np.array(self.x_obs))
        self.C_inv = self.robust_inverse(self.C, np.array(self.x_obs))
        self.s_k = self.C_inv @ np.ones(len(self.x_obs))
        self.S_k = np.ones(len(self.x_obs)) @ self.C_inv @ np.ones(len(self.x_obs))

    def robust_inverse(self, C, x_obs):
        if self.y_obs_vars is None:
            return np.linalg.inv(C + 1e-8 * np.eye(len(x_obs)))
        else:
            return np.linalg.inv(C + self.y_obs_vars * np.eye(len(x_obs)))


    def predict_posterior_batch(self, x_new):
        """
        Compute multiple posterior mean and variances over a set of M values.
        """
        M_values = self.Ms

        means, vars = [], []
        for M in M_values:
            for h in [-1,1]:
                mu_pred, sigma2_pred = self.predict_posterior(x_new, M=M, h=h)
                means.append(mu_pred)
                vars.append(sigma2_pred)

        self.post_means = means
        self.post_vars = vars

        return means, vars

    def expected_ucb(self, x_new, tau=1):
        means, vars = [], []
        for M in self.Ms:
            for h in [-1,1]:
                mu_pred, sigma2_pred = self.predict_posterior(x_new, M=M, h=h)
                means.append(mu_pred)
                vars.append(sigma2_pred)
        means = np.array(means)
        vars = np.array(vars)
        ucbs = -means + tau*vars
        expected_ucb = np.mean(ucbs)
        return expected_ucb


    def worst_case_ucb(self, x, tau=1):
        """
        Compute the worst-case prior mean for the GP-UCB.
        It selects M and h such that the prior mean is farthest from observed y.
        """
        if self.y_obs is None or len(self.y_obs) == 0:
            raise ValueError("No observations available for worst-case computation.")

        mu, sigma = self.predict_posterior(x, M=self.worst_M, h=self.worst_h)
        return -mu + tau * sigma

    def average_posterior(self, x_new, tau):
        """
        Compute average posterior mean and variance over a set of M values.
        """
        means, vars = self.predict_posterior_batch(x_new)
        return np.mean(means), np.mean(vars)



    def predict_posterior(self, x_new, M=0.0, h=1):
        """
        Predict GP posterior mean and variance for a new point.
        M: prior mean parameter
        h: +/- 1
        """
        num_obs = len(self.y_obs)

        # Cross-covariance vector
        k = np.array([self.kernel(np.array([x_new]), np.array([xi]))[0, 0] for xi in self.x_obs])
        # Self-covariance scalar
        K_xx = self.kernel(np.array([x_new]), np.array([x_new]))[0, 0]

        # Adjusted prior mean term
        y_hat = ((M + 1) * self.s_k.T @ self.y_obs + self.c * M * h) / (self.c + (M + 1) * self.S_k)

        # Posterior mean
        mu_pred = k @ self.C_inv @ (self.y_obs - y_hat * np.ones(num_obs)) + y_hat

        # Prior term
        prior_term = (M + 1) * (1 - k @ self.s_k) * (1 - k.T @ self.s_k) / (self.c + (M + 1) * self.S_k)

        # Posterior variance
        sigma2_pred = K_xx - k @ self.C_inv @ k.T + prior_term

        return mu_pred, sigma2_pred

    def compute_mll_for_prior_mean(self, M):
        """
        Compute the marginal log-likelihood for a given constant prior mean M.
        """
        y_centered = self.y_obs - M
        term1 = -0.5 * y_centered.T @ self.C_inv @ y_centered
        #sign, logdet = np.linalg.slogdet(self.C)  # more stable than np.log(np.linalg.det(...))
        #term2 = -0.5 * logdet
        #term3 = -0.5 * len(self.y_obs) * np.log(2 * np.pi)
        mll = term1 #+ term2 + term3
        return mll

    def most_likely_ucb(self, x_new, tau=1):
        mu, sigma = self.predict_posterior(x_new, M=self.most_likely_M, h=self.most_likely_h)
        return -mu + tau * sigma


    def CredalAF(self, x, rho=1.0):
        means, sigmas = self.predict_posterior_batch(x)
        crossproduct = False
        if crossproduct:
            M, S = np.meshgrid(np.array(means), np.array(sigmas), indexing="ij")
            ucb_vals = -M + rho * S
            ucb_vals = ucb_vals.flatten()
        else:
            ucb_vals = - np.array(means) + rho * np.array(sigmas)

        return ucb_vals

    def UCB(self, x, tau=1.0):
        mu, sigma = self.predict_posterior(x)
        """Lower Confidence Bound acquisition function."""
        return -mu + tau * sigma
        #kx = self.k_vec(x)
        #mean_term = - kx @ self.C_inv @ np.array(self.y_obs)
        #var_term = tau * (self.k_scalar(x) - kx @ self.C_inv @ kx.T)
        #return mean_term + var_term

    def MEAN(self, x):
        """Lower Confidence Bound acquisition function."""
        kx = self.k_vec(x)
        mean_term = kx @ self.C_inv @ np.array(self.y_obs)
        return - mean_term

    def PROBO(self, x, tau=1.0, rho=1.0):
        "GLCB from julian rodemann"
        kx = self.k_vec(x)
        mean_term = - kx @ self.C_inv @ np.array(self.y_obs)
        var_term = tau * (self.k_scalar(x) - kx @ self.C_inv @ kx.T)
        return mean_term + var_term + rho * self.mu_bounds_diff(x)

    def noise_variance(self, x):
        x = np.asarray(x)  # ensure x is a NumPy array
        return (0.01 + 0.19 * expit(x))

    def RAHBO(self, x):
        return self.UCB(x) - 2*self.noise_variance(x)


    def set_AF(self, AF_name):
        """Set the acquisition function dynamically by name."""
        if AF_name == "UACB":
            self._AF = self.UACB
        elif AF_name == "RAHBO":
            self._AF = self.RAHBO
        else:
            raise ValueError(f"Unknown acquisition function: {AF_name}")

    def AF(self, x):
        """Call the currently set acquisition function."""
        return self._AF(x)


    def select_next_x(self, bound=5.0, tau=1.0, method="L-BFGS-B", n_restarts=100):
        """Select next x by maximizing an acquisition function (LCB here)."""
        best_val = np.inf
        best_x = None

        for _ in range(n_restarts):
            x0 = np.random.uniform(-bound, bound, size=1)
            res = scipy.optimize.minimize(lambda x: - self.AF(float(x)),
                                          x0,
                                          bounds=[(-bound, bound)],
                                          method=method)
            if res.fun < best_val:
                best_val = res.fun
                best_x = float(res.x[0])

        best_x = self.next_x

        return best_x


    def batch_predict(self, x_pred, M_values, h_values=None):
        """
        Run posterior predictions for multiple M values and h values.
        Returns arrays of shape [n_models, n_points].
        """
        x_pred = np.atleast_1d(x_pred)

        all_y_preds = []
        all_sigmas = []

        for M in M_values:
            for h in [-1,1]:
                y_pred, sigmas = zip(*[self.predict_posterior(x, M=abs(M), h=h) for x in x_pred])
                sigmas = np.maximum(sigmas, 0)
                all_y_preds.append(y_pred)
                all_sigmas.append(sigmas)

        return np.array(all_y_preds), np.array(all_sigmas)

    def predict_uncertainties(self, x_pred, M_values, h_values=None):
        """
        Compute epistemic, aleatoric, and total uncertainty for given test points.
        """
        all_y_preds, all_sigmas = self.batch_predict(x_pred, M_values, h_values)

        self.all_y_preds = all_y_preds
        self.all_sigmas = all_sigmas


        epistemic = np.var(all_y_preds, axis=0, ddof=1)
        aleatoric = np.mean(all_sigmas, axis=0)
        total = epistemic + aleatoric

        return epistemic, aleatoric, total



    def plot_motiv(self, f, f_noise, noise_var, bound=5.0, Ms=None):
        if Ms is None:
            Ms = self.Ms

        kernel_list = [self.kernel]

        x_pred = np.linspace(-bound, bound, 1000)
        y_dense = f(x_pred)
        y_dense_obs = f_noise(x_pred)
        min_idx = np.argmin(y_dense)
        x_min, y_min = x_pred[min_idx], y_dense[min_idx]

        fig, axs = plt.subplots(2, 2, figsize=(14, 4.5), sharex=True) #, gridspec_kw={'height_ratios': [1, 1, 1]})

        axs[0, 0].grid(True)
        axs[0, 1].grid(True)
        axs[1, 0].grid(True)
        axs[1, 1].grid(True)

        #axs[0].set_title("Target function")
        axs[0,0].plot(x_pred, y_dense, "k--", label="Function f(x)")
        axs[0,0].plot(self.x_obs, self.y_obs, "ro", markersize=8, label="Noisy observations")
        axs[0, 0].plot([x_min], [y_min], "*", markersize=16, label="Minimum", color="gold")

        axs[0,1].plot(x_pred, y_dense, "k--")
        axs[0, 1].plot([x_min], [y_min], "*", markersize=16, label="Minimum", color="gold")
        axs[0,1].plot(self.x_obs, self.y_obs, "ro", markersize=8)



        axs[0,0].legend(loc="upper left", ncol=3, facecolor="white", framealpha=1)
        ct = 0
        kernel_names = {matern: "Matérn", rbf: "RBF"}
        accept = [0, 2 * len(Ms)]

        for kernel in kernel_list:
            for M in Ms:
                h = -1 if M < 0 else 1
                predictions = [self.predict_posterior(x_new, M=abs(M), h=h) for x_new in x_pred]
                y_pred, sigmas = np.transpose(predictions)
                sigmas = np.maximum(sigmas, 0)
                if (M == 0) and (h == 1):
                    total = sigmas
                    axs[0, 0].plot(self.x_obs, self.y_obs, "ro", markersize=8)
                    axs[0, 0].plot(x_pred, f(x_pred), "k--")
                    axs[0, 0].plot(x_pred, y_pred,
                                   color="blue",
                                   label=f"GP mean {kernel_names.get(kernel, 'Kernel')}" if ct in accept else None)
                    axs[0, 0].fill_between(x_pred,
                                           y_pred - np.sqrt(sigmas) * 1.96,
                                           y_pred + np.sqrt(sigmas) * 1.96,
                                           color="blue",
                                           alpha=0.2,
                                           label=f"GP Uncertainty {kernel_names.get(kernel, 'Kernel')}" if ct in accept else None)
                    axs[0, 0].plot(x_min, y_min, "y*", markersize=16)
                    axs[0, 0].grid(True)

                axs[0,1].plot(self.x_obs, self.y_obs, "ro", markersize=8)
                axs[0,1].plot(x_pred, f(x_pred), "k--")
                axs[0,1].plot(x_pred, y_pred,
                            color="blue",
                            label=f"GP mean {kernel_names.get(kernel, 'Kernel')}" if ct in accept else None)
                axs[0,1].fill_between(x_pred,
                                    y_pred - np.sqrt(sigmas) * 1.96,
                                    y_pred + np.sqrt(sigmas) * 1.96,
                                    color="blue",
                                    alpha=0.2,
                                    label=f"GP Uncertainty {kernel_names.get(kernel, 'Kernel')}" if ct in accept else None)
                axs[0,1].plot(x_min, y_min, "y*", markersize=16)
                axs[0,1].grid(True)

                if ct == 0:
                    all_y_preds = [y_pred]
                    all_sigmas = [sigmas]
                else:
                    all_y_preds.append(y_pred)
                    all_sigmas.append(sigmas)
                ct += 1

        #################################################################
        ## standard BO AFs
        #################################################################


        ucb_vals = []
        for x in x_pred:
            ucb_vals.append(self.UCB(x))
        max_idx_ucb = np.argmax(ucb_vals)
        ucb_val = ucb_vals[max_idx_ucb]
        mus_0, sigmas_0 = [], []

        # axs[1,1].plot(x_pred, ucb_vals, color="forestgreen")
        # axs[1,1].plot(x_pred[max_idx_ucb], ucb_val, '*', markersize=20, color="forestgreen")
        axs[1, 0].plot(x_pred, ucb_vals, color="forestgreen", label="UCB")
        axs[1, 0].plot(x_pred[max_idx_ucb], ucb_val, '*', markersize=20, color="forestgreen")

        # RAHBO
        RAHBO = False
        if RAHBO:
            rahbo_vals = []
            for x in x_pred:
                rahbo_vals.append(self.RAHBO(x))
            axs[1, 0].plot(x_pred, rahbo_vals, label="RAHBO", color="red", linestyle="-")
            max_idx_rahbo = np.argmax(rahbo_vals)
            rahbo_val = rahbo_vals[max_idx_rahbo]
            axs[1, 0].plot(x_pred[max_idx_rahbo], rahbo_val, '*', markersize=20, color="red")
            axs[1, 1].plot(x_pred, rahbo_vals, color="red", linestyle="-")
            axs[1, 1].plot(x_pred[max_idx_rahbo], rahbo_val, '*', markersize=20, color="red")



        #################################################################
        ## credal BO AFs
        #################################################################
        credal_intervals = [self.CredalAF(x) for x in x_pred]

        # Separate lower and upper bounds
        lower_bounds = [float(interval.min()) for interval in credal_intervals]
        upper_bounds = [float(interval.max()) for interval in credal_intervals]
        axs[1,1].fill_between(x_pred, lower_bounds, upper_bounds, color='orange', alpha=0.3, label='Cred. AF')
        # Add lower and upper bounds as dashed lines
        #axs[1, 1].plot(x_pred, lower_bounds, linestyle='-', color='orange', alpha=0.8)
        #axs[1, 1].plot(x_pred, upper_bounds, linestyle='-', color='orange', alpha=0.8)

        probo_vals = []
        for x in x_pred:
            probo_vals.append(self.PROBO(x))
        #axs[1,1].plot(x_pred, probo_vals, color="teal", linestyle="-") #, marker="x", markevery=slice(25, None, 60), markersize=8)
        max_idx_probo = np.argmax(probo_vals)
        probo_val = probo_vals[max_idx_probo]
        #axs[1,1].plot(x_pred[max_idx_probo], probo_val, '*', markersize=20, color="teal")
        mus_0 = []
        sigmas_0 = []
        for x in x_pred:
            mu_0, sigma_0 = self.predict_posterior(x)
            mus_0.append(mu_0)
            sigmas_0.append(sigma_0)
        ucb_0 = -np.array(mus_0) + np.array(sigmas_0)
        max_idx_ucb_0 = np.argmax(ucb_0)
        axs[1,1].plot(x_pred, ucb_0, color="red", label="UCB", linestyle=":")
        axs[1,1].plot(x_pred[max_idx_ucb_0], ucb_0[max_idx_ucb_0], '*', markersize=20, color="red")
        axs[1, 0].plot(x_pred, probo_vals, label="PROBO", color="teal",
                       linestyle="-")  # , marker="x", markevery=slice(25, None, 60), markersize=8)
        axs[1, 0].plot(x_pred[max_idx_probo], probo_val, '*', markersize=20, color="teal")

        axs[1, 0].legend()


        ucb_mean = []
        for x in x_pred:
            ucb_mean.append(self.expected_ucb(x))
        max_idx_ucb_mean = np.argmax(ucb_mean)
        ucb_val_mean = ucb_mean[max_idx_ucb_mean]
        axs[1, 1].plot(x_pred, ucb_mean, color="black", linewidth=2, label="Exp. UCB",linestyle="dashdot")
        axs[1, 1].plot(x_pred[max_idx_ucb_mean], ucb_val_mean, '*', markersize=20, color="black")


        worst_case = []
        for x in x_pred:
            worst_case.append(self.worst_case_ucb(x))
        max_idx_worst_case = np.argmax(worst_case)
        ucb_val_worst_case = worst_case[max_idx_worst_case]
        #axs[1, 1].plot(x_pred, worst_case, color="black", linewidth=2, label="Pessimistic UCB",linestyle=":")
        #axs[1, 1].plot(x_pred[max_idx_worst_case], ucb_val_worst_case, '*', markersize=20, color="black")


        mll = []
        for x in x_pred:
            mll.append(self.most_likely_ucb(x))
        max_idx_mll = np.argmax(mll)
        ucb_val_mll = mll[max_idx_mll]
        axs[1, 1].plot(x_pred, mll, color="black", linewidth=2, label="Most likely UCB",linestyle="-")
        axs[1, 1].plot(x_pred[max_idx_mll], ucb_val_mll, '*', markersize=20, color="black")

        weights = np.exp(self.mlls - np.max(self.mlls))  # shift for numerical stability
        weights /= np.sum(weights)

        all_ucb_distributions = {}  # dict: x -> list of ucb values
        all_ucb_distributions_weights = {}  # dict: x -> list of ucb values

        for x in x_pred[::100]:
            ucbs = []

            for i,M_ in enumerate(self.Ms_):
                M = np.abs(M_)
                h = float(np.sign(M_))
                mu_pred, sigma2_pred = self.predict_posterior(x, M=M, h=h)
                #ucbs_scatters.append(-np.array(mu_pred) + np.array(sigma2_pred))
                ucb = -mu_pred +  sigma2_pred
                ucbs.append(ucb)

                axs[1,1].hlines(ucb, x-weights[i]*3, x+weights[i]*3, color="blue", alpha=0.8)
            all_ucb_distributions[x] = np.array(ucbs)
            all_ucb_distributions_weights[x] = (np.array(ucbs), weights.copy())

        def empirical_cdf(samples, grid):
            samples = np.sort(samples)
            return np.searchsorted(samples, grid, side="right") / len(samples)

        def weighted_empirical_cdf(values, weights, grid):
            """
            Compute weighted empirical CDF on a fixed grid.

            Args:
                values: array of sample values
                weights: array of nonnegative weights (need not sum to 1)
                grid: grid points to evaluate CDF at
            Returns:
                cdf array same length as grid
            """
            # normalize weights
            weights = np.array(weights, dtype=float)
            weights = weights / np.sum(weights)

            # sort values with weights
            idx = np.argsort(values)
            values_sorted = values[idx]
            weights_sorted = weights[idx]
            cumweights = np.cumsum(weights_sorted)

            # for each grid point, find where it falls in sorted values
            cdf = np.searchsorted(values_sorted, grid, side="right")
            cdf = np.array([cumweights[i - 1] if i > 0 else 0.0 for i in cdf])
            return cdf

        def check_weighted_SSD(values_X, weights_X, values_Y, weights_Y, x2, x1, grid_size=500, plot=True):
            xmin = min(np.min(values_X), np.min(values_Y))
            xmax = max(np.max(values_X), np.max(values_Y))
            grid = np.linspace(xmin, xmax, grid_size)

            F_X = weighted_empirical_cdf(values_X, weights_X, grid)
            F_Y = weighted_empirical_cdf(values_Y, weights_Y, grid)

            # Integrated CDFs
            int_FX = np.cumsum(F_X) * (grid[1] - grid[0])
            int_FY = np.cumsum(F_Y) * (grid[1] - grid[0])

            cond = np.all(int_FX <= int_FY + 1e-9) and np.any(int_FX < int_FY - 1e-9)

            if plot:


                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 2, figsize=(12, 4))

                axs[0].plot(grid, F_X, label=f"{x2}", color="blue")
                axs[0].plot(grid, F_Y, label=f"{x1}", color="red")
                axs[0].set_title("Weighted CDFs")
                axs[0].legend();
                axs[0].grid(True)

                axs[1].plot(grid, int_FX, label="∫F_X", color="blue")
                axs[1].plot(grid, int_FY, label="∫F_Y", color="red")
                axs[1].fill_between(grid, int_FX, int_FY, color="gray", alpha=0.3)
                axs[1].set_title("Integrated CDFs (SSD check)")
                axs[1].legend();
                axs[1].grid(True)

                plt.show()

            return cond, int_FX - int_FY, grid

        def check_SSD(samples_X, samples_Y, x2, x1, grid_size=500, plot=True):
            xmin = min(np.min(samples_X), np.min(samples_Y))
            xmax = max(np.max(samples_X), np.max(samples_Y))
            grid = np.linspace(xmin, xmax, grid_size)

            F_X = empirical_cdf(samples_X, grid)
            F_Y = empirical_cdf(samples_Y, grid)

            # Integrated CDFs
            int_FX = np.cumsum(F_X) * (grid[1] - grid[0])
            int_FY = np.cumsum(F_Y) * (grid[1] - grid[0])

            cond = np.all(int_FX <= int_FY + 1e-9) and np.any(int_FX < int_FY - 1e-9)

            if plot:
                fig, axs = plt.subplots(1, 3, figsize=(16, 4))

                # Plot CDFs
                axs[0].plot(grid, F_X, label=f"{x2}", color="blue")
                axs[0].plot(grid, F_Y, label=f"{x1}", color="red")
                axs[0].set_title("CDFs")
                axs[0].set_xlabel("x");
                axs[0].set_ylabel("F(x)")
                axs[0].legend();
                axs[0].grid(True)

                # Plot integrated CDFs
                axs[1].plot(grid, int_FX, label="∫F_X", color="blue")
                axs[1].plot(grid, int_FY, label="∫F_Y", color="red")
                axs[1].fill_between(grid, int_FX, int_FY, color="gray", alpha=0.3)
                axs[1].set_title("Integrated CDFs (SSD check)")
                axs[1].set_xlabel("x");
                axs[1].set_ylabel("∫ F(s) ds")
                axs[1].legend();
                axs[1].grid(True)

                # Violin plots of raw distributions
                axs[2].violinplot([samples_X, samples_Y], positions=[1, 2], showmeans=True, showextrema=True)
                axs[2].set_xticks([1, 2])
                axs[2].set_xticklabels(["X distribution", "Y distribution"])
                axs[2].set_title("Distribution Comparison (Violin Plots)")
                axs[2].grid(True)

                plt.tight_layout()
                plt.show()

            return cond, int_FX - int_FY, grid

        domination_counts = {x: 0 for x in all_ucb_distributions.keys()}

        for x1, dist1 in all_ucb_distributions.items():
            for x2, dist2 in all_ucb_distributions.items():
                if x1 == x2:
                    continue
                cond, _, _ = check_SSD(dist2, dist1, x2, x1, plot=False)
                if cond:  # x2 dominates x1
                    domination_counts[x1] += 1

        # Step 2: Select non-dominated or least dominated
        min_dom = min(domination_counts.values())
        candidates = [x for x, c in domination_counts.items() if c == min_dom]

        # Step 3: If tie, use expected utility - lam * variance
        if len(candidates) > 1:
            scores = {}
            for x in candidates:
                dist = np.array(all_ucb_distributions[x])
                mean = dist.mean()
                var = dist.var()
                scores[x] = mean - 1 * var

            max_score = max(scores.values())
            candidates = [x for x, s in scores.items() if np.isclose(s, max_score)]

        # Step 4: Fallback random
        best_x = random.choice(candidates)

        all_ucb_distributions = {}  # dict: x -> (ucb values, weights)
        #print("Best x according to SSD:", best_x)
        #axs[1, 1].plot(best_x, self.UCB(best_x), '*', markersize=20, color="red", label="SSD - credal interval")
        axs[1, 1].axvline(
            x=best_x,
            color="orange",
            linestyle="-.",
            linewidth=1.5,
            label="SSD - cred."
        )

        domination_counts = {x: 0 for x in all_ucb_distributions_weights.keys()}

        for x1, (dist1, w1) in all_ucb_distributions_weights.items():
            for x2, (dist2, w2) in all_ucb_distributions_weights.items():
                if x1 == x2:
                    continue
                cond, _, _ = check_weighted_SSD(dist2, w2, dist1, w1, x2, x1, plot=False)
                if cond:  # x2 dominates x1
                    domination_counts[x1] += 1

        # Step 2: Collect non-dominated or least dominated candidates
        min_dom = min(domination_counts.values())
        candidates = [x for x, c in domination_counts.items() if c == min_dom]

        # Step 3: If more than one candidate remains, use expected utility - lam * variance
        if len(candidates) > 1:
            scores = {}
            for x in candidates:
                dist, w = all_ucb_distributions_weights[x]

                # compute expected utility
                exp_util = np.sum(np.array(dist) * np.array(w))

                # compute variance as weighted second moment minus mean^2
                mean = exp_util
                var = np.sum(w * (dist - mean) ** 2)

                score = exp_util - 1 * var
                scores[x] = score

            max_score = max(scores.values())
            candidates = [x for x, s in scores.items() if np.isclose(s, max_score)]

        # Step 4: Fallback random if still tied
        best_x = random.choice(candidates)

        self.next_x = best_x
        #print("Best x according to weighted SSD:", best_x)
        #axs[1, 1].plot(best_x, self.UCB(best_x), '*', markersize=20, color="blue", label="SSD - MLL interval")
        axs[1, 1].axvline(
            x=best_x,
            color="blue",
            linestyle=":",
            linewidth=1.5,
            label="SSD - MLL"
        )
        axs[1,1].legend(loc='upper center',
            bbox_to_anchor=(0.5, 1.25),
            ncol=5,
            facecolor='white',
            framealpha=1,
        )

        return fig, axs

