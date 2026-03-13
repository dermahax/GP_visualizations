# Gaussian Process Visualizations

A collection of  Jupyter notebooks that build up intuition for **Gaussian Processes (GPs)** through step-by-step visualizations.

## Examples

### `gp_prior_plots.ipynb` — GP Prior -> Posterior
- GP **prior** mean with 2-sigma confidence band
- GP prior with 3 random draws
- GP **posterior** updating as observations are added one by one (animation)

| Prior CI | Prior + random draws | Posterior (updating) |
|---|---|---|
| <img src="imgs/gp_prior_confidence_only.png" width="260"/> | <img src="imgs/gp_prior_random_draws.png" width="260"/> | <img src="imgs/gp_posterior_animation.gif" width="260"/> |

### `gp_hyperparams.ipynb` — Effect of hyperparameter
Demonstrates GP regression with a squared-exponential (RBF) kernel on synthetic data:
- Raw data with noise error bars
- GP fit with the **correct** length-scale (ℓ = 1.0)
- GP fit with a **too-short** length-scale (ℓ = 0.3) — over-fitting
- GP fit with a **too-long** length-scale (ℓ = 3.0) — under-fitting
- Random function samples drawn from each model

| Correct ℓ | Short ℓ | Long ℓ |
|---|---|---|
| <img src="imgs/gp_plot_correct_lengthscale.png" width="260"/> | <img src="imgs/gp_plot_short_lengthscale.png" width="260"/> | <img src="imgs/gp_plot_long_lengthscale.png" width="260"/> |



### `gp_kernels.ipynb` — Kernel Comparison
Shows random draws from the GP prior and posterior fitting for three kernels: Squared Exponential, Linear, and Periodic.

| Squared Exponential | Linear | Periodic |
|---|---|---|
| <img src="imgs/gp_kernel_se_draws.png" width="260"/> | <img src="imgs/gp_kernel_linear_draws.png" width="260"/> | <img src="imgs/gp_kernel_periodic_draws.png" width="260"/> |

## Installation

```bash
pip install -e .
```

## Usage

```bash
python gp_plots.py          # GP regression plots
python gp_prior_plots.py    # prior plots + animation GIF
```

Or open the notebooks in Jupyter / VS Code to explore interactively with inline animation players:

| Notebook | Contents |
|---|---|
| `gp_prior_plots.ipynb` | GP prior CI, random draws, posterior update animation |
| `gp_hyperparams.ipynb` | GP regression with correct, too-short, and too-long length-scales |
| `gp_kernels.ipynb` | Prior draws + posterior fitting animations for SE, Linear, and Periodic kernels |

### `data_creation.py` — Shared Dataset

A standalone module used by all notebooks to generate the same synthetic dataset, ensuring direct comparability across experiments.
The observations are drawn from a GP with noise $\sigma_n = 0.1$.
