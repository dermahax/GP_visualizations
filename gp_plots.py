import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------
# 1. Squared Exponential Kernel (Rasmussen & Williams notation)
# -----------------------------------------------------------


def squared_exponential(x1, x2, lengthscale=1.0, variance=1.0):
    dists = np.subtract.outer(x1, x2)**2
    return variance * np.exp(-0.5 * dists / lengthscale**2)


# -----------------------------------------------------------
# 2. Generate Synthetic Data (Signal + Noise)
# -----------------------------------------------------------
np.random.seed(42)

N = 12
X_train = np.random.uniform(-5, 5, size=N)
X_train.sort()

ell_true = 1.0     # true lengthscale
sf_true = 1.0     # true signal variance
sn_true = 0.1     # true noise std.dev

K = squared_exponential(
    X_train, X_train, lengthscale=ell_true, variance=sf_true**2)
f = np.random.multivariate_normal(np.zeros(N), K)
y = f + np.random.normal(0, sn_true, size=N)

X_test = np.linspace(-5.5, 5.5, 200)

# -----------------------------------------------------------
# 3. GP Posterior Computation
# -----------------------------------------------------------


def gp_posterior(X, y, X_star, ell, sf, sn):
    K = squared_exponential(X, X,       lengthscale=ell, variance=sf**2)
    Ks = squared_exponential(X, X_star,  lengthscale=ell, variance=sf**2)
    Kss = squared_exponential(X_star, X_star, lengthscale=ell, variance=sf**2)

    Ky = K + sn**2 * np.eye(len(X))
    L = np.linalg.cholesky(Ky + 1e-10*np.eye(len(X)))

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mu = Ks.T @ alpha

    v = np.linalg.solve(L, Ks)
    cov = Kss - v.T @ v

    return mu, cov


# -----------------------------------------------------------
# 4. Conditional Latent Function Sample (for panel a)
# -----------------------------------------------------------
K_xx = squared_exponential(
    X_train, X_train, lengthscale=ell_true, variance=sf_true**2)
K_xxs = squared_exponential(
    X_train, X_test,  lengthscale=ell_true, variance=sf_true**2)
K_xsx = K_xxs.T
K_xsxs = squared_exponential(
    X_test, X_test,  lengthscale=ell_true, variance=sf_true**2)

L_true = np.linalg.cholesky(K_xx + 1e-10*np.eye(len(X_train)))
alpha_t = np.linalg.solve(L_true.T, np.linalg.solve(L_true, f))
mu_fs = K_xsx @ alpha_t
cov_fs = K_xsxs - K_xsx @ np.linalg.solve(K_xx, K_xxs)

f_star = np.random.multivariate_normal(
    mu_fs, cov_fs + 1e-10*np.eye(len(X_test)))

# -----------------------------------------------------------
# 5. GP Posteriors for Various Hyperparameters
# -----------------------------------------------------------
ell_values = [1.0, 0.3, 3.0]
posts = {ell: gp_posterior(X_train, y, X_test, ell, sf_true, sn_true)
         for ell in ell_values}

# -----------------------------------------------------------
# 6. Plot: Data Only + Three GP Fits (3 Separate Plots)
# -----------------------------------------------------------

# Set seaborn theme for nicer plots
sns.set_theme(style="whitegrid", palette="husl")
sns.set_context("notebook", font_scale=1.1)

# get y ranges:
# Collect all y-values that influence scaling
y_min = np.min(y - sn_true)
y_max = np.max(y + sn_true)

# Also include GP posterior ranges
for ell in ell_values:
    mu, cov = posts[ell]
    sd = np.sqrt(np.diag(cov))
    y_min = min(y_min, np.min(mu - 2*sd))
    y_max = max(y_max, np.max(mu + 2*sd))

# Optional: include conditional latent sample (panel a)
y_min = min(y_min, np.min(f_star))
y_max = max(y_max, np.max(f_star))

# Add a small margin
margin = 0.1 * (y_max - y_min)
y_min -= margin
y_max += margin

# ----- Plot 0: Data Only (No Fitted Function) -----
fig0, ax0 = plt.subplots(figsize=(10, 6))

ax0.errorbar(
    X_train, y, yerr=sn_true,
    fmt='o', ms=10, elinewidth=2, capsize=5, label="Observed data", color='steelblue'
)

ax0.set_title("Observed Data with Noise Error Bars",
              fontsize=14, fontweight='bold')
ax0.set_xlabel("Input (x)", fontsize=12)
ax0.set_ylabel("Output (y)", fontsize=12)
ax0.set_xlim(-5.5, 5.5)
ax0.set_ylim(y_min, y_max)
ax0.legend(loc='best', frameon=True, shadow=True)
ax0.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gp_plot_data_only.png", dpi=150, bbox_inches='tight')
plt.show()

# ----- Plot 1: GP Regression with ℓ = 1.0 (correct) -----
fig1, ax1 = plt.subplots(figsize=(10, 6))
ell = 1.0
mu, cov = posts[ell]
sd = np.sqrt(np.diag(cov))

ax1.fill_between(X_test, mu - 2*sd, mu + 2*sd, alpha=0.3,
                 label="95% confidence interval")
ax1.plot(X_test, mu, lw=2.5, label="GP mean prediction")
ax1.plot(X_test, f_star, "--", lw=2, alpha=0.7,
         label="Conditional latent sample")
ax1.errorbar(
    X_train, y, yerr=sn_true,
    fmt='o', ms=8, elinewidth=2, capsize=4, label="Observed data", color='steelblue'
)

ax1.set_title("GP Regression: ℓ = 1.0 (correct length-scale)",
              fontsize=14, fontweight='bold')
ax1.set_xlabel("Input (x)", fontsize=12)
ax1.set_ylabel("Output (y)", fontsize=12)
ax1.set_xlim(-5.5, 5.5)
ax1.set_ylim(y_min, y_max)
ax1.legend(loc='best', frameon=True, shadow=True)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gp_plot_correct_lengthscale.png", dpi=150, bbox_inches='tight')
plt.show()

# ----- Plot 2: GP Regression with ℓ = 0.3 (too short) -----
fig2, ax2 = plt.subplots(figsize=(10, 6))
ell = 0.3
mu, cov = posts[ell]
sd = np.sqrt(np.diag(cov))

ax2.fill_between(X_test, mu - 2*sd, mu + 2*sd, alpha=0.3,
                 label="95% confidence interval")
ax2.plot(X_test, mu, lw=2.5, label="GP mean prediction")
ax2.errorbar(
    X_train, y, yerr=sn_true,
    fmt='o', ms=8, elinewidth=2, capsize=4, label="Observed data", color='steelblue'
)

ax2.set_title("GP Regression: ℓ = 0.3 (too short)",
              fontsize=14, fontweight='bold')
ax2.set_xlabel("Input (x)", fontsize=12)
ax2.set_ylabel("Output (y)", fontsize=12)
ax2.set_xlim(-5.5, 5.5)
ax2.set_ylim(y_min, y_max)
ax2.legend(loc='best', frameon=True, shadow=True)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gp_plot_short_lengthscale.png", dpi=150, bbox_inches='tight')
plt.show()

# ----- Plot 3: GP Regression with ℓ = 3.0 (too long) -----
fig3, ax3 = plt.subplots(figsize=(10, 6))
ell = 3.0
mu, cov = posts[ell]
sd = np.sqrt(np.diag(cov))

ax3.fill_between(X_test, mu - 2*sd, mu + 2*sd, alpha=0.3,
                 label="95% confidence interval")
ax3.plot(X_test, mu, lw=2.5, label="GP mean prediction")
ax3.errorbar(
    X_train, y, yerr=sn_true,
    fmt='o', ms=8, elinewidth=2, capsize=4, label="Observed data", color='steelblue'
)

ax3.set_title("GP Regression: ℓ = 3.0 (too long)",
              fontsize=14, fontweight='bold')
ax3.set_xlabel("Input (x)", fontsize=12)
ax3.set_ylabel("Output (y)", fontsize=12)
ax3.set_xlim(-5.5, 5.5)
ax3.set_ylim(y_min, y_max)
ax3.legend(loc='best', frameon=True, shadow=True)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gp_plot_long_lengthscale.png", dpi=150, bbox_inches='tight')
plt.show()

# ----- Plot 4: Random Function Samples from All Three Cases -----
fig4, ax4 = plt.subplots(figsize=(10, 6))

# Sample one random function from each posterior
colors = ['#e74c3c', '#3498db', '#2ecc71']  # red, blue, green
for ell, color in zip(ell_values, colors):
    mu, cov = posts[ell]
    # Draw one random sample from the posterior
    sample = np.random.multivariate_normal(mu, cov + 1e-10*np.eye(len(X_test)))
    ax4.plot(X_test, sample, lw=2.5, alpha=0.8, color=color)

# Plot the data
ax4.errorbar(
    X_train, y, yerr=sn_true,
    fmt='o', ms=8, elinewidth=2, capsize=4, color='steelblue', label="Observed data"
)

ax4.set_title("Random Function Samples from Different Length-scales",
              fontsize=14, fontweight='bold')
ax4.set_xlabel("Input (x)", fontsize=12)
ax4.set_ylabel("Output (y)", fontsize=12)
ax4.set_xlim(-5.5, 5.5)
ax4.set_ylim(y_min, y_max)
ax4.legend(loc='best', frameon=True, shadow=True)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gp_plot_random_samples.png", dpi=150, bbox_inches='tight')
plt.show()
