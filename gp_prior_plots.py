import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


def squared_exponential(x1, x2, lengthscale=1.0, variance=1.0):
    dists = np.subtract.outer(x1, x2) ** 2
    return variance * np.exp(-0.5 * dists / lengthscale ** 2)


def gp_posterior(X, y, X_star, ell, sf, sn):
    K = squared_exponential(X, X, lengthscale=ell, variance=sf ** 2)
    Ks = squared_exponential(X, X_star, lengthscale=ell, variance=sf ** 2)
    Kss = squared_exponential(
        X_star, X_star, lengthscale=ell, variance=sf ** 2)

    Ky = K + sn ** 2 * np.eye(len(X))
    L = np.linalg.cholesky(Ky + 1e-10 * np.eye(len(X)))

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mu = Ks.T @ alpha

    v = np.linalg.solve(L, Ks)
    cov = Kss - v.T @ v

    return mu, cov


def generate_gp_plots_data():
    # Dedicated random state keeps this dataset identical to gp_plots.py.
    rs = np.random.RandomState(42)

    N = 12
    X_train = rs.uniform(-5, 5, size=N)
    X_train.sort()

    ell_true = 1.0
    sf_true = 1.0
    sn_true = 0.1

    K = squared_exponential(
        X_train, X_train, lengthscale=ell_true, variance=sf_true ** 2)
    f = rs.multivariate_normal(np.zeros(N), K)
    y = f + rs.normal(0, sn_true, size=N)

    return X_train, y, ell_true, sf_true, sn_true


# Reproducibility for random GP draws
np.random.seed(42)

# Prior setup (no data)
X_prior = np.linspace(-5.5, 5.5, 200)
ell = 1.0
sf = 1.0
sigma = 2.0

K_prior = squared_exponential(
    X_prior, X_prior, lengthscale=ell, variance=sf ** 2)
prior_mean = np.zeros(len(X_prior))
prior_sd = np.sqrt(np.diag(K_prior))

# Match plotting style from gp_plots.py
sns.set_theme(style="whitegrid", palette="husl")
sns.set_context("notebook", font_scale=1.1)

y_min = np.min(prior_mean - sigma * prior_sd)
y_max = np.max(prior_mean + sigma * prior_sd)
margin = 0.1 * (y_max - y_min)
y_min -= margin
y_max += margin

# ----- Plot 1: Prior confidence interval only -----
fig1, ax1 = plt.subplots(figsize=(10, 6))

ax1.fill_between(
    X_prior,
    prior_mean - sigma * prior_sd,
    prior_mean + sigma * prior_sd,
    alpha=0.35,
    label="95% confidence interval",
)
ax1.plot(X_prior, prior_mean, lw=2, label="Prior mean")

ax1.set_title("GP Prior: Mean 0 with 2-sigma Confidence Interval",
              fontsize=14, fontweight="bold")
ax1.set_xlabel("Input (x)", fontsize=12)
ax1.set_ylabel("Output (y)", fontsize=12)
ax1.set_xlim(-5.5, 5.5)
ax1.set_ylim(y_min, y_max)
ax1.legend(loc="best", frameon=True, shadow=True)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gp_prior_confidence_only.png", dpi=150, bbox_inches="tight")
plt.show()

# ----- Plot 2: Prior confidence interval + random draws -----
fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.fill_between(
    X_prior,
    prior_mean - sigma * prior_sd,
    prior_mean + sigma * prior_sd,
    alpha=0.35,
    label="95% confidence interval",
)
ax2.plot(X_prior, prior_mean, lw=2, label="Prior mean")

draw_colors = sns.color_palette("husl", 3)
for i, color in enumerate(draw_colors, start=1):
    sample = np.random.multivariate_normal(
        prior_mean, K_prior + 1e-10 * np.eye(len(X_prior))
    )
    ax2.plot(X_prior, sample, lw=2.2, alpha=0.85,
             color=color, label=f"Random GP draw {i}")

ax2.set_title("GP Prior with 3 Random Draws", fontsize=14, fontweight="bold")
ax2.set_xlabel("Input (x)", fontsize=12)
ax2.set_ylabel("Output (y)", fontsize=12)
ax2.set_xlim(-5.5, 5.5)
ax2.set_ylim(y_min, y_max)
ax2.legend(loc="best", frameon=True, shadow=True)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gp_prior_random_draws.png", dpi=150, bbox_inches="tight")
plt.show()

# ----- Plots 3-7: Posterior updates after adding data points -----
X_train_full, y_train_full, ell_data, sf_data, sn_data = generate_gp_plots_data()
num_points_to_add = 10

posteriors = []
posterior_y_min = np.inf
posterior_y_max = -np.inf

for n in range(1, num_points_to_add + 1):
    X_n = X_train_full[:n]
    y_n = y_train_full[:n]

    mu_n, cov_n = gp_posterior(X_n, y_n, X_prior, ell_data, sf_data, sn_data)
    sd_n = np.sqrt(np.diag(cov_n))
    posteriors.append((n, X_n, y_n, mu_n, sd_n))

    posterior_y_min = min(
        posterior_y_min,
        np.min(mu_n - sigma * sd_n),
        np.min(y_n - sn_data),
    )
    posterior_y_max = max(
        posterior_y_max,
        np.max(mu_n + sigma * sd_n),
        np.max(y_n + sn_data),
    )

posterior_margin = 0.1 * (posterior_y_max - posterior_y_min)
posterior_y_min -= posterior_margin
posterior_y_max += posterior_margin

for n, X_n, y_n, mu_n, sd_n in posteriors:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(
        X_prior,
        mu_n - sigma * sd_n,
        mu_n + sigma * sd_n,
        alpha=0.35,
        label="95% confidence interval",
    )
    ax.plot(X_prior, mu_n, lw=2.2, label="Posterior mean")
    ax.errorbar(
        X_n,
        y_n,
        yerr=sn_data,
        fmt="o",
        ms=8,
        elinewidth=2,
        capsize=4,
        label="Observed data",
        color="steelblue",
    )

    point_label = "point" if n == 1 else "points"
    ax.set_title(
        f"GP Posterior After Adding {n} Data {point_label}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Input (x)", fontsize=12)
    ax.set_ylabel("Output (y)", fontsize=12)
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(posterior_y_min, posterior_y_max)
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"gp_posterior_added_points_{n}.png", dpi=150, bbox_inches="tight")
    plt.show()

# ----- Animation: GIF of posterior updates -----
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
ci_color = colors[0]
mean_color = colors[0]

fig_anim, ax_anim = plt.subplots(figsize=(10, 6))


def draw_frame(frame_data):
    n, X_n, y_n, mu_n, sd_n = frame_data
    ax_anim.cla()
    ax_anim.fill_between(
        X_prior,
        mu_n - sigma * sd_n,
        mu_n + sigma * sd_n,
        color=ci_color,
        alpha=0.35,
        label="95% confidence interval",
    )
    ax_anim.plot(X_prior, mu_n, color=mean_color,
                 lw=2.2, label="Posterior mean")
    ax_anim.errorbar(
        X_n,
        y_n,
        yerr=sn_data,
        fmt="o",
        ms=8,
        elinewidth=2,
        capsize=4,
        label="Observed data",
        color="steelblue",
    )
    point_label = "point" if n == 1 else "points"
    ax_anim.set_title(
        f"GP Posterior After Adding {n} Data {point_label}",
        fontsize=14,
        fontweight="bold",
    )
    ax_anim.set_xlabel("Input (x)", fontsize=12)
    ax_anim.set_ylabel("Output (y)", fontsize=12)
    ax_anim.set_xlim(-5.5, 5.5)
    ax_anim.set_ylim(posterior_y_min, posterior_y_max)
    ax_anim.legend(loc="best", frameon=True, shadow=True)
    ax_anim.grid(True, alpha=0.3)


anim = animation.FuncAnimation(
    fig_anim,
    draw_frame,
    frames=posteriors,
    interval=4000,  # 4 seconds per frame
    repeat=True,
)

plt.tight_layout()
anim.save("gp_posterior_animation.gif", writer="pillow", fps=0.25, dpi=100)
print("Saved gp_posterior_animation.gif")
plt.close(fig_anim)
