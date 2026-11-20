
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

# every figure is saved in this folder (relative to where the script is run)
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# x2 = sin(pi*x1) curve
CURVE = np.linspace(-1, 1, 400)

def make_cmaps():
    """Blue-to-orange colormap for probability maps (blue = 0, orange = 1)."""
    blues = mpl.colormaps["Blues_r"](np.linspace(0.25, 1, 128))
    oranges = mpl.colormaps["Oranges"](np.linspace(0, 0.75, 128))
    return ListedColormap(np.vstack((blues, oranges)), name="BLueOrange")

PROBA_CMAP = make_cmaps()
AMOUNT_CMAP = mpl.colormaps["Purples"]

def plot_map(ax, X1, X2, values, title, cmap, vmax):
    """Exercise 2.3: draw `values` as a filled contour map on the grid (X1, X2)
    in `ax`, with the Bayes boundary x2 = sin(pi*x1) overlaid. Returns the map."""
    Z = np.reshape(values, X1.shape)
    vmax = max(vmax, 1e-12)
    levels = np.linspace(0, vmax, 21)
    filled = ax.contourf(X1,X2,Z, levels=levels, cmap=cmap)

    ax.plot(CURVE, np.sin(np.pi * CURVE), color="black", linewidth=1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.figure.colorbar(filled, ax=ax, shrink=0.85, ticks=MaxNLocator(6))
    return Z

def plot_method_maps(name, X1, X2, p, avg, bias2, var, vmax_bias, vmax_var, fname):
    """Exercise 2.3: save a 2x2 figure with the Bayes model, the average model,
    the squared bias and the variance maps of one method."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5), layout="constrained")
    plot_map(axes[0, 0], X1, X2, p, "Bayes model $h_B(x)$", PROBA_CMAP, 1.0)
    plot_map(axes[0,1], X1, X2, avg, f"{name}: average model", PROBA_CMAP, 1.0)
    plot_map(axes[1,0], X1, X2, bias2, f"{name}: squared bias", AMOUNT_CMAP, vmax_bias)
    plot_map(axes[1,1], X1, X2, var, f"{name}: variance", AMOUNT_CMAP, vmax_var)
    fig.savefig(fname, transparent=True)
    plt.close(fig)

def plot_all(X1, X2, p, results, same_scale=False):
    """Exercise 2.3: call plot_method_maps for each method in `results`
    (name -> (avg, bias2, var, noise, err)). If `same_scale`, all methods share
    the same colour scale for bias and variance."""
    max_bias_all = 0.0
    max_var_all = 0.0
    for avg, bias2,var,noise,err in results.values():
        max_bias_all = max(max_bias_all, bias2.max())
        max_var_all = max(max_var_all, var.max())

    for name, (avg, bias2,var,noise,err) in results.items():
        if same_scale:
            vmax_bias, vmax_var =max_bias_all, max_var_all
        else:
            vmax_bias, vmax_var =bias2.max(), var.max()
        fname = f"{FIG_DIR}/q2_3_{name}.pdf"
        plot_method_maps(name, X1, X2, p, avg, bias2, var, vmax_bias, vmax_var, fname)
        print("saved", fname)
        print(f"{name}: vmax_bias: {vmax_bias}, vmax_var: {vmax_var}")

# EXERCICE 2.4

XLABEL = {
    "complexity": {
        "Tree": "maximum depth",
        "kNN": "$k$ (number of neighbours)",
        "MLP": "number of hidden neurons",
    },
    "N": "learning sample size $N$",
    "beta": r"noise parameter $\beta$",
}
STYLE = {
    "error": ("Expected error", "k", "-"),
    "noise": ("Residual error", "tab:gray", "--"),
    "bias2": ("Squared bias", "tab:blue", "-"),
    "variance": ("Variance", "tab:red", "-"),
}


def plot_mean_curves(fname="q2_4_results.json"):
    """Exercise 2.4: plot the mean error, residual error, squared bias and
    variance versus complexity, N and beta (one figure each, one subplot per
    method) from the JSON results written by q2_4.run."""
    with open(fname) as f:
        results = json.load(f)
    methods = [m for m in ["MLP", "kNN", "Tree"] if m in results]
    for exp in ["complexity", "N", "beta"]:
        fig, axes = plt.subplots(
            1,
            len(methods),
            figsize=(5 * len(methods), 4),
            layout="constrained",
            squeeze=False,
        )
        for ax, m in zip(axes[0], methods):
            data = results[m][exp]
            xs = sorted(data, key=float)
            x = [float(v) for v in xs]
            for key, (label, color, ls) in STYLE.items():
                ax.plot(
                    x,
                    [data[v][key] for v in xs],
                    ls,
                    color=color,
                    marker="o",
                    ms=3,
                    label=label,
                )
            if exp != "complexity" or m in ("kNN", "MLP"):
                ax.set_xscale("log")
            lab = XLABEL[exp][m] if exp == "complexity" else XLABEL[exp]
            ax.set_xlabel(lab)
            ax.set_title(m)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("mean value over the input space")
        axes[0][0].legend()
        fig.savefig(f"{FIG_DIR}/q2_4_{exp}.pdf")
        plt.close(fig)
        print("saved", f"{FIG_DIR}/q2_4_{exp}.pdf")



def optimum(curve):
    """Exercise 2.5: return the k minimising the mean expected error, and that error."""
    ks = sorted(curve, key=int)
    errs = [curve[k]["error"] for k in ks]
    i = int(np.argmin(errs))
    return int(ks[i]), errs[i]


def plot_knn_optimum(fname="q2_5_results.json"):
    """Exercise 2.5: for kNN, plot the mean expected error versus k for each
    value of N (resp. beta), and the optimal k versus N (resp. beta)."""
    with open(fname) as f:
        results = json.load(f)

    label = {"N": "N", "beta": r"\beta"}
    for exp in ["N", "beta"]:
        values = sorted(results[exp], key=float)
        cmap = plt.get_cmap("viridis")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")

        # Gauche : erreur attendue en fonction de k, une courbe par valeur de N ou beta
        opt = []
        for i, v in enumerate(values):
            curve = results[exp][v]
            ks = sorted(curve, key=int)
            color = cmap(i / max(len(values) - 1, 1))
            ax1.plot([int(k) for k in ks], [curve[k]["error"] for k in ks],
                     "-", color=color, label=f"${label[exp]}={float(v):g}$")
            k_opt, e_opt = optimum(curve)
            ax1.plot(k_opt, e_opt, "o", color=color, ms=7, mec="k")
            opt.append((float(v), k_opt))
        ax1.set_xscale("log")
        ax1.set_xlabel("$k$ (number of neighbours)")
        ax1.set_ylabel("mean expected error")
        ax1.set_title("Expected error vs $k$ (dots: optimum)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # Droite : k optimal en fonction de N ou beta
        x, y = zip(*opt)
        ax2.plot(x, y, "o-", color="k")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(f"${label[exp]}$")
        ax2.set_ylabel("optimal $k$")
        ax2.set_title(f"Optimal $k$ vs ${label[exp]}$")
        ax2.grid(alpha=0.3, which="both")

        fig.savefig(f"{FIG_DIR}/q2_5_{exp}.pdf")
        plt.close(fig)
        print(exp, "optimal k:", opt)


BAGGING_STYLE = {"error": ("Expected error", "k"),
         "bias2": ("Squared bias", "tab:blue"),
         "variance": ("Variance", "tab:red")}


def plot_bagging(order, fname="q2_6_results.json"):
    """Exercise 2.6: for each base model (in the given `order`), plot the mean
    expected error, squared bias and variance versus the number of bagged
    models B, with the values of the model without bagging as dotted lines."""
    with open(fname) as f:
        results = json.load(f)
    names = [n for n in order if n in results]
    ncol = 3
    nrow = (len(names) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow),
                             layout="constrained", squeeze=False)
    for ax, n in zip(axes.flat, names):
        data = results[n]
        Bs = sorted((int(b) for b in data if b != "0"))
        for key, (label, color) in BAGGING_STYLE.items():
            ax.plot(Bs, [data[str(b)][key] for b in Bs], "o-", color=color,
                    ms=3, label=label)
            # valeur du modèle seul, sans bagging
            ax.axhline(data["0"][key], color=color, ls=":", lw=1)
        ax.axhline(data["0"]["noise"], color="tab:gray", ls="--", label="Residual error")
        ax.set_xscale("log")
        ax.set_xlabel("number of bagged models $B$")
        ax.set_title(n)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        print(n, {b: {k: round(v, 4) for k, v in data[b].items()} for b in sorted(data, key=int)})
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    axes[0][0].set_ylabel("mean value over the input space")
    axes[0][0].legend(fontsize=8, title="dotted: without bagging", title_fontsize=8)
    fig.savefig(f"{FIG_DIR}/q2_6.pdf")
    plt.close(fig)


MISCLASS_XLABEL = {"kNN": "$k$ (number of neighbours)", "Tree": "maximum depth"}


def plot_misclassification(fname="q2_7_results.json"):
    """Exercise 2.7: for kNN and Tree, compare the mean misclassification error
    with the mean squared error versus complexity, with their optima (stars)."""
    with open(fname) as f:
        results = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, m in zip(axes, ["kNN", "Tree"]):
        data = results[m]
        xs = sorted(data, key=int)
        x = np.array([int(v) for v in xs])
        mis = np.array([data[v]["misclassification"] for v in xs])
        sq = np.array([data[v]["squared_error"] for v in xs])
        bayes = data[xs[0]]["bayes_misclassification"]
        resid = data[xs[0]]["residual_error"]

        ax.plot(x, mis, "o-", color="tab:green", ms=3, label="Misclassification error")
        ax.axhline(bayes, color="tab:green", ls=":", label="Bayes misclassification error")
        ax.plot(x, sq, "o-", color="k", ms=3, label="Squared error (Q2.4)")
        ax.axhline(resid, color="tab:gray", ls="--", label="Residual error (squared)")
        for y, c in [(mis, "tab:green"), (sq, "k")]:
            i = int(np.argmin(y))
            ax.plot(x[i], y[i], "*", color=c, ms=14, mec="k")
            print(m, "optimum", "misclassification" if c != "k" else "squared", x[i], round(y[i], 4))
        if m == "kNN":
            ax.set_xscale("log")
        ax.set_xlabel(MISCLASS_XLABEL[m])
        ax.set_ylabel("mean value over the input space")
        ax.set_title(m + " (stars: optimum)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.savefig(f"{FIG_DIR}/q2_7.pdf")
    plt.close(fig)


if __name__ == "__main__":
    from data import make_grid, bayes_model, BETA_DEFAULT
    X1, X2, X_test = make_grid(100)
    p = bayes_model(X_test, BETA_DEFAULT)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    plot_map(axes[0], X1, X2, p, "Bayes model $h_B(x)$", PROBA_CMAP, 1.0)
    plot_map(axes[1], X1, X2, p * (1-p), "Residual error", AMOUNT_CMAP, 0.25)
    fig.savefig(f"{FIG_DIR}/check_bayes.pdf")
    plt.show()


