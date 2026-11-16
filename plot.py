
import numpy as np
import matplotlib as mpl
from fontTools.merge import layout
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# x2 = sin(pi*x1) curve
CURVE = np.linspace(-1, 1, 400)

def make_cmaps():
    blues = mpl.colormaps["Blues_r"](np.linspace(0.25, 1, 128))
    oranges = mpl.colormaps["Oranges"](np.linspace(0, 0.75, 128))
    return ListedColormap(np.vstack((blues, oranges)), name="BLueOrange")

PROBA_CMAP = make_cmaps()
AMOUNT_CMAP = mpl.colormaps["Purples"]

def plot_map(ax, X1, X2, values, title, cmap, vmax):
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
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5), layout="constrained")
    plot_map(axes[0, 0], X1, X2, p, "Bayes model $h_B(x)$", PROBA_CMAP, 1.0)
    plot_map(axes[0,1], X1, X2, avg, f"{name}: average model", PROBA_CMAP, 1.0)
    plot_map(axes[1,0], X1, X2, bias2, f"{name}: squared bias", AMOUNT_CMAP, vmax_bias)
    plot_map(axes[1,1], X1, X2, var, f"{name}: variance", AMOUNT_CMAP, vmax_var)
    fig.savefig(fname, transparent=True)
    plt.close(fig)

def plot_all(X1, X2, p, results, folder="figures", same_scale=False):
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
        fname = f"q2_3_{name}.pdf"
        plot_method_maps(name, X1, X2, p, avg, bias2, var, vmax_bias, vmax_var, fname)
        print("saved", fname)


if __name__ == "__main__":
    from data import make_grid, bayes_model, BETA_DEFAULT
    X1, X2, X_test = make_grid(100)
    p = bayes_model(X_test, BETA_DEFAULT)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    plot_map(axes[0], X1, X2, p, "Bayes model $h_B(x)$", PROBA_CMAP, 1.0)
    plot_map(axes[1], X1, X2, p * (1-p), "Residual error", AMOUNT_CMAP, 0.25)
    fig.savefig("check_bayes.pdf")
    plt.show()


