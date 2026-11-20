import json
import sys
import warnings

from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from data import BETA_DEFAULT, N_DEFAULT, SEED, bayes_model, make_grid
from plot import plot_mean_curves
from protocol import collect_predictions, decompose

warnings.filterwarnings("ignore", category=ConvergenceWarning)

G = 40
_, _, X_TEST = make_grid(G)


M = {"Tree": 100, "kNN": 100, "MLP": 50}

# Default params used in exercice 2.3
DEFAULT_PARAM = {"Tree": 6, "kNN": 45, "MLP": 350}


def make_model(method, param):
    if method == "Tree":
        return DecisionTreeClassifier(max_depth=param)
    if method == "kNN":
        return KNeighborsClassifier(n_neighbors=param)
    if method == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=param, solver="lbfgs", alpha=1e-5, max_iter=3000
        )
    raise ValueError(method)


COMPLEXITY = {
    "Tree": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20],
    "kNN": [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 80, 100, 125],
    "MLP": [1, 2, 3, 5, 10, 20, 50, 100, 200, 350],
}
N_VALUES = [50, 100, 250, 500, 1000]
BETA_VALUES = [0.5, 1, 2, 5, 10, 20, 50]


def mean_decomposition(method, param, N=N_DEFAULT, beta=BETA_DEFAULT):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    p = bayes_model(X_TEST, beta)
    P = collect_predictions(
        make_model(method, param), M[method], X_TEST, N=N, beta=beta, seed=SEED
    )
    _, bias2, var, noise, err = decompose(P, p)
    return {
        "error": err.mean(),
        "noise": noise.mean(),
        "bias2": bias2.mean(),
        "variance": var.mean(),
    }


def run(methods):
    jobs = []
    for m in methods:
        for c in COMPLEXITY[m]:
            jobs.append((m, "complexity", c, {"param": c}))
        for n in N_VALUES:
            jobs.append((m, "N", n, {"param": DEFAULT_PARAM[m], "N": n}))
        for b in BETA_VALUES:
            jobs.append((m, "beta", b, {"param": DEFAULT_PARAM[m], "beta": b}))

    out = Parallel(n_jobs=-1, verbose=5)(
        delayed(mean_decomposition)(m, **kw) for (m, _, _, kw) in jobs
    )

    try:
        with open("q2_4_results.json") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
    for (m, exp, x, _), r in zip(jobs, out):
        results.setdefault(m, {}).setdefault(exp, {})[str(x)] = r
    with open("q2_4_results.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "plot"
    methods = {"tree": ["Tree"], "knn": ["kNN"], "mlp": ["MLP"], "all": ["Tree", "kNN", "MLP"]}
    if what in methods:
        run(methods[what])
    elif what != "plot":
        sys.exit(f"unknown argument {what!r}, expected one of: plot, {', '.join(methods)}")
    plot_mean_curves()
