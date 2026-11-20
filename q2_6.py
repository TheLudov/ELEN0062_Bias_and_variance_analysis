import json
import sys
import warnings

from joblib import Parallel, delayed
from sklearn.ensemble import BaggingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from data import BETA_DEFAULT, N_DEFAULT, SEED, bayes_model, make_grid
from plot import plot_bagging
from protocol import collect_predictions, decompose

G = 40
_, _, X_TEST = make_grid(G)

BASE = {
    "Tree (d=6)": lambda: DecisionTreeClassifier(max_depth=6),
    "Tree (fully grown)": lambda: DecisionTreeClassifier(),
    "kNN (k=45)": lambda: KNeighborsClassifier(n_neighbors=45),
    "kNN (k=1)": lambda: KNeighborsClassifier(n_neighbors=1),
    "MLP (h=350)": lambda: MLPClassifier(hidden_layer_sizes=350, solver="lbfgs",
                                         alpha=1e-5, max_iter=3000),
}
FAST = ["Tree (d=6)", "Tree (fully grown)", "kNN (k=45)", "kNN (k=1)"]
SLOW = ["MLP (h=350)"]

B_VALUES = {"fast": [0, 1, 2, 5, 10, 20, 50, 100], "mlp": [0, 1, 2, 5, 10, 20]}
M = {"fast": 100, "mlp": 30}


def make_model(name, B):
    if B == 0:
        return BASE[name]()
    return BaggingClassifier(estimator=BASE[name](), n_estimators=B)


def evaluate(name, B, m):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    p = bayes_model(X_TEST, BETA_DEFAULT)
    P = collect_predictions(make_model(name, B), m, X_TEST,
                            N=N_DEFAULT, beta=BETA_DEFAULT, seed=SEED)
    _, bias2, var, noise, err = decompose(P, p)
    return {"error": err.mean(), "noise": noise.mean(),
            "bias2": bias2.mean(), "variance": var.mean()}


def run(group):
    names = FAST if group == "fast" else SLOW
    jobs = [(n, B) for n in names for B in B_VALUES[group]]
    out = Parallel(n_jobs=-1, verbose=5)(
        delayed(evaluate)(n, B, M[group]) for n, B in jobs)
    try:
        with open("q2_6_results.json") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
    for (n, B), r in zip(jobs, out):
        results.setdefault(n, {})[str(B)] = r
    with open("q2_6_results.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "plot"
    groups = {"fast": ["fast"], "mlp": ["mlp"], "all": ["fast", "mlp"]}
    if what in groups:
        for g in groups[what]:
            run(g)
    elif what != "plot":
        sys.exit(f"unknown argument {what!r}, expected one of: plot, {', '.join(groups)}")
    plot_bagging(list(BASE))
