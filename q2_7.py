import json
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from data import BETA_DEFAULT, N_DEFAULT, SEED, bayes_model, make_grid
from plot import plot_misclassification
from protocol import collect_predictions, decompose

G = 40
_, _, X_TEST = make_grid(G)
M = 100

COMPLEXITY = {
    "kNN": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 45, 60, 80, 100, 125, 150, 200],
    "Tree": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20],
}


def make_model(method, param):
    if method == "kNN":
        return KNeighborsClassifier(n_neighbors=param)
    return DecisionTreeClassifier(max_depth=param)


def misclassification(P, p):
    """Erreur de classification moyenne (sur les M modèles et les T points)."""
    pred = P > 0.5                         
    err = np.where(pred, 1 - p, p)          
    return err.mean()


def evaluate(method, param):
    p = bayes_model(X_TEST, BETA_DEFAULT)
    P = collect_predictions(make_model(method, param), M, X_TEST,
                            N=N_DEFAULT, beta=BETA_DEFAULT, seed=SEED)
    _, bias2, var, noise, err = decompose(P, p)
    return {"misclassification": misclassification(P, p),
            "squared_error": err.mean(),
            "bayes_misclassification": np.minimum(p, 1 - p).mean(),
            "residual_error": noise.mean()}


def run():
    jobs = [(m, c) for m in COMPLEXITY for c in COMPLEXITY[m]]
    out = Parallel(n_jobs=-1, verbose=5)(delayed(evaluate)(m, c) for m, c in jobs)
    results = {}
    for (m, c), r in zip(jobs, out):
        results.setdefault(m, {})[str(c)] = r
    with open("q2_7_results.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if what == "run":
        run()
    elif what != "plot":
        sys.exit(f"unknown argument {what!r}, expected one of: plot, run")
    plot_misclassification()
