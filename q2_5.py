import json
import sys

from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier

from data import BETA_DEFAULT, N_DEFAULT, SEED, bayes_model, make_grid
from plot import plot_knn_optimum
from protocol import collect_predictions, decompose

G = 40                     
_, _, X_TEST = make_grid(G)
M = 100                    

K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80,
            100, 125, 150, 200, 250, 300, 400]
N_VALUES = [50, 100, 250, 500, 1000, 2000]
BETA_VALUES = [0.5, 1, 2, 5, 10, 20, 50]


def mean_decomposition(k, N, beta):
    """Protocole de la Q2.2 pour un kNN -> moyennes sur la grille."""
    p = bayes_model(X_TEST, beta)
    P = collect_predictions(KNeighborsClassifier(n_neighbors=k), M, X_TEST,
                            N=N, beta=beta, seed=SEED)
    _, bias2, var, noise, err = decompose(P, p)
    return {"error": err.mean(), "noise": noise.mean(),
            "bias2": bias2.mean(), "variance": var.mean()}


def run():
    jobs = []   
    for N in N_VALUES:
        for k in K_VALUES:
            if k <= N:                       
                jobs.append(("N", N, k, N, BETA_DEFAULT))
    for b in BETA_VALUES:
        for k in K_VALUES:
            if k <= N_DEFAULT:
                jobs.append(("beta", b, k, N_DEFAULT, b))

    out = Parallel(n_jobs=-1, verbose=5)(
        delayed(mean_decomposition)(k, N, b) for (_, _, k, N, b) in jobs)

    results = {"N": {}, "beta": {}}
    for (exp, v, k, _, _), r in zip(jobs, out):
        results[exp].setdefault(str(v), {})[str(k)] = r
    with open("q2_5_results.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if what == "run":
        run()
    elif what != "plot":
        sys.exit(f"unknown argument {what!r}, expected one of: plot, run")
    plot_knn_optimum()
