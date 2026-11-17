from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from data import make_grid, bayes_model, BETA_DEFAULT, SEED
from plot import plot_all
from protocol import collect_predictions, decompose

if '__main__' == __name__:
    X1, X2, X_test = make_grid(G=100)
    p = bayes_model(X_test, beta=BETA_DEFAULT)
    methods = {
        "MLP": MLPClassifier(hidden_layer_sizes=350, solver="lbfgs", alpha=1e-5, max_iter=3000),
        "kNN": KNeighborsClassifier(n_neighbors=45),
        "Tree": DecisionTreeClassifier(max_depth=6)
    }
    results= {}
    for (name, classifier) in methods.items():
        P = collect_predictions(classifier, M=200, X_test=X_test, beta=BETA_DEFAULT, seed=SEED)
        results[name] = decompose(P, p)
    plot_all(X1, X2, p, results)