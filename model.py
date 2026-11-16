from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def make_mlp(h, seed):
    return MLPClassifier(hidden_layer_sizes=h, solver="lbfgs", alpha=1e-5, max_iter=2000,random_state=seed)

def make_knn(k):
    return KNeighborsClassifier(n_neighbors=k)

def make_tree(d, seed):
    return DecisionTreeClassifier(max_depth=d, random_state=seed)