import numpy as np
from sklearn.utils import check_random_state

N_DEFAULT = 250
BETA_DEFAULT = 5
SEED = 42

def bayes_model(X, beta):
    (x1, x2) = X[:,0], X[:,1]
    z = -beta * (x2 - np.sin(np.pi * x1))
    return np.divide(1, (1 + np.exp(z)))

def make_learning_sample(N= N_DEFAULT, beta=BETA_DEFAULT,random_state=None):
    rng = check_random_state(random_state)
    X = rng.uniform(-1, 1, (N, 2))
    p = bayes_model(X, beta)
    # With n=1 binomial=bernouilli
    y = rng.binomial(1, p, N)
    return X, y

def make_grid(G):
    axis = np.linspace(-1, 1, G)
    X1, X2 = np.meshgrid(axis, axis)
    X_test = np.column_stack((X1.flatten(), X2.flatten()))
    return X1, X2, X_test
