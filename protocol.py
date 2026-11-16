import numpy as np
from sklearn.base import clone

from data import N_DEFAULT, BETA_DEFAULT, make_learning_sample, SEED


def collect_predictions(template, M, X_test, N=N_DEFAULT, beta=BETA_DEFAULT, seed=SEED):
    seed_sequence = np.random.SeedSequence(seed).spawn(M)
    T = X_test.shape[0]
    P = np.zeros((M, T))
    for i in range(M):
        data_seed, model_seed = seed_sequence[i].generate_state(2)
        X, y = make_learning_sample(N, beta, data_seed)
        model = clone(template)
        if "random_state" in model.get_params().keys():
            model.random_state = model_seed
        model.fit(X, y)
        proba = model.predict_proba(X_test)
        if 1 in model.classes_:
            c = np.where(model.classes_ == 1)[0][0]
            P[i, :] = proba[:, c]
        else:
            P[i, :] = 0
    return P

def decompose(P, p):
    avg = np.mean(P, axis=0)
    var = np.mean(np.power((P-avg),2),axis=0)
    bias2 = np.power(p-avg,2)
    noise = p*(1-p)
    err = noise + bias2 + var
    return avg,bias2,var,noise,err