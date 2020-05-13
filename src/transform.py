from functools import partial

import numpy as np
from scipy.optimize import minimize


def sig(x):
    return 1 / (1 + np.exp(-np.array(x)))




def diff_score(scores):
    return np.expand_dims(scores, axis=2) - np.expand_dims(scores, axis=1)


def pairwise_probability(scores):
    return sig(diff_score(scores))


a = [0.1, 0.5, 0.4]

A = [[5 / 6, -1 / 6, 0],
     [4 / 5, 0., -1 / 5],
     [0., 4 / 9, -5 / 9]]


def objective(A, x):
    return np.sum(np.array(A).dot(x) ** 2)


def con(x):
    return np.array([1] * len(x)).dot(x) - 1


def optimize(A):
    x0 = [1 / np.shape(A)[1]] * np.shape(A)[1]
    return minimize(partial(objective, A), x0, constraints={'type': 'eq', 'fun': con})


def build_diags(pairwise_probabilities):
    matrices =[]
    for i in range(len(pairwise_probabilities)-1):
        diag = []
        for l in range(i+1,len(pairwise_probabilities[i])):
            diag.append(pairwise_probabilities[i][l])
        matrices.append(np.hstack([np.zeros([len(diag), i]), np.expand_dims(1-np.array(diag),axis=1), -np.diag(diag)]))

    return np.concatenate(matrices, axis=0)



