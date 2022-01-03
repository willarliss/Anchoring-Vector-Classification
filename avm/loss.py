import numpy as np


def cross_entropy(y_true, y_prob, eps=1e-9):

    y_prob[y_prob==0] += eps
    y_prob[y_prob==1] -= eps
    y_prob = np.log(y_prob)

    return -(y_true*y_prob).sum(1)


def anchor_entropy_loss(X, y, A, func):

    dist = np.empty_like(y, dtype=float)
    for i in range(A.shape[0]):
        dist[:,i] = func(X, A[i].reshape(1,-1)).flatten()

    return cross_entropy(
        np.where(y==1, 1, 0),
        1 - dist/dist.sum(1).reshape(-1,1),
    ).mean()


def anchor_distance_loss(X, y, A, func):

    dist = y.copy().astype(float)

    for t in range(A.shape[0]):
        dist[:, t] *= func(X, A[t].reshape(1,-1)).squeeze()

    pos_mean = np.nanmean(np.where(dist<0, np.nan, dist), axis=1)
    neg_mean = np.nanmean(np.where(dist>=0, np.nan, dist), axis=1)

    return np.mean(pos_mean+neg_mean)
