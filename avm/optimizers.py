import warnings
import numpy as np

from .loss import anchor_distance_loss, anchor_entropy_loss
from .utils import gradient_approx, hessian_approx

anchor_loss = anchor_entropy_loss # anchor_distance_loss


def gradient_descent(X_train, y_train, distance_fn=None, *,
                     tol=1e-4, max_iter=10, eta=1., random_state=None):

    rng = np.random.default_rng(random_state)

    shape = (y_train.shape[1], X_train.shape[1])

    anchor = np.zeros(shape)
    for c in range(shape[0]):
        anchor[c] = X_train[y_train[:,c]==1].mean(0)
    anchor += rng.normal(size=shape, scale=1e-4, loc=0.)

    def loss(_params):
        return anchor_loss(
            X=X_train,
            y=y_train,
            A=_params.reshape(shape),
            func=distance_fn,
        ) + 0.

    count = 0
    while True:

        for _ in range(X_train.shape[0]):

            grad = gradient_approx(loss, anchor.flatten())
            anchor -= eta * grad.reshape(shape)

            if np.mean(grad**2)**0.5 < tol and count >= 1:
                return anchor

        if count >= max_iter:
            warnings.warn('Not converged: Reached max iterations.')
            return anchor

        count += 1


def newton_raphson(X_train, y_train, distance_fn=None, *,
                   tol=1e-4, max_iter=10, eta=1.):

    shape = (y_train.shape[1], X_train.shape[1])

    anchor = np.zeros(shape)
    for c in range(shape[0]):
        anchor[c] = X_train[y_train[:,c]==1].mean(0)

    def loss(_params):
        return anchor_loss(
            X=X_train,
            y=y_train,
            A=_params.reshape(shape),
            func=distance_fn,
        ) + 0.

    count = 0
    while True:

        grad = gradient_approx(loss, anchor.flatten())
        hess = hessian_approx(loss, anchor.flatten())

        if np.linalg.matrix_rank(hess) < hess.shape[1]:
            warnings.warn('Not converged: Singular matrix.')
            return anchor

        anchor -= eta * np.linalg.inv(hess).dot(grad).reshape(shape)

        if np.mean(grad**2)**0.5 < tol and count >= 1:
            return anchor

        if count >= max_iter:
            warnings.warn('Not converged: Reached max iterations.')
            return anchor

        count += 1


def batch_grad_desc(X_train, y_train, distance_fn=None, *,
                    tol=1e-4, max_iter=10, eta=1., batch_size=1, random_state=None):

    rng = np.random.default_rng(random_state)

    shape = (y_train.shape[1], X_train.shape[1])

    anchor = np.zeros(shape)
    for c in range(shape[0]):
        anchor[c] = X_train[y_train[:,c]==1].mean(0)
    anchor += rng.normal(size=shape, scale=1e-4, loc=0.)

    count = 0
    while True:

        for _ in range(0, X_train.shape[0]-batch_size+1, batch_size):
            idx = rng.choice(X_train.shape[0], size=batch_size, replace=False)

            def loss(_params):
                return anchor_loss(
                    X=X_train[idx],
                    y=y_train[idx],
                    A=_params.reshape(shape),
                    func=distance_fn,
                ) + 0.

            grad = gradient_approx(loss, anchor.flatten())
            anchor -= eta * grad.reshape(shape)

            if np.mean(grad**2)**0.5 < tol and count >= 1:
                return anchor

        if count >= max_iter:
            warnings.warn('Not converged: Reached max iterations.')
            return anchor

        count += 1
