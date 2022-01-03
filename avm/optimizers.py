import warnings
import numpy as np

#from .loss import anchor_distance_loss as anchor_loss
from .loss import anchor_entropy_loss as anchor_loss
from .utils import gradient_approx, hessian_approx, class_centroids, convergence


def gradient_descent(X_train, y_train, distance_fn=None, *,
                     tol=1e-4, max_iter=100, eta=1., random_state=None):

    rng = np.random.default_rng(random_state)

    shape = (y_train.shape[1], X_train.shape[1])

    anchor = class_centroids(X_train, y_train)
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

        grad = gradient_approx(loss, anchor.flatten())
        anchor -= eta * grad.reshape(shape)

        if convergence(grad, tol) and (count >= 1):
            return anchor

        if count >= max_iter:
            warnings.warn('Not converged: Reached max iterations.')
            return anchor

        count += 1


def newton_raphson(X_train, y_train, distance_fn=None, *,
                   tol=1e-4, max_iter=10, eta=1.):

    shape = (y_train.shape[1], X_train.shape[1])

    anchor = class_centroids(X_train, y_train)

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

        if convergence(grad, tol) and (count >= 1):
            return anchor

        if count >= max_iter:
            warnings.warn('Not converged: Reached max iterations.')
            return anchor

        count += 1


def batch_grad_desc(X_train, y_train, distance_fn=None, *,
                    tol=1e-4, max_iter=10, eta=1., batch_size=1, random_state=None):

    rng = np.random.default_rng(random_state)

    shape = (y_train.shape[1], X_train.shape[1])
    length = X_train.shape[0]

    anchor = class_centroids(X_train, y_train)
    anchor += rng.normal(size=shape, scale=1e-4, loc=0.)

    global_center = class_centroids(X_train, np.ones((X_train.shape[0], 1)))

    count = 0
    while True:

        for _ in range(0, length-batch_size+1, batch_size):
            idx = rng.choice(length, size=batch_size, replace=False)

            def loss(_params):
                return anchor_loss(
                    X=X_train[idx],
                    y=y_train[idx],
                    A=_params.reshape(shape),
                    func=distance_fn,
                )

            grad = gradient_approx(loss, anchor.flatten())
            anchor -= eta * grad.reshape(shape)

            if convergence(grad, tol) and (count >= 1):
                return anchor

        if count >= max_iter:
            warnings.warn('Not converged: Reached max iterations.')
            return anchor

        count += 1
