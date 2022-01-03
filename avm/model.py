import numpy as np

from .optimizers import gradient_descent, newton_raphson, batch_grad_desc
from .base import AVMBase


class AnchoringVectorClassifier(AVMBase):

    def __init__(self, *,
                 metric='euclidean',
                 p=None,
                 kernel=None,
                 coef0=0.0,
                 gamma=None,
                 degree=2,
                 eta=1.,
                 tol=1e-4,
                 max_iter=500,
                 solver='gradient-desc',
                 batch_size=1,
                 random_state=None):

        super().__init__(
            metric=metric,
            p=p,
            kernel=kernel,
            coef0=coef0,
            gamma=gamma,
            degree=degree,
            eta=eta,
            tol=tol,
            max_iter=max_iter,
            solver=solver,
            batch_size=batch_size,
            random_state=random_state,
        )

    def _decode_pred(self, pred):

        return self._labeler.inverse_transform(
            np.where(pred==pred.min(1).reshape(-1,1), 1, -1)
        )

    def _predict(self, X):

        distance = np.zeros((X.shape[0], self.anchor_.shape[0]))

        for c in range(self.anchor_.shape[0]):
            distance[:,c] = self.distance_fn(X, self.anchor_[c].reshape(1,-1)).flatten()

        return distance

    def fit(self, X, y):

        X, y = self._validate(X, y, fitting=True)

        X = self.kernel_fn.fit_transform(X)

        if self.solver == 'gradient-desc':
            self.anchor_ = gradient_descent(
                X_train=X,
                y_train=y,
                distance_fn=self.distance_fn,
                tol=self.tol,
                max_iter=self.max_iter,
                eta=self.eta,
                random_state=self.random_state
            )

        elif self.solver == 'newton-raphson':
            self.anchor_ = newton_raphson(
                X_train=X,
                y_train=y,
                distance_fn=self.distance_fn,
                tol=self.tol,
                max_iter=self.max_iter,
                eta=self.eta,
            )

        elif self.solver == 'batch-gradient-desc':
            self.anchor_ = batch_grad_desc(
                X_train=X,
                y_train=y,
                distance_fn=self.distance_fn,
                tol=self.tol,
                max_iter=self.max_iter,
                eta=self.eta,
                batch_size=self.batch_size,
                random_state=self.random_state,
            )

        else:
            raise ValueError(f'Unknown solver: {self.solver}')

        self._is_fit = True

        return self

    def predict(self, X):

        X = self._validate(X, fitting=False)

        distance = self._predict(
            self.kernel_fn.transform(X)
        )

        return self._decode_pred(distance)

    def predict_proba(self, X):

        X = self._validate(X, fitting=False)

        distance = self._predict(
            self.kernel_fn.transform(X)
        )

        return 1 - distance/distance.sum(1).reshape(-1,1)
