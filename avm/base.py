from abc import ABCMeta

import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.utils import check_X_y, check_array
from sklearn.kernel_approximation import Nystroem
from sklearn.base import BaseEstimator, ClassifierMixin

from .utils import LabelEncoding, DummyTransform


class AVMBase(BaseEstimator, ClassifierMixin, metaclass=ABCMeta):

    def __init__(self, *,
                 metric='euclidean',
                 p=None,
                 kernel=None,
                 coef0=0.0,
                 gamma=None,
                 degree=2,
                 eta=1.,
                 tol=1e-4,
                 max_iter=10,
                 solver='gradient-desc',
                 batch_size=1,
                 random_state=None):

        self.metric = metric
        self.p = p
        self.kernel = kernel
        self.coef0 = coef0
        self.gamma = gamma
        self.degree = degree
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter
        self.solver = solver
        self.batch_size = batch_size
        self.random_state = random_state

        self._is_fit = False
        self._labeler = LabelEncoding()

        self.anchor_ = None
        self.kernel_fn = None
        self.distance_fn = None

    @property
    def kernel_fn(self):

        return self._kernel_fn

    @kernel_fn.setter
    def kernel_fn(self, _):

        if self.kernel:
            self._kernel_fn = Nystroem(
                kernel=self.kernel,
                coef0=self.coef0,
                gamma=self.gamma,
                degree=self.degree,
                random_state=self.random_state,
            )

        else:
            self._kernel_fn = DummyTransform()

    @property
    def distance_fn(self):

        return self._distance_fn

    @distance_fn.setter
    def distance_fn(self, _):

        if self.p:
            self._distance_fn = DistanceMetric.get_metric(self.metric, p=self.p).pairwise

        else:
            self._distance_fn = DistanceMetric.get_metric(self.metric).pairwise

    def _validate(self, X, y=None, fitting=True):

        if fitting:
            return check_X_y(
                X=X,
                y=self._labeler.fit_transform(y),
                multi_output=True,
                order='C',
                dtype=np.float
            )

        if not self._is_fit:
            raise AssertionError('Model must be fit before calling predict.')

        return check_array(
            array=X,
            ensure_min_features=self.kernel_fn.n_features_in_,
            order='C',
            dtype=np.float,
        )

    def fit(self, X, y):

        raise NotImplementedError

    def predict(self, X):

        raise NotImplementedError

    def predict_proba(self, X):

        raise NotImplementedError
