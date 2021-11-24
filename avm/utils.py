import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import FunctionTransformer

EPS = 1e-9 # 1.1e-16


class LabelEncoding(LabelBinarizer):

    def __init__(self):

        super().__init__(neg_label=-1, pos_label=1, sparse_output=False)

    def transform(self, y):

        Y = super().transform(y)

        if Y.shape[1] == 1:
            Y = np.c_[Y*-1, Y]
        return Y

    def inverse_transform(self, Y, threshold=None):

        if Y.shape[1] == 2:
            Y = Y[:,1]

        return super().inverse_transform(Y, threshold)


class DummyTransform(FunctionTransformer):

    def __init__(self):

        super().__init__(func=lambda x: x, inverse_func=lambda x: x)

        self.n_features_in_ = None

    def fit(self, X, y=None):

        self.n_features_in_ = X.shape[1]

        return super().fit(X, y)


def gradient_approx(func, params, eps=EPS):

    len_ = params.shape[0]
    grad = np.full(len_, np.nan)

    for idx in range(len_):
        e_i = np.zeros(len_)
        e_i[idx] = np.sqrt(eps)

        args1 = params.copy() + e_i
        args2 = params.copy() - e_i

        grad[idx] = (func(args1) + -func(args2)) / (2*e_i.sum())

    return grad


def hessian_approx(func, params, eps=EPS):

    len_ = params.shape[0]
    hess = np.full((len_, len_), np.nan)

    for idx in range(len_):
        e_i = np.zeros(len_)
        e_i[idx] = np.sqrt(eps)

        args1 = params.copy() + e_i
        args2 = params.copy() - e_i

        hess[idx, :] = (gradient_approx(func, args1) + -gradient_approx(func, args2)) / (2*e_i.sum())

    return hess


def linear_kernel(X):

    return np.array([X.dot(x) for x in X])
