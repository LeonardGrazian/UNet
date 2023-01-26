
import numpy as np


class PILToNumpy(object):
    def __call__(self, X):
        return np.array(X)


class Shift(object):
    def __init__(self, shift_amount):
        self.shift_amount = shift_amount

    def __call__(self, X):
        return X + self.shift_amount


class OneHot(object):
    def __init__(self, onehot_dim, onehot_min):
        self.onehot_dim = onehot_dim
        self.onehot_min = onehot_min

    def __call__(self, X):
        if len(X.shape) == 3:
            assert not np.any(X[:, :, 0] != X[:, :, 1])
            assert not np.any(X[:, :, 0] != X[:, :, 2])
            assert not np.any(X[:, :, 1] != X[:, :, 2])
            X = X[:, :, 0]

        z = np.zeros((X.size, self.onehot_dim))
        z[np.arange(X.size), X.flatten() - self.onehot_min] = 1
        z = z.reshape(X.shape + (self.onehot_dim,))
        return np.moveaxis(z, 2, 0)
