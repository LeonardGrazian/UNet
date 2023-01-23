
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
        assert len(X.shape) == 2

        z = np.zeros((X.size, self.onehot_dim))
        z[np.arange(X.size), X.flatten() - self.onehot_min] = 1
        z = z.reshape(X.shape + (self.onehot_dim,))
        return np.moveaxis(z, 2, 0)
