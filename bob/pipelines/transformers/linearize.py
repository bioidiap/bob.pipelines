import numpy as np

from sklearn.preprocessing import FunctionTransformer

from ..wrappers import wrap


def linearize(X):
    X = np.asarray(X)
    return np.reshape(X, (X.shape[0], -1))


class Linearize(FunctionTransformer):
    """Extracts features by simply concatenating all elements of the data into
    one long vector."""

    def __init__(self, **kwargs):
        super().__init__(func=linearize, **kwargs)


def SampleLinearize(**kwargs):
    return wrap([Linearize, "sample"], **kwargs)


def CheckpointSampleLinearize(**kwargs):
    return wrap([Linearize, "sample", "checkpoint"], **kwargs)
