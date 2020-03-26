#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


from bob.pipelines.mixins import CheckpointMixin, SampleMixin
from sklearn.preprocessing import FunctionTransformer
import numpy as np


def linearize(X):
    X = np.asarray(X)
    return np.reshape(X, (X.shape[0], -1))


class Linearize(FunctionTransformer):
    """Extracts features by simply concatenating all elements of the data into one long vector.
    """

    def __init__(self, **kwargs):
        super().__init__(func=linearize, **kwargs)


class SampleLinearize(SampleMixin, Linearize):
    pass


class CheckpointSampleLinearize(CheckpointMixin, SampleMixin, Linearize):
    pass
