from sklearn.decomposition import PCA

from ..wrappers import wrap


def SamplePCA(**kwargs):
    """Enables SAMPLE handling for :any:`sklearn.decomposition.PCA`"""
    return wrap([PCA, "sample"], **kwargs)


def CheckpointSamplePCA(**kwargs):
    """Enables SAMPLE and CHECKPOINTIN handling for
    :any:`sklearn.decomposition.PCA`"""
    return wrap([PCA, "sample", "checkpoint"], **kwargs)
