#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


from bob.pipelines.mixins import CheckpointMixin, SampleMixin
from sklearn.decomposition import PCA


class SamplePCA(SampleMixin, PCA):
    """
    Enables SAMPLE handling for https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """

    pass


class CheckpointSamplePCA(CheckpointMixin, SampleMixin, PCA):
    """
    Enables SAMPLE and CHECKPOINTIN handling for https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """

    pass
