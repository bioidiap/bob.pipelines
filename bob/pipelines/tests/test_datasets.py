#!/usr/bin/env python
# coding=utf-8

"""Test code for datasets"""

import os

import numpy as np
import pkg_resources
import pytest

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from bob.pipelines.datasets import FileListDatabase
from bob.pipelines.transformers import Str_To_Types


def iris_data_tranform(samples):
    for s in samples:
        data = np.array([s.sepal_length, s.sepal_width, s.petal_length, s.petal_width])
        s.data = data
    return samples


def test_iris_list_database():
    dataset_protocol_path = pkg_resources.resource_filename(
        __name__, os.path.join("data", "iris_database")
    )

    database = FileListDatabase(dataset_protocol_path, None)
    assert database.protocol == "default"
    assert database.protocols() == ["default"]
    assert database.groups() == ["test", "train"]
    with pytest.raises(ValueError):
        database.protocol = "none"

    samples = database.samples()
    assert len(samples) == 150
    assert samples[0].data is None
    assert samples[0].sepal_length == "5"
    assert samples[0].petal_width == "0.2"
    assert samples[0].target == "Iris-setosa"

    with pytest.raises(ValueError):
        database.samples(groups="random")

    database.transformer = make_pipeline(
        Str_To_Types(
            fieldtypes=dict(
                sepal_length=float,
                sepal_width=float,
                petal_length=float,
                petal_width=float,
            )
        ),
        FunctionTransformer(iris_data_tranform),
    )
    samples = database.samples(groups="train")
    assert len(samples) == 75
    np.testing.assert_allclose(samples[0].data, [5.1, 3.5, 1.4, 0.2])
    assert samples[0].sepal_length == 5.1
    assert samples[0].petal_width == 0.2
    assert samples[0].target == "Iris-setosa"
