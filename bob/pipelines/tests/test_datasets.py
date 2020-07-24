#!/usr/bin/env python
# coding=utf-8

"""Test code for datasets"""

import os

import nose.tools
import pkg_resources

from ..datasets.csv import CSVDataset
from ..datasets.json import JSONDataset
from ..sample import Sample


def _data_file(f):
    return pkg_resources.resource_filename(__name__, os.path.join("data", f))


def _raw_data_loader(context, d):
    return Sample(
        data=[
            float(d["sepal_length"]),
            float(d["sepal_width"]),
            float(d["petal_length"]),
            float(d["petal_width"]),
            d["species"][5:],
        ],
        key=(context["subset"] + str(context["order"])),
    )


def test_csv_loading():

    # tests if we can build a simple CSV loader for the Iris Flower dataset
    subsets = {
        "train": _data_file("iris-train.csv"),
        "test": _data_file("iris-test.csv"),
    }

    fieldnames = (
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    )

    dataset = CSVDataset(subsets, fieldnames, _raw_data_loader)
    dataset.check()

    data = dataset.subsets()

    nose.tools.eq_(len(data["train"]), 75)
    for k in data["train"]:
        for f in range(4):
            nose.tools.eq_(type(k.data[f]), float)
        nose.tools.eq_(type(k.data[4]), str)
        nose.tools.eq_(type(k.key), str)

    nose.tools.eq_(len(data["test"]), 75)
    for k in data["test"]:
        for f in range(4):
            nose.tools.eq_(type(k.data[f]), float)
        nose.tools.eq_(type(k.data[4]), str)
        assert k.data[4] in ("setosa", "versicolor", "virginica")
        nose.tools.eq_(type(k.key), str)


def test_json_loading():

    # tests if we can build a simple JSON loader for the Iris Flower dataset
    protocols = {"default": _data_file("iris.json")}

    fieldnames = (
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    )

    dataset = JSONDataset(protocols, fieldnames, _raw_data_loader)

    data = dataset.subsets("default")
    dataset.check()

    nose.tools.eq_(len(data["train"]), 75)
    for k in data["train"]:
        for f in range(4):
            nose.tools.eq_(type(k.data[f]), float)
        nose.tools.eq_(type(k.data[4]), str)
        nose.tools.eq_(type(k.key), str)

    nose.tools.eq_(len(data["test"]), 75)
    for k in data["test"]:
        for f in range(4):
            nose.tools.eq_(type(k.data[f]), float)
        nose.tools.eq_(type(k.data[4]), str)
        nose.tools.eq_(type(k.key), str)
