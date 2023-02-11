#!/usr/bin/env python
# coding=utf-8

"""Test code for datasets"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from bob.pipelines import FileListDatabase
from bob.pipelines.transformers import Str_To_Types

DATA_PATH = Path(__file__).parent / "data"


def iris_data_transform(samples):
    for s in samples:
        data = np.array(
            [s.sepal_length, s.sepal_width, s.petal_length, s.petal_width]
        )
        s.data = data
    return samples


def test_iris_list_database():
    protocols_path = DATA_PATH / "iris_database"

    database = FileListDatabase(
        name="iris", protocol=None, dataset_protocols_path=protocols_path
    )
    assert database.name == "iris"
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
        FunctionTransformer(iris_data_transform),
    )
    samples = database.samples(groups="train")
    assert len(samples) == 75
    np.testing.assert_allclose(samples[0].data, [5.1, 3.5, 1.4, 0.2])
    assert samples[0].sepal_length == 5.1
    assert samples[0].petal_width == 0.2
    assert samples[0].target == "Iris-setosa"


def test_filelist_class(monkeypatch):
    protocols_path = Path(DATA_PATH / "iris_database")

    class DBLocal(FileListDatabase):
        name = "iris"
        dataset_protocols_path = protocols_path

    assert DBLocal.protocols() == ["default"]
    assert DBLocal.retrieve_dataset_protocols() == protocols_path

    with TemporaryDirectory(prefix="bobtest_") as tmpdir:
        tmp_home = Path(tmpdir)
        monkeypatch.setenv("HOME", tmp_home.as_posix())

        class DBDownloadDefault(FileListDatabase):
            name = "atnt"
            dataset_protocols_checksum = "f529acef"
            dataset_protocols_urls = [
                "https://www.idiap.ch/software/bob/databases/latest/base/atnt-f529acef.tar.gz"
            ]

        assert DBDownloadDefault.protocols() == ["idiap_protocol"]
        assert (
            DBDownloadDefault.retrieve_dataset_protocols()
            == tmp_home / "bob_data" / "protocols" / "atnt-f529acef.tar.gz"
        )

    with TemporaryDirectory(prefix="bobtest_") as tmpdir:
        tmp_home = Path(tmpdir)
        monkeypatch.setenv("HOME", tmp_home.as_posix())
        desired_name = "atnt_filename.tar.gz"

        class DBDownloadCustomFilename(FileListDatabase):
            name = "atnt"
            dataset_protocols_checksum = "f529acef"
            dataset_protocols_urls = [
                "https://www.idiap.ch/software/bob/databases/latest/base/atnt-f529acef.tar.gz"
            ]
            dataset_protocols_name = desired_name

        assert DBDownloadCustomFilename.protocols() == ["idiap_protocol"]
        assert (
            DBDownloadCustomFilename.retrieve_dataset_protocols()
            == tmp_home / "bob_data" / "protocols" / desired_name
        )

    with TemporaryDirectory(prefix="bobtest_") as tmpdir:
        tmp_home = Path(tmpdir)
        monkeypatch.setenv("HOME", tmp_home.as_posix())
        desired_category = "custom_category"

        class DBDownloadCustomCategory(FileListDatabase):
            name = "atnt"
            category = desired_category
            dataset_protocols_checksum = "f529acef"
            dataset_protocols_urls = [
                "https://www.idiap.ch/software/bob/databases/latest/base/atnt-f529acef.tar.gz"
            ]

        assert DBDownloadCustomCategory.protocols() == ["idiap_protocol"]
        assert (
            DBDownloadCustomCategory.retrieve_dataset_protocols()
            == tmp_home
            / "bob_data"
            / "protocols"
            / desired_category
            / "atnt-f529acef.tar.gz"
        )
