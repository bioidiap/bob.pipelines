import os
import tempfile

from functools import partial

import dask
import dask_ml.decomposition
import dask_ml.preprocessing
import dask_ml.wrappers
import numpy as np
import xarray as xr

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

import bob.io.base
import bob.pipelines


def _build_toy_samples(delayed=False):
    X = np.ones(shape=(10, 5), dtype=int)
    if delayed:

        def _load(index, attr):
            if attr == "data":
                return X[index]
            if attr == "key":
                return str(index)

        samples = [
            bob.pipelines.DelayedSample(
                partial(_load, i, "data"),
                delayed_attributes=dict(key=partial(_load, i, "key")),
            )
            for i in range(len(X))
        ]
    else:
        samples = [
            bob.pipelines.Sample(data, key=str(i)) for i, data in enumerate(X)
        ]
    return X, samples


def test_samples_to_dataset():
    X, samples = _build_toy_samples()
    dataset = bob.pipelines.xr.samples_to_dataset(samples)
    assert dataset.dims == {
        "sample": X.shape[0],
        "dim_0": X.shape[1],
    }, dataset.dims
    np.testing.assert_array_equal(dataset["data"], X)
    np.testing.assert_array_equal(dataset["key"], [str(i) for i in range(10)])


def test_delayed_samples_to_dataset():
    X, samples = _build_toy_samples(delayed=True)
    dataset = bob.pipelines.xr.samples_to_dataset(samples)
    assert dataset.dims == {
        "sample": X.shape[0],
        "dim_0": X.shape[1],
    }, dataset.dims
    np.testing.assert_array_equal(dataset["data"], X)
    np.testing.assert_array_equal(dataset["key"], [str(i) for i in range(10)])


def _build_iris_dataset(shuffle=False, delayed=False):
    iris = datasets.load_iris()

    X = iris.data
    keys = [str(k) for k in range(len(X))]

    if delayed:

        def _load(index, attr):
            if attr == "data":
                return X[index]
            if attr == "key":
                return str(index)
            if attr == "target":
                return iris.target[index]

        samples = [
            bob.pipelines.DelayedSample(
                partial(_load, i, "data"),
                delayed_attributes=dict(
                    key=partial(_load, i, "key"),
                    target=partial(_load, i, "target"),
                ),
            )
            for i in range(len(X))
        ]
    else:
        samples = [
            bob.pipelines.Sample(x, target=y, key=k)
            for x, y, k in zip(iris.data, iris.target, keys)
        ]
    meta = xr.DataArray(X[0], dims=("feature",))
    dataset = bob.pipelines.xr.samples_to_dataset(
        samples, meta=meta, npartitions=3, shuffle=shuffle
    )
    return dataset


def test_dataset_pipeline():
    for delayed in (True, False):
        ds = _build_iris_dataset(delayed=delayed)
        estimator = bob.pipelines.xr.DatasetPipeline(
            [
                PCA(n_components=0.99),
                {
                    "estimator": LinearDiscriminantAnalysis(),
                    "fit_input": ["data", "target"],
                },
            ]
        )

        estimator = estimator.fit(ds)
        ds = estimator.decision_function(ds)
        ds.compute()


def test_dataset_pipeline_with_shapes():
    ds = _build_iris_dataset()
    estimator = bob.pipelines.xr.DatasetPipeline(
        [
            {"estimator": PCA(n_components=3), "output_dims": [("feature", 3)]},
            {
                "estimator": LinearDiscriminantAnalysis(),
                "fit_input": ["data", "target"],
                "output_dims": [("probabilities", 3)],
            },
        ]
    )

    estimator = estimator.fit(ds)
    ds = estimator.decision_function(ds)
    ds.compute()


def test_dataset_pipeline_with_checkpoints():
    iris_ds = _build_iris_dataset()
    with tempfile.TemporaryDirectory() as d:
        scaled_features = os.path.join(d, "scaled_features")
        scaler_model = os.path.join(d, "scaler.pkl")
        pca_model = os.path.join(d, "pca.pkl")
        pca_features = os.path.join(d, "pca_features")
        lda_model = os.path.join(d, "lda.pkl")
        estimator = bob.pipelines.xr.DatasetPipeline(
            [
                {
                    "estimator": StandardScaler(),
                    "output_dims": [("feature", None)],
                    "model_path": scaler_model,
                    "features_dir": scaled_features,
                },
                {
                    "estimator": PCA(n_components=3),
                    "output_dims": [("pca_features", 3)],
                    "model_path": pca_model,
                    "features_dir": pca_features,
                },
                {
                    "estimator": LinearDiscriminantAnalysis(),
                    "fit_input": ["data", "target"],
                    "output_dims": [("probabilities", 3)],
                    "model_path": lda_model,
                },
            ]
        )

        estimator.fit(iris_ds)
        ds = estimator.decision_function(iris_ds)
        oracle = ds.compute()

        assert os.path.isfile(pca_model)
        paths = os.listdir(pca_features)
        assert len(paths) == 150, (len(paths), paths)
        assert os.path.isfile(lda_model)
        for i, path in enumerate(
            sorted(
                os.listdir(pca_features),
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
            )
        ):
            path = os.path.join(pca_features, path)
            assert path.endswith(f"{i}.hdf5"), path
            np.testing.assert_array_equal(bob.io.base.load(path).shape, (3,))

        # now this time it should load features
        # delete one of the features
        os.remove(os.path.join(pca_features, paths[0]))
        estimator.fit(iris_ds)
        ds = estimator.decision_function(iris_ds)
        xr.testing.assert_allclose(ds, oracle)


class FailingPCA(PCA):
    def transform(self, X):
        Xt = super().transform(X)
        Xt[::2] = np.nan
        return Xt


def test_dataset_pipeline_with_failures():
    iris_ds = _build_iris_dataset()
    estimator = bob.pipelines.xr.DatasetPipeline(
        [
            dict(
                estimator=FailingPCA(n_components=3),
                output_dims=[("pca_features", 3)],
            ),
            dict(dataset_map=lambda x: x.persist().dropna("sample")),
            dict(
                estimator=LinearDiscriminantAnalysis(),
                fit_input=["data", "target"],
            ),
        ]
    )

    estimator = estimator.fit(iris_ds)
    ds = estimator.decision_function(iris_ds)
    ds = ds.compute()
    assert ds.dims == {"sample": 75, "c": 3}, ds.dims


def test_dataset_pipeline_with_dask_ml():
    scaler = dask_ml.preprocessing.StandardScaler()
    pca = dask_ml.decomposition.PCA(n_components=3, random_state=0)
    clf = SGDClassifier(random_state=0, loss="log_loss", penalty="l2", tol=1e-3)
    clf = dask_ml.wrappers.Incremental(clf, scoring="accuracy")

    iris_ds = _build_iris_dataset(shuffle=True)

    estimator = bob.pipelines.xr.DatasetPipeline(
        [
            dict(
                estimator=scaler,
                output_dims=[("feature", None)],
                input_dask_array=True,
            ),
            dict(
                estimator=pca,
                output_dims=[("pca_features", 3)],
                input_dask_array=True,
            ),
            dict(
                estimator=clf,
                fit_input=["data", "target"],
                output_dims=[],
                input_dask_array=True,
                fit_kwargs=dict(classes=range(3)),
            ),
        ]
    )

    with dask.config.set(scheduler="synchronous"):
        estimator = estimator.fit(iris_ds)
        ds = estimator.predict(iris_ds)
        ds = ds.compute()
    correct_classification = np.asarray(ds.data == ds.target).sum()
    assert correct_classification > 80, correct_classification
    assert ds.dims == {"sample": 150}, ds.dims
