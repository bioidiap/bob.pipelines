import os
import shutil
import tempfile

import dask.array as da
import dask_ml.cluster
import dask_ml.datasets
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_array, check_is_fitted

import bob.pipelines

from bob.pipelines import hash_string
from bob.pipelines.wrappers import getattr_nested


def _offset_add_func(X, offset=1):
    return X + offset


class DummyWithFit(TransformerMixin, BaseEstimator):
    """See https://scikit-learn.org/stable/developers/develop.html and
    https://github.com/scikit-learn-contrib/project-
    template/blob/master/skltemplate/_template.py."""

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        self.model_ = np.ones((self.n_features_in_, 2))

        # Return the transformer
        return self

    def transform(self, X):
        # Check is fit had been called
        check_is_fitted(self, "n_features_in_")
        # Input validation
        X = check_array(X)
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Shape of input is different from what was seen" "in `fit`"
            )
        return X @ self.model_


class DummyTransformer(TransformerMixin, BaseEstimator):
    """See https://scikit-learn.org/stable/developers/develop.html and
    https://github.com/scikit-learn-contrib/project-
    template/blob/master/skltemplate/_template.py."""

    def __init__(self, i=None, **kwargs):
        super().__init__(**kwargs)
        self.i = i

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Input validation
        X = check_array(X)
        # Check that the input is of the same shape as the one passed
        # during fit.
        return _offset_add_func(X)

    def _more_tags(self):
        return {"requires_fit": False}


class HalfFailingDummyTransformer(DummyTransformer):
    """Transformer that fails for some samples (all even indices fail)"""

    def transform(self, X):
        X = check_array(X, force_all_finite=False)
        X = _offset_add_func(X)
        output = []
        for i, x in enumerate(X):
            output.append(x if i % 2 else None)
        return output


class FullFailingDummyTransformer(DummyTransformer):
    """Transformer that fails for all samples"""

    def transform(self, X):
        return [None] * len(X)


class DummyWithTags(DummyTransformer):
    """Transformer that specifies tags"""

    def transform(self, X, extra_arg_1, extra_arg_2):
        np.testing.assert_equal(np.array(X), extra_arg_1)
        np.testing.assert_equal(np.array(X), extra_arg_2)
        return super().transform(X)

    def fit(self, X, y, extra):
        np.testing.assert_equal(np.array(X), y)
        np.testing.assert_equal(np.array(X) + 1, extra)
        return self

    def _more_tags(self):
        return {
            "bob_output": "annotations",
            "bob_transform_extra_input": (
                ("extra_arg_1", "data"),
                ("extra_arg_2", "data"),
            ),
            "bob_fit_extra_input": (("y", "data"), ("extra", "annotations")),
        }


class DummyWithTagsNotData(DummyTransformer):
    """Transformer that specifies a different field than `data` for argument 1."""

    def transform(self, X, extra_arg_1, extra_arg_2):
        np.testing.assert_equal(np.array(X), extra_arg_1)
        np.testing.assert_equal(np.array(X) - 1, extra_arg_2)
        return super().transform(X)

    def fit(self, X, y, extra):
        np.testing.assert_equal(np.array(X), y)
        np.testing.assert_equal(np.array(X) - 1, extra)
        return self

    def _more_tags(self):
        return {
            "bob_output": "annotations_2",
            "bob_input": "annotations",
            "bob_transform_extra_input": (
                ("extra_arg_1", "annotations"),
                ("extra_arg_2", "data"),
            ),
            "bob_fit_extra_input": (("y", "annotations"), ("extra", "data")),
        }


class DummyWithDask(DummyTransformer):
    """Transformer that specifies the supports_dask_array tag."""

    def __init__(self):
        self.model_ = np.zeros(2)

    def transform(self, X):
        return X

    def fit(self, X, y=None):
        assert isinstance(X, da.Array)
        self.model_ = X.sum(axis=0) + self.model_
        self.model_ = self.model_.compute()
        return self

    def _more_tags(self):
        return {
            "bob_output": "annotations",
            "bob_fit_supports_dask_array": True,
            "requires_fit": True,
        }


def _assert_all_close_numpy_array(oracle, result):
    oracle, result = np.array(oracle), np.array(result)
    assert (
        oracle.shape == result.shape
    ), f"Expected: {oracle.shape} but got: {result.shape}"
    assert np.allclose(oracle, result), f"Expected: {oracle} but got: {result}"


def test_sklearn_compatible_estimator():
    # check classes for API consistency
    check_estimator(DummyWithFit())


def test_function_sample_transfomer():
    X = np.zeros(shape=(10, 2), dtype=int)
    samples = [bob.pipelines.Sample(data) for data in X]

    transformer = bob.pipelines.wrap(
        [FunctionTransformer, "sample"],
        func=_offset_add_func,
        kw_args=dict(offset=3),
        validate=True,
    )

    features = transformer.transform(samples)
    _assert_all_close_numpy_array(X + 3, [s.data for s in features])

    features = transformer.fit_transform(samples)
    _assert_all_close_numpy_array(X + 3, [s.data for s in features])


def test_fittable_sample_transformer():
    X = np.ones(shape=(10, 2), dtype=int)
    samples = [bob.pipelines.Sample(data) for data in X]

    # Mixing up with an object
    transformer = bob.pipelines.wrap([DummyWithFit, "sample"])
    features = transformer.fit(samples).transform(samples)
    _assert_all_close_numpy_array(X + 1, [s.data for s in features])

    features = transformer.fit_transform(samples)
    _assert_all_close_numpy_array(X + 1, [s.data for s in features])


def test_tagged_sample_transformer():
    X = np.ones(shape=(10, 2), dtype=int)
    samples = [bob.pipelines.Sample(data) for data in X]

    # Mixing up with an object
    transformer = bob.pipelines.wrap([DummyWithTags, "sample"])
    features = transformer.transform(samples)
    _assert_all_close_numpy_array(X + 1, [s.annotations for s in features])
    _assert_all_close_numpy_array(X, [s.data for s in features])
    transformer.fit(samples)


def test_tagged_input_sample_transformer():
    X = np.ones(shape=(10, 2), dtype=int)
    samples = [bob.pipelines.Sample(data) for data in X]

    # Mixing up with an object
    annotator = bob.pipelines.wrap([DummyWithTags, "sample"])
    features = annotator.transform(samples)
    transformer = bob.pipelines.wrap([DummyWithTagsNotData, "sample"])
    features = transformer.transform(samples)
    _assert_all_close_numpy_array(X + 2, [s.annotations_2 for s in features])
    _assert_all_close_numpy_array(X, [s.data for s in features])
    transformer.fit(samples)


def test_dask_tag_transformer():
    X = np.ones(shape=(10, 2), dtype=int)
    samples = [bob.pipelines.Sample(data) for data in X]
    sample_bags = bob.pipelines.ToDaskBag().transform(samples)

    transformer = bob.pipelines.wrap([DummyWithDask, "sample", "dask"])

    transformer.fit(sample_bags)
    model_ = getattr_nested(transformer, "model_")
    np.testing.assert_equal(model_, X.sum(axis=0))


def test_dask_tag_checkpoint_transformer():
    X = np.ones(shape=(10, 2), dtype=int)
    samples = [bob.pipelines.Sample(data) for data in X]
    sample_bags = bob.pipelines.ToDaskBag().transform(samples)

    with tempfile.TemporaryDirectory() as d:
        transformer = bob.pipelines.wrap(
            [DummyWithDask, "sample", "checkpoint", "dask"],
            model_path=d + "/ckpt.h5",
        )
        transformer.fit(sample_bags)
        model_ = getattr_nested(transformer, "model_")
        np.testing.assert_equal(model_, X.sum(axis=0))
        # Fit with no data to verify loading from checkpoint
        transformer_2 = bob.pipelines.wrap(
            [DummyWithDask, "sample", "checkpoint", "dask"],
            model_path=d + "/ckpt.h5",
        )
        transformer_2.fit(None)
        model_ = getattr_nested(transformer, "model_")
        np.testing.assert_equal(model_, X.sum(axis=0))


def test_dask_tag_daskml_estimator():
    X, labels = make_blobs(
        n_samples=1000,
        n_features=2,
        random_state=0,
        centers=[[-1, -1], [1, 1]],
        cluster_std=0.1,  # Makes it easy to split
    )
    samples = [bob.pipelines.Sample(data) for data in X]
    sample_bags = bob.pipelines.ToDaskBag(partition_size=300).transform(samples)

    estimator = dask_ml.cluster.KMeans(
        n_clusters=2, init_max_iter=2, random_state=0
    )

    for fit_supports_dask_array in [True, False]:
        transformer = bob.pipelines.wrap(
            ["sample", "dask"],
            estimator=estimator,
            fit_supports_dask_array=fit_supports_dask_array,
        )
        transformer.fit(sample_bags)
        labels_ = getattr_nested(transformer, "labels_")
        if labels_[0] != labels[0]:
            # if the labels are flipped during kmeans
            labels = 1 - labels
        np.testing.assert_array_equal(labels_, labels)

    for fit_supports_dask_array in [True, False]:
        with tempfile.TemporaryDirectory() as d:
            transformer = bob.pipelines.wrap(
                ["sample", "checkpoint", "dask"],
                estimator=estimator,
                fit_supports_dask_array=fit_supports_dask_array,
                model_path=f"{d}/ckpt.pkl",
            )
            for i in range(2):
                X = sample_bags
                if i == 1:
                    X = None
                transformer.fit(X)
                labels_ = getattr_nested(transformer, "labels_")
                if labels_[0] != labels[0]:
                    # if the labels are flipped during kmeans
                    labels = 1 - labels
                np.testing.assert_array_equal(labels_, labels)


def test_failing_sample_transformer():
    X = np.zeros(shape=(10, 2))
    samples = [bob.pipelines.Sample(data) for i, data in enumerate(X)]
    expected = np.full_like(X, 2, dtype=object)
    expected[::2] = None
    expected[1::4] = None

    transformer = Pipeline(
        [
            ("1", bob.pipelines.wrap([HalfFailingDummyTransformer, "sample"])),
            ("2", bob.pipelines.wrap([HalfFailingDummyTransformer, "sample"])),
        ]
    )
    features = transformer.transform(samples)

    features = [f.data for f in features]
    assert len(expected) == len(
        features
    ), f"Expected: {len(expected)} but got: {len(features)}"
    assert all(
        (e == f).all() for e, f in zip(expected, features)
    ), f"Expected: {expected} but got: {features}"

    samples = [bob.pipelines.Sample(data) for data in X]
    expected = [None] * X.shape[0]
    transformer = Pipeline(
        [
            ("1", bob.pipelines.wrap([FullFailingDummyTransformer, "sample"])),
            ("2", bob.pipelines.wrap([FullFailingDummyTransformer, "sample"])),
        ]
    )
    features = transformer.transform(samples)

    features = [f.data for f in features]
    assert len(expected) == len(
        features
    ), f"Expected: {len(expected)} but got: {len(features)}"
    assert all(
        e == f for e, f in zip(expected, features)
    ), f"Expected: {expected} but got: {features}"


def test_failing_checkpoint_transformer():
    X = np.zeros(shape=(10, 2))
    samples = [bob.pipelines.Sample(data, key=i) for i, data in enumerate(X)]
    expected = np.full_like(X, 2)
    expected[::2] = None
    expected[1::4] = None
    expected = list(expected)

    with tempfile.TemporaryDirectory() as d:
        features_dir_1 = os.path.join(d, "features_1")
        features_dir_2 = os.path.join(d, "features_2")
        transformer = Pipeline(
            [
                (
                    "1",
                    bob.pipelines.wrap(
                        [HalfFailingDummyTransformer, "sample", "checkpoint"],
                        features_dir=features_dir_1,
                    ),
                ),
                (
                    "2",
                    bob.pipelines.wrap(
                        [HalfFailingDummyTransformer, "sample", "checkpoint"],
                        features_dir=features_dir_2,
                    ),
                ),
            ]
        )
        features = transformer.transform(samples)

        np_features = np.array(
            [
                np.full(X.shape[1], np.nan) if f.data is None else f.data
                for f in features
            ]
        )
        assert len(expected) == len(
            np_features
        ), f"Expected: {len(expected)} but got: {len(np_features)}"
        assert np.allclose(
            expected, np_features, equal_nan=True
        ), f"Expected: {expected} but got: {np_features}"

    samples = [bob.pipelines.Sample(data, key=i) for i, data in enumerate(X)]
    expected = [None] * X.shape[0]

    with tempfile.TemporaryDirectory() as d:
        features_dir_1 = os.path.join(d, "features_1")
        features_dir_2 = os.path.join(d, "features_2")
        transformer = Pipeline(
            [
                (
                    "1",
                    bob.pipelines.wrap(
                        [FullFailingDummyTransformer, "sample", "checkpoint"],
                        features_dir=features_dir_1,
                    ),
                ),
                (
                    "2",
                    bob.pipelines.wrap(
                        [FullFailingDummyTransformer, "sample", "checkpoint"],
                        features_dir=features_dir_2,
                    ),
                ),
            ]
        )
        features = transformer.transform(samples)

        assert len(expected) == len(
            features
        ), f"Expected: {len(expected)} but got: {len(features)}"
        assert all(
            e == f.data for e, f in zip(expected, features)
        ), f"Expected: {expected} but got: {features}"


def _assert_checkpoints(
    features, oracle, model_path, features_dir, not_requires_fit
):
    _assert_all_close_numpy_array(oracle, [s.data for s in features])
    if not_requires_fit:
        assert not os.path.exists(model_path)
    else:
        assert os.path.exists(model_path), os.listdir(
            os.path.dirname(model_path)
        )
    assert os.path.isdir(features_dir)
    for i in range(len(oracle)):
        assert os.path.isfile(os.path.join(features_dir, f"{i}.h5"))


def _assert_delayed_samples(samples):
    for s in samples:
        assert isinstance(s, bob.pipelines.DelayedSample)


def test_checkpoint_function_sample_transfomer():
    X = np.arange(20, dtype=int).reshape(10, 2)
    samples = [
        bob.pipelines.Sample(data, key=str(i)) for i, data in enumerate(X)
    ]
    offset = 3
    oracle = X + offset

    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "model.pkl")
        features_dir = os.path.join(d, "features")

        transformer = bob.pipelines.wrap(
            [FunctionTransformer, "sample", "checkpoint"],
            func=_offset_add_func,
            kw_args=dict(offset=offset),
            validate=True,
            model_path=model_path,
            features_dir=features_dir,
        )

        features = transformer.transform(samples)
        _assert_checkpoints(features, oracle, model_path, features_dir, True)

        features = transformer.fit_transform(samples)
        _assert_checkpoints(features, oracle, model_path, features_dir, True)
        _assert_delayed_samples(features)

        # remove all files and call fit_transform again
        shutil.rmtree(d)
        features = transformer.fit_transform(samples)
        _assert_checkpoints(features, oracle, model_path, features_dir, True)

    # test when both model_path and features_dir is None
    transformer = bob.pipelines.wrap(
        [FunctionTransformer, "sample", "checkpoint"],
        func=_offset_add_func,
        kw_args=dict(offset=offset),
        validate=True,
    )
    features = transformer.transform(samples)
    _assert_all_close_numpy_array(oracle, [s.data for s in features])

    # test when both model_path and features_dir is None
    with tempfile.TemporaryDirectory() as dir_name:
        transformer = bob.pipelines.wrap(
            [FunctionTransformer, "sample", "checkpoint"],
            func=_offset_add_func,
            kw_args=dict(offset=offset),
            validate=True,
            features_dir=dir_name,
            hash_fn=hash_string,
        )

        features = transformer.transform(samples)
        # Checking if we can cast the has as integer
        assert isinstance(int(features[0]._load.args[0].split("/")[-2]), int)

        _assert_all_close_numpy_array(oracle, [s.data for s in features])


def test_checkpoint_fittable_sample_transformer():
    X = np.ones(shape=(10, 2), dtype=int)
    samples = [
        bob.pipelines.Sample(data, key=str(i)) for i, data in enumerate(X)
    ]
    oracle = X + 1

    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "model.pkl")
        features_dir = os.path.join(d, "features")

        transformer = bob.pipelines.wrap(
            [DummyWithFit, "sample", "checkpoint"],
            model_path=model_path,
            features_dir=features_dir,
        )
        assert bob.pipelines.estimator_requires_fit(transformer)
        features = transformer.fit(samples).transform(samples)
        _assert_checkpoints(features, oracle, model_path, features_dir, False)

        features = transformer.fit_transform(samples)
        _assert_checkpoints(features, oracle, model_path, features_dir, False)
        _assert_delayed_samples(features)

        # remove all files and call fit_transform again
        shutil.rmtree(d)
        features = transformer.fit_transform(samples)
        _assert_checkpoints(features, oracle, model_path, features_dir, False)


def _build_estimator(path, i):
    base_dir = os.path.join(path, f"transformer{i}")
    os.makedirs(base_dir, exist_ok=True)
    model_path = os.path.join(base_dir, "model.pkl")
    features_dir = os.path.join(base_dir, "features")

    transformer = bob.pipelines.wrap(
        [DummyWithFit, "sample", "checkpoint"],
        model_path=model_path,
        features_dir=features_dir,
    )
    return transformer


def _build_transformer(path, i, force=False):
    features_dir = os.path.join(path, f"transformer{i}")
    estimator = bob.pipelines.wrap(
        [DummyTransformer, "sample", "checkpoint"],
        i=i,
        features_dir=features_dir,
        force=force,
    )
    return estimator


def test_checkpoint_fittable_pipeline():
    X = np.ones(shape=(10, 2), dtype=int)
    samples = [
        bob.pipelines.Sample(data, key=str(i)) for i, data in enumerate(X)
    ]
    samples_transform = [
        bob.pipelines.Sample(data, key=str(i + 10)) for i, data in enumerate(X)
    ]
    oracle = X + 3

    with tempfile.TemporaryDirectory() as d:
        pipeline = Pipeline(
            [(f"{i}", _build_estimator(d, i)) for i in range(2)]
        )
        pipeline.fit(samples)

        transformed_samples = pipeline.transform(samples_transform)

        _assert_all_close_numpy_array(
            oracle, [s.data for s in transformed_samples]
        )


def test_checkpoint_transform_pipeline():
    def _run(dask_enabled):
        X = np.ones(shape=(10, 2), dtype=int)
        samples_transform = [
            bob.pipelines.Sample(data, key=str(i)) for i, data in enumerate(X)
        ]
        offset = 2
        oracle = X + offset

        with tempfile.TemporaryDirectory() as d:
            pipeline = Pipeline(
                [(f"{i}", _build_transformer(d, i)) for i in range(offset)]
            )
            if dask_enabled:
                pipeline = bob.pipelines.wrap(["dask"], pipeline)
                transformed_samples = pipeline.transform(
                    samples_transform
                ).compute(scheduler="single-threaded")
            else:
                transformed_samples = pipeline.transform(samples_transform)

            _assert_all_close_numpy_array(
                oracle, [s.data for s in transformed_samples]
            )

    _run(dask_enabled=True)
    _run(dask_enabled=False)


def test_checkpoint_transform_pipeline_force():
    with tempfile.TemporaryDirectory() as d:

        def _run():
            X = np.ones(shape=(10, 2), dtype=int)
            samples_transform = [
                bob.pipelines.Sample(data, key=str(i))
                for i, data in enumerate(X)
            ]
            offset = 2
            oracle = X + offset

            pipeline = Pipeline(
                [
                    (f"{i}", _build_transformer(d, i, force=True))
                    for i in range(offset)
                ]
            )

            pipeline = bob.pipelines.wrap(["dask"], pipeline)
            transformed_samples = pipeline.transform(samples_transform).compute(
                scheduler="single-threaded"
            )

            _assert_all_close_numpy_array(
                oracle, [s.data for s in transformed_samples]
            )

        _run()
        _run()


def test_checkpoint_fit_transform_pipeline():
    def _run(dask_enabled):
        X = np.ones(shape=(10, 2), dtype=int)
        samples = [
            bob.pipelines.Sample(data, key=str(i)) for i, data in enumerate(X)
        ]
        samples_transform = [
            bob.pipelines.Sample(data, key=str(i + 10))
            for i, data in enumerate(X)
        ]
        oracle = X + 2

        with tempfile.TemporaryDirectory() as d:
            fitter = ("0", _build_estimator(d, 0))
            transformer = ("1", _build_transformer(d, 1))
            pipeline = Pipeline([fitter, transformer])
            if dask_enabled:
                pipeline = bob.pipelines.wrap(
                    ["dask"], pipeline, fit_tag="GPU", npartitions=1
                )
                pipeline = pipeline.fit(samples)
                tags = bob.pipelines.dask_tags(pipeline)

                assert len(tags) == 1, tags
                transformed_samples = pipeline.transform(samples_transform)

                transformed_samples = transformed_samples.compute(
                    scheduler="single-threaded"
                )
            else:
                pipeline = pipeline.fit(samples)
                transformed_samples = pipeline.transform(samples_transform)

            _assert_all_close_numpy_array(
                oracle, [s.data for s in transformed_samples]
            )

    _run(dask_enabled=True)
    _run(dask_enabled=False)


def _get_local_client():
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(
        nanny=False, processes=False, n_workers=1, threads_per_worker=1
    )
    cluster.scale_up(1)
    return Client(cluster)  # start local workers as threads


def test_checkpoint_fit_transform_pipeline_with_dask_non_pickle():
    def _run(dask_enabled):
        X = np.ones(shape=(10, 2), dtype=int)
        samples = [
            bob.pipelines.Sample(data, key=str(i)) for i, data in enumerate(X)
        ]
        samples_transform = [
            bob.pipelines.Sample(data, key=str(i + 10))
            for i, data in enumerate(X)
        ]
        oracle = X + 2

        with tempfile.TemporaryDirectory() as d:
            fitter = ("0", _build_estimator(d, 0))
            transformer = (
                "1",
                _build_transformer(d, 1),
            )

            pipeline = Pipeline([fitter, transformer])
            if dask_enabled:
                dask_client = _get_local_client()
                pipeline = bob.pipelines.wrap(["dask"], pipeline)
                pipeline = pipeline.fit(samples)
                transformed_samples = pipeline.transform(
                    samples_transform
                ).compute(scheduler=dask_client)
            else:
                pipeline = pipeline.fit(samples)
                transformed_samples = pipeline.transform(samples_transform)

            _assert_all_close_numpy_array(
                oracle, [s.data for s in transformed_samples]
            )

    _run(True)
    _run(False)


def test_dask_checkpoint_transform_pipeline():
    X = np.ones(shape=(10, 2), dtype=int)
    samples_transform = [
        bob.pipelines.Sample(data, key=str(i)) for i, data in enumerate(X)
    ]
    with tempfile.TemporaryDirectory() as d:
        bag_transformer = bob.pipelines.ToDaskBag()
        estimator = bob.pipelines.wrap(
            ["dask"], _build_transformer(d, 0), transform_tag="CPU"
        )
        X_tr = estimator.transform(bag_transformer.transform(samples_transform))
        assert len(bob.pipelines.dask_tags(estimator)) == 1
        assert len(X_tr.compute(scheduler="single-threaded")) == 10


def test_checkpoint_transform_pipeline_with_sampleset():
    def _run(dask_enabled):
        X = np.ones(shape=(10, 2), dtype=int)
        samples_transform = bob.pipelines.SampleSet(
            [
                bob.pipelines.Sample(data, key=str(i))
                for i, data in enumerate(X)
            ],
            key="1",
        )
        offset = 2
        oracle = X + offset

        with tempfile.TemporaryDirectory() as d:
            pipeline = Pipeline(
                [(f"{i}", _build_transformer(d, i)) for i in range(offset)]
            )
            if dask_enabled:
                pipeline = bob.pipelines.wrap(["dask"], pipeline)
                transformed_samples = pipeline.transform(
                    [samples_transform]
                ).compute(scheduler="single-threaded")
            else:
                transformed_samples = pipeline.transform([samples_transform])

            _assert_all_close_numpy_array(
                oracle,
                [
                    s.data
                    for sample_set in transformed_samples
                    for s in sample_set
                ],
            )
            assert np.all([len(s) == 10 for s in transformed_samples])

    _run(dask_enabled=True)
    _run(dask_enabled=False)


def test_estimator_requires_fit():
    all_wraps = [
        ["sample"],
        ["sample", "checkpoint"],
        ["sample", "checkpoint", "dask"],
        ["sample", "dask"],
        ["checkpoint"],
        ["checkpoint", "dask"],
        ["dask"],
    ]

    for estimator, requires_fit in [
        (DummyTransformer(), False),
        (DummyWithFit(), True),
    ]:
        assert bob.pipelines.estimator_requires_fit(estimator) is requires_fit

        # test on a pipeline
        pipeline = Pipeline([(f"{i}", estimator) for i in range(2)])
        assert bob.pipelines.estimator_requires_fit(pipeline) is requires_fit

        # now test if wrapped, is also correct
        for wraps in all_wraps:
            est = bob.pipelines.wrap(wraps, estimator)
            assert bob.pipelines.estimator_requires_fit(est) is requires_fit

            # test on a pipeline
            pipeline = Pipeline([(f"{i}", est) for i in range(2)])
            assert (
                bob.pipelines.estimator_requires_fit(pipeline) is requires_fit
            )

    pipeline = Pipeline(
        [
            ("DummyTransformer", DummyTransformer()),
            ("DummyWithFit", DummyWithFit()),
        ]
    )
    assert bob.pipelines.estimator_requires_fit(pipeline) is True
    for wrap in all_wraps:
        est = bob.pipelines.wrap(wrap, pipeline)
        assert bob.pipelines.estimator_requires_fit(est) is True

    # test with a FunctionTransformer
    estimator = FunctionTransformer(lambda x: x)
    assert bob.pipelines.estimator_requires_fit(estimator) is False
    for wrap in all_wraps:
        est = bob.pipelines.wrap(wrap, estimator)
        assert bob.pipelines.estimator_requires_fit(est) is False
