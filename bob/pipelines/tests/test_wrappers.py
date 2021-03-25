import os
import shutil
import tempfile

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

import bob.pipelines as mario

from bob.pipelines.utils import hash_string


def _offset_add_func(X, offset=1):
    return X + offset


class DummyWithFit(TransformerMixin, BaseEstimator):
    """See https://scikit-learn.org/stable/developers/develop.html and
    https://github.com/scikit-learn-contrib/project-
    template/blob/master/skltemplate/_template.py."""

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_ = X.shape[1]

        self.model_ = np.ones((self.n_features_, 2))

        # Return the transformer
        return self

    def transform(self, X):
        # Check is fit had been called
        check_is_fitted(self, "n_features_")
        # Input validation
        X = check_array(X)
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
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
        return {"stateless": True, "requires_fit": False}


def _assert_all_close_numpy_array(oracle, result):
    oracle, result = np.array(oracle), np.array(result)
    assert (
        oracle.shape == result.shape
    ), f"Expected: {oracle.shape} but got: {result.shape}"
    assert np.allclose(oracle, result), f"Expected: {oracle} but got: {result}"


def test_sklearn_compatible_estimator():
    # check classes for API consistency
    check_estimator(DummyWithFit)


def test_function_sample_transfomer():

    X = np.zeros(shape=(10, 2), dtype=int)
    samples = [mario.Sample(data) for data in X]

    transformer = mario.wrap(
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
    samples = [mario.Sample(data) for data in X]

    # Mixing up with an object
    transformer = mario.wrap([DummyWithFit, "sample"])
    features = transformer.fit(samples).transform(samples)
    _assert_all_close_numpy_array(X + 1, [s.data for s in features])

    features = transformer.fit_transform(samples)
    _assert_all_close_numpy_array(X + 1, [s.data for s in features])


def _assert_checkpoints(features, oracle, model_path, features_dir, stateless):
    _assert_all_close_numpy_array(oracle, [s.data for s in features])
    if stateless:
        assert not os.path.exists(model_path)
    else:
        assert os.path.exists(model_path), os.listdir(os.path.dirname(model_path))
    assert os.path.isdir(features_dir)
    for i in range(len(oracle)):
        assert os.path.isfile(os.path.join(features_dir, f"{i}.h5"))


def _assert_delayed_samples(samples):
    for s in samples:
        assert isinstance(s, mario.DelayedSample)


def test_checkpoint_function_sample_transfomer():

    X = np.arange(20, dtype=int).reshape(10, 2)
    samples = [mario.Sample(data, key=str(i)) for i, data in enumerate(X)]
    offset = 3
    oracle = X + offset

    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "model.pkl")
        features_dir = os.path.join(d, "features")

        transformer = mario.wrap(
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
    transformer = mario.wrap(
        [FunctionTransformer, "sample", "checkpoint"],
        func=_offset_add_func,
        kw_args=dict(offset=offset),
        validate=True,
    )
    features = transformer.transform(samples)
    _assert_all_close_numpy_array(oracle, [s.data for s in features])

    # test when both model_path and features_dir is None
    with tempfile.TemporaryDirectory() as dir_name:
        transformer = mario.wrap(
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
    samples = [mario.Sample(data, key=str(i)) for i, data in enumerate(X)]
    oracle = X + 1

    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "model.pkl")
        features_dir = os.path.join(d, "features")

        transformer = mario.wrap(
            [DummyWithFit, "sample", "checkpoint"],
            model_path=model_path,
            features_dir=features_dir,
        )
        assert not mario.utils.is_estimator_stateless(transformer)
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

    transformer = mario.wrap(
        [DummyWithFit, "sample", "checkpoint"],
        model_path=model_path,
        features_dir=features_dir,
    )
    return transformer


def _build_transformer(path, i):

    features_dir = os.path.join(path, f"transformer{i}")
    estimator = mario.wrap(
        [DummyTransformer, "sample", "checkpoint"], i=i, features_dir=features_dir
    )
    return estimator


def test_checkpoint_fittable_pipeline():

    X = np.ones(shape=(10, 2), dtype=int)
    samples = [mario.Sample(data, key=str(i)) for i, data in enumerate(X)]
    samples_transform = [
        mario.Sample(data, key=str(i + 10)) for i, data in enumerate(X)
    ]
    oracle = X + 3

    with tempfile.TemporaryDirectory() as d:
        pipeline = Pipeline([(f"{i}", _build_estimator(d, i)) for i in range(2)])
        pipeline.fit(samples)

        transformed_samples = pipeline.transform(samples_transform)

        _assert_all_close_numpy_array(oracle, [s.data for s in transformed_samples])


def test_checkpoint_transform_pipeline():
    def _run(dask_enabled):

        X = np.ones(shape=(10, 2), dtype=int)
        samples_transform = [mario.Sample(data, key=str(i)) for i, data in enumerate(X)]
        offset = 2
        oracle = X + offset

        with tempfile.TemporaryDirectory() as d:
            pipeline = Pipeline(
                [(f"{i}", _build_transformer(d, i)) for i in range(offset)]
            )
            if dask_enabled:
                pipeline = mario.wrap(["dask"], pipeline)
                transformed_samples = pipeline.transform(samples_transform).compute(
                    scheduler="single-threaded"
                )
            else:
                transformed_samples = pipeline.transform(samples_transform)

            _assert_all_close_numpy_array(oracle, [s.data for s in transformed_samples])

    _run(dask_enabled=True)
    _run(dask_enabled=False)


def test_checkpoint_fit_transform_pipeline():
    def _run(dask_enabled):
        X = np.ones(shape=(10, 2), dtype=int)
        samples = [mario.Sample(data, key=str(i)) for i, data in enumerate(X)]
        samples_transform = [
            mario.Sample(data, key=str(i + 10)) for i, data in enumerate(X)
        ]
        oracle = X + 2

        with tempfile.TemporaryDirectory() as d:
            fitter = ("0", _build_estimator(d, 0))
            transformer = ("1", _build_transformer(d, 1))
            pipeline = Pipeline([fitter, transformer])
            if dask_enabled:
                pipeline = mario.wrap(["dask"], pipeline, fit_tag="GPU", npartitions=1)
                pipeline = pipeline.fit(samples)
                tags = mario.dask_tags(pipeline)

                assert len(tags) == 1, tags
                transformed_samples = pipeline.transform(samples_transform)

                transformed_samples = transformed_samples.compute(
                    scheduler="single-threaded"
                )
            else:
                pipeline = pipeline.fit(samples)
                transformed_samples = pipeline.transform(samples_transform)

            _assert_all_close_numpy_array(oracle, [s.data for s in transformed_samples])

    _run(dask_enabled=True)
    _run(dask_enabled=False)


def _get_local_client():
    from dask.distributed import Client
    from dask.distributed import LocalCluster

    cluster = LocalCluster(
        nanny=False, processes=False, n_workers=1, threads_per_worker=1
    )
    cluster.scale_up(1)
    return Client(cluster)  # start local workers as threads


def test_checkpoint_fit_transform_pipeline_with_dask_non_pickle():
    def _run(dask_enabled):
        X = np.ones(shape=(10, 2), dtype=int)
        samples = [mario.Sample(data, key=str(i)) for i, data in enumerate(X)]
        samples_transform = [
            mario.Sample(data, key=str(i + 10)) for i, data in enumerate(X)
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
                pipeline = mario.wrap(["dask"], pipeline)
                pipeline = pipeline.fit(samples)
                transformed_samples = pipeline.transform(samples_transform).compute(
                    scheduler=dask_client
                )
            else:
                pipeline = pipeline.fit(samples)
                transformed_samples = pipeline.transform(samples_transform)

            _assert_all_close_numpy_array(oracle, [s.data for s in transformed_samples])

    _run(True)
    _run(False)


def test_dask_checkpoint_transform_pipeline():
    X = np.ones(shape=(10, 2), dtype=int)
    samples_transform = [mario.Sample(data, key=str(i)) for i, data in enumerate(X)]
    with tempfile.TemporaryDirectory() as d:
        bag_transformer = mario.ToDaskBag()
        estimator = mario.wrap(["dask"], _build_transformer(d, 0), transform_tag="CPU")
        X_tr = estimator.transform(bag_transformer.transform(samples_transform))
        assert len(mario.dask_tags(estimator)) == 1
        assert len(X_tr.compute(scheduler="single-threaded")) == 10


def test_checkpoint_transform_pipeline_with_sampleset():
    def _run(dask_enabled):

        X = np.ones(shape=(10, 2), dtype=int)
        samples_transform = mario.SampleSet(
            [mario.Sample(data, key=str(i)) for i, data in enumerate(X)], key="1"
        )
        offset = 2
        oracle = X + offset

        with tempfile.TemporaryDirectory() as d:
            pipeline = Pipeline(
                [(f"{i}", _build_transformer(d, i)) for i in range(offset)]
            )
            if dask_enabled:
                pipeline = mario.wrap(["dask"], pipeline)
                transformed_samples = pipeline.transform([samples_transform]).compute(
                    scheduler="single-threaded"
                )
            else:
                transformed_samples = pipeline.transform([samples_transform])

            _assert_all_close_numpy_array(
                oracle,
                [s.data for sample_set in transformed_samples for s in sample_set],
            )
            assert np.all([len(s) == 10 for s in transformed_samples])

    _run(dask_enabled=True)
    _run(dask_enabled=False)
