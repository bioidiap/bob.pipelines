import numpy as np
import os
import tempfile
import shutil

from bob.pipelines.sample import Sample, SampleSet, DelayedSample
from bob.pipelines.processor import (
    SampleMixin,
    SampleFunctionTransformer,
    CheckpointMixin,
    CheckpointSampleFunctionTransformer,
)
from bob.pipelines.processor.processor import _is_estimator_stateless
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_array, check_is_fitted


def _offset_add_func(X, offset=1):
    return X + offset


class DummyWithFit(TransformerMixin, BaseEstimator):
    """See https://scikit-learn.org/stable/developers/develop.html and
    https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py"""

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


class DummyTransformer(TransformerMixin):
    """See https://scikit-learn.org/stable/developers/develop.html and
    https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py"""

    def transform(self, X):

        # Input validation
        X = check_array(X)
        # Check that the input is of the same shape as the one passed
        # during fit.

        return _offset_add_func(X)


class SampleDummyWithFit(SampleMixin, DummyWithFit):
    pass


class CheckpointSampleDummyWithFit(CheckpointMixin, SampleMixin, DummyWithFit):
    pass


class CheckpointSampleDummyWithFit(CheckpointMixin, SampleMixin, DummyWithFit):
    pass


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
    samples = [Sample(data) for data in X]

    transformer = SampleFunctionTransformer(
        _offset_add_func, kw_args=dict(offset=3), validate=True
    )

    features = transformer.transform(samples)
    _assert_all_close_numpy_array(X + 3, [s.data for s in features])

    features = transformer.fit_transform(samples)
    _assert_all_close_numpy_array(X + 3, [s.data for s in features])


def test_fittable_sample_transformer():

    X = np.ones(shape=(10, 2), dtype=int)
    samples = [Sample(data) for data in X]

    transformer = SampleDummyWithFit()
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
        assert isinstance(s, DelayedSample)


def test_checkpoint_function_sample_transfomer():

    X = np.arange(20, dtype=int).reshape(10, 2)
    samples = [Sample(data, key=str(i)) for i, data in enumerate(X)]
    oracle = X + 3

    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "model.pkl")
        features_dir = os.path.join(d, "features")

        transformer = CheckpointSampleFunctionTransformer(
            func=_offset_add_func,
            kw_args=dict(offset=3),
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


def test_checkpoint_fittable_sample_transformer():
    X = np.ones(shape=(10, 2), dtype=int)
    samples = [Sample(data, key=str(i)) for i, data in enumerate(X)]
    oracle = X + 1

    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "model.pkl")
        features_dir = os.path.join(d, "features")

        transformer = CheckpointSampleDummyWithFit(
            model_path=model_path, features_dir=features_dir
        )
        assert not _is_estimator_stateless(transformer)

        features = transformer.fit(samples).transform(samples)
        _assert_checkpoints(features, oracle, model_path, features_dir, False)

        features = transformer.fit_transform(samples)
        _assert_checkpoints(features, oracle, model_path, features_dir, False)
        _assert_delayed_samples(features)

        # remove all files and call fit_transform again
        shutil.rmtree(d)
        features = transformer.fit_transform(samples)
        _assert_checkpoints(features, oracle, model_path, features_dir, False)


from bob.io.base import create_directories_safe


def _build_estimator(path, i):
    base_dir = os.path.join(path, f"transformer{i}")
    create_directories_safe(base_dir)
    model_path = os.path.join(base_dir, "model.pkl")
    features_dir = os.path.join(base_dir, "features")
    return CheckpointSampleDummyWithFit(
        model_path=model_path, features_dir=features_dir
    )


def _build_transformer(path, i):
    features_dir = os.path.join(path, f"transformer{i}")
    return CheckpointSampleFunctionTransformer(
        func=_offset_add_func,
        kw_args=dict(offset=1),
        validate=True,
        features_dir=features_dir,
    )


def test_checkpoint_fittable_pipeline():

    from sklearn.pipeline import Pipeline

    X = np.ones(shape=(10, 2), dtype=int)
    samples = [Sample(data, key=str(i)) for i, data in enumerate(X)]
    samples_transform = [Sample(data, key=str(i + 10)) for i, data in enumerate(X)]
    oracle = X + 3

    with tempfile.TemporaryDirectory() as d:
        pipeline = Pipeline([(f"{i}", _build_estimator(d, i)) for i in range(2)])
        pipeline.fit(samples)

        transformed_samples = pipeline.transform(samples_transform)

        _assert_all_close_numpy_array(oracle, [s.data for s in transformed_samples])


def test_checkpoint_transform_pipeline():

    from sklearn.pipeline import Pipeline

    X = np.ones(shape=(10, 2), dtype=int)
    samples_transform = [Sample(data, key=str(i)) for i, data in enumerate(X)]
    offset = 2
    oracle = X + offset

    with tempfile.TemporaryDirectory() as d:
        pipeline = Pipeline([(f"{i}", _build_transformer(d, i)) for i in range(offset)])
        transformed_samples = pipeline.transform(samples_transform)

        _assert_all_close_numpy_array(oracle, [s.data for s in transformed_samples])


def test_checkpoint_fit_transform_pipeline():

    from sklearn.pipeline import Pipeline

    X = np.ones(shape=(1, 2), dtype=int)
    samples = [Sample(data, key=str(i)) for i, data in enumerate(X)]
    samples_transform = [Sample(data, key=str(i + 10)) for i, data in enumerate(X)]
    oracle = X + 2

    with tempfile.TemporaryDirectory() as d:
        fitter = ("0", _build_estimator(d, 0))
        transformer = ("1", _build_transformer(d, 1))
        pipeline = Pipeline([fitter, transformer], memory=d)
        pipeline.fit(samples)

        transformed_samples = pipeline.transform(samples_transform)

        _assert_all_close_numpy_array(oracle, [s.data for s in transformed_samples])
