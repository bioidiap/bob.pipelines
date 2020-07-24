import os
import tempfile

import numpy as np

from sklearn.utils.validation import check_is_fitted

import bob.pipelines as mario


def test_linearize():
    def _assert(Xt, oracle):
        assert np.allclose(Xt, oracle), (Xt, oracle)

    X = np.zeros(shape=(10, 10, 10))
    oracle = X.reshape((10, -1))

    # Test the transformer only
    transformer = mario.transformers.Linearize()
    X_tr = transformer.transform(X)
    _assert(X_tr, oracle)

    # Test wrapped in to a Sample
    samples = [mario.Sample(x, key=f"{i}") for i, x in enumerate(X)]
    transformer = mario.transformers.SampleLinearize()
    X_tr = transformer.transform(samples)
    _assert([s.data for s in X_tr], oracle)

    # Test checkpoint
    with tempfile.TemporaryDirectory() as d:
        transformer = mario.transformers.CheckpointSampleLinearize(features_dir=d)
        X_tr = transformer.transform(samples)
        _assert([s.data for s in X_tr], oracle)
        assert os.path.exists(os.path.join(d, "1.h5"))


def test_pca():

    # Test wrapped in to a Sample
    X = np.random.rand(100, 10)
    samples = [mario.Sample(data, key=str(i)) for i, data in enumerate(X)]

    # fit
    n_components = 2
    estimator = mario.transformers.SamplePCA(n_components=n_components)
    estimator = estimator.fit(samples)

    # https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
    assert check_is_fitted(estimator, "n_components_") is None

    # transform
    samples_tr = estimator.transform(samples)
    assert samples_tr[0].data.shape == (n_components,)

    # Test Checkpoining
    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "model.pkl")
        estimator = mario.transformers.CheckpointSamplePCA(
            n_components=n_components, features_dir=d, model_path=model_path
        )

        # fit
        estimator = estimator.fit(samples)
        assert check_is_fitted(estimator, "n_components_") is None
        assert os.path.exists(model_path)

        # transform
        samples_tr = estimator.transform(samples)
        assert samples_tr[0].data.shape == (n_components,)
        assert os.path.exists(os.path.join(d, samples_tr[0].key + ".h5"))
