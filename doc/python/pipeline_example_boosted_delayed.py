from bob.pipelines.sample import DelayedSample
from bob.pipelines.mixins import SampleMixin, mix_me_up
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline
import numpy
import pickle
import functools


class MyTransformer(TransformerMixin, BaseEstimator):
    def transform(self, X, metadata=None):
        # Transform `X` with metadata
        if metadata is None:
            return X
        return [x + m for x, m in zip(X, metadata)]

    def fit(self, X):
        pass

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

# Mixing up MyTransformer with the capabilities of handling Samples
MyBoostedTransformer = mix_me_up((SampleMixin,), MyTransformer)

# X is stored in the disk
X = open("delayed_sample.pkl", "rb") 

# Wrapping X with Samples
X_as_sample = DelayedSample(functools.partial(pickle.load, X), metadata=1)

# Building an arbitrary pipeline
pipeline = make_pipeline(
    MyBoostedTransformer(transform_extra_arguments=(("metadata", "metadata"),)),
    MyBoostedTransformer(transform_extra_arguments=(("metadata", "metadata"),)),
)
X_transformed = pipeline.transform([X_as_sample])