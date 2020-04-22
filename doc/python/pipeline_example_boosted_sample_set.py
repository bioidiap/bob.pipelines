from bob.pipelines.sample import Sample, SampleSet
from bob.pipelines.mixins import SampleMixin, mix_me_up
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline
import numpy


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

# Creating X 
X1 = numpy.zeros((2, 2))
X2 = numpy.ones((2, 2))

# Wrapping X with Samples
X1_as_sample = Sample(X1, metadata=1)
X2_as_sample = Sample(X2, metadata=1)

X_sample_set = SampleSet([X1_as_sample, X2_as_sample], class_name=1)

# Building an arbitrary pipeline
pipeline = make_pipeline(
    MyBoostedTransformer(transform_extra_arguments=(("metadata", "metadata"),)),
    MyBoostedTransformer(transform_extra_arguments=(("metadata", "metadata"),)),
)
X_transformed = pipeline.transform([X_sample_set])