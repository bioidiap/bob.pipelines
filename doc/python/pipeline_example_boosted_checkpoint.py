from bob.pipelines.sample import Sample
from bob.pipelines.mixins import SampleMixin, CheckpointMixin, mix_me_up
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


# Mixing up MyTransformer with the capabilities of handling Samples AND checkpointing
MyBoostedTransformer = mix_me_up((CheckpointMixin, SampleMixin), MyTransformer)

# Creating X
X = numpy.zeros((2, 2))
# Wrapping X with Samples
X_as_sample = Sample(X, key="1", metadata=1)

# Building an arbitrary pipeline
pipeline = make_pipeline(
    MyBoostedTransformer(
        transform_extra_arguments=(("metadata", "metadata"),),
        features_dir="./checkpoint_1",
    ),
    MyBoostedTransformer(
        transform_extra_arguments=(("metadata", "metadata"),),
        features_dir="./checkpoint_2",
    ),
)
X_transformed = pipeline.transform([X_as_sample])
