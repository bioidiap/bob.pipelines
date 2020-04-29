from bob.pipelines.sample import Sample
from bob.pipelines.mixins import SampleMixin, CheckpointMixin, mix_me_up
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline
from bob.pipelines.mixins import estimator_dask_it
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


class MyFitTranformer(TransformerMixin, BaseEstimator):
    def __init__(self, *args, **kwargs):
        self._fit_model = None
        super().__init__(*args, **kwargs)

    def transform(self, X):
        # Transform `X`
        return [x @ self._fit_model for x in X]

    def fit(self, X):
        self._fit_model = numpy.array([[1, 2], [3, 4]])
        return self


# Mixing up MyTransformer with the capabilities of handling Samples AND checkpointing
MyBoostedTransformer = mix_me_up((CheckpointMixin, SampleMixin), MyTransformer)
MyBoostedFitTransformer = mix_me_up((CheckpointMixin, SampleMixin), MyFitTranformer)

# Creating X
X = numpy.zeros((2, 2))
# Wrapping X with Samples
X_as_sample = [Sample(X, key=str(i), metadata=1) for i in range(10)]

# Building an arbitrary pipeline
pipeline = make_pipeline(
    MyBoostedTransformer(
        transform_extra_arguments=(("metadata", "metadata"),),
        features_dir="./checkpoint_1",
    ),
    MyBoostedFitTransformer(
        features_dir="./checkpoint_2", model_path="./checkpoint_2/model.pkl",
    ),
)

# Create a dask graph from a pipeline
dasked_pipeline = estimator_dask_it(pipeline, npartitions=5)

# Run the task graph in the local computer in a single tread
X_transformed = dasked_pipeline.fit_transform(X_as_sample).compute(
    scheduler="single-threaded"
)
