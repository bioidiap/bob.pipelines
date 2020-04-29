from bob.pipelines.sample import Sample
from bob.pipelines.mixins import SampleMixin, CheckpointMixin, mix_me_up
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline
from bob.pipelines.mixins import estimator_dask_it
import numpy
import time


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
        # Faking big processing
        time.sleep(120)
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

# Setting up SGE
Q_1DAY_GPU_SPEC = {
    "default": {
        "queue": "q_1day",
        "memory": "8GB",
        "io_big": True,
        "resource_spec": "",
        "resources": "",
    },
    "gpu": {
        "queue": "q_gpu",
        "memory": "12GB",
        "io_big": False,
        "resource_spec": "",
        "resources": {"gpu": 1},
    },
}

from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster
from dask.distributed import Client

cluster = SGEMultipleQueuesCluster(sge_job_spec=Q_1DAY_GPU_SPEC)
cluster.scale(1)  # Submitting 1 job in the q_gpu queue
cluster.adapt(minimum=1, maximum=10)
client = Client(cluster)  # Creating the scheduler

# Create a dask graph from a pipeline
# and tagging the the fit method of the second estimator to run in the GPU
dasked_pipeline = estimator_dask_it(pipeline, npartitions=5, fit_tag=[(1, "gpu")])

dasked_pipeline.fit(X_as_sample)  # Create the dask-graph for fitting
X_transformed = dasked_pipeline.transform(
    X_as_sample
)  # Create the dask graph for transform and returns a dask bag
X_transformed = X_transformed.compute(scheduler=client)  # RUN THE GRAPH

client.shutdown()
