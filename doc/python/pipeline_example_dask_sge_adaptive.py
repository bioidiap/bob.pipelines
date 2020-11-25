import numpy

from dask.distributed import Client
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline

from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster

from bob.pipelines.sample import Sample
import bob.pipelines
import os


class MyTransformer(TransformerMixin, BaseEstimator):
    def transform(self, X, metadata=None):
        # Transform `X` with metadata
        return X

    def fit(self, X, y=None):
        pass

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}


class MyFitTranformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self._fit_model = None

    def transform(self, X, metadata=None):
        # Transform `X`
        return [x @ self._fit_model for x in X]

    def fit(self, X):
        self._fit_model = numpy.array([[1, 2], [3, 4]])
        return self


# Creating X
X = numpy.zeros((2, 2))
# Wrapping X with Samples
X_as_sample = [Sample(X, key=str(i), metadata=1) for i in range(10)]

# Building an arbitrary pipeline
model_path = "./dask_tmp"
os.makedirs(model_path, exist_ok=True)
pipeline = make_pipeline(MyTransformer(), MyFitTranformer())

# Wrapping with sample, checkpoint and dask
# NOTE that pipeline.fit will run in `q_short_gpu`
pipeline = bob.pipelines.wrap(
    ["sample", "checkpoint", "dask"],
    pipeline,
    model_path=model_path,
    transform_extra_arguments=(("metadata", "metadata"),),
    fit_tag="q_short_gpu",
)

# Creating my cluster obj.
cluster = SGEMultipleQueuesCluster()
client = Client(cluster)  # Creating the scheduler


# Run the task graph in the local computer in a single tread
# NOTE THAT resources is set in .compute
X_transformed = pipeline.fit_transform(X_as_sample).compute(
    scheduler=client, resources=cluster.get_sge_resources()
)
import shutil

shutil.rmtree(model_path)
