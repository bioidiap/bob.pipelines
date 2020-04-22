from bob.pipelines.sample import Sample
from bob.pipelines.mixins import SampleMixin, mix_me_up
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline
import numpy


class MyTransformer(TransformerMixin, BaseEstimator):
    def transform(self, X, metadata=None):
        # Transform `X` with metadata
        if metadata is None:
            return X
        return [x + m["offset"] for x, m in zip(X, metadata)]

    def fit(self, X):
        pass

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

# Creating X 
X = numpy.zeros((2, 2))

# Building an arbitrary pipeline
pipeline = make_pipeline(MyTransformer(), MyTransformer())

X_transformed = pipeline.transform([X])
