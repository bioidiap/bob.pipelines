from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ..wrappers import wrap


def SampleFunctionTransformer(**kwargs):
    """Class that transforms Scikit learn FunctionTransformer (https://scikit-l
    earn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer
    .html) work with :any:`Sample`-based pipelines."""
    return wrap([FunctionTransformer, "sample"], **kwargs)


def CheckpointSampleFunctionTransformer(**kwargs):
    """Class that transforms Scikit learn FunctionTransformer (https://scikit-l
    earn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer
    .html) work with :any:`Sample`-based pipelines.

    Furthermore, it makes it checkpointable
    """
    return wrap([FunctionTransformer, "sample", "checkpoint"], **kwargs)


class StatelessPipeline(Pipeline):
    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

    def fit(self, X, y=None, **fit_params):
        """Does nothing"""
        return self
