from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from ..mixins import SampleMixin, CheckpointMixin


class SampleFunctionTransformer(SampleMixin, FunctionTransformer):
    """Class that transforms Scikit learn FunctionTransformer
    (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
    work with :any:`Sample`-based pipelines.
    """


class CheckpointSampleFunctionTransformer(
    CheckpointMixin, SampleMixin, FunctionTransformer
):
    """Class that transforms Scikit learn FunctionTransformer
    (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
    work with :any:`Sample`-based pipelines.

    Furthermore, it makes it checkpointable
    """


class StatelessPipeline(Pipeline):
    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

    def fit(self, X, y=None, **fit_params):
        return self
