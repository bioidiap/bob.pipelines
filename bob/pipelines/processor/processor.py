"""Sample-based Processors"""
from ..sample import Sample, DelayedSample
import os
import pickle
import functools
import bob.io.base
from sklearn.preprocessing import FunctionTransformer


class SampleMixin:
    """Mixin class to make scikit-learn estimators work in :any:`Sample`-based
    pipelines.

    .. todo::

        Also implement ``predict``, ``predict_proba``, and ``score``. See:
        https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects

    .. todo::

        Allow handling the targets given to the ``fit`` method.
    """

    def transform(self, samples):
        features = super().transform([s.data for s in samples])
        new_samples = [Sample(data, parent=s) for data, s in zip(features, samples)]
        return new_samples

    def fit(self, samples, y=None):
        return super().fit([s.data for s in samples])


class CheckpointMixin:
    """Mixin class that allows :any:`Sample`-based estimators save their results into
    disk."""

    def __init__(self, model_path=None, features_dir=None, extension=".h5", **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.features_dir = features_dir
        self.extension = extension

    def transform_one_sample(self, sample):

        # Check if the sample is already processed.
        path = self.make_path(sample)

        if path is None or not os.path.isfile(path):
            new_sample = super().transform([sample])[0]

            # save the new sample
            self.save(new_sample)
        else:
            new_sample = self.load(path)

        return new_sample

    def transform(self, samples):
        return [self.transform_one_sample(s) for s in samples]

    def fit(self, samples, y=None):

        if self.model_path is not None and os.path.isfile(self.model_path):
            return self.load_model()

        super().fit(samples, y=y)
        return self.save_model()

    def fit_transform(self, samples, y=None):
        return self.fit(samples, y=y).transform(samples)

    def make_path(self, sample):
        if self.features_dir is None:
            return None
        return os.path.join(self.features_dir, sample.key + self.extension)

    def recover_key_from_path(self, path):
        key = path.replace(os.path.abspath(self.features_dir), "")
        key = path[: -len(self.extension)]
        return key

    def save(self, sample):
        path = self.make_path(sample)
        return bob.io.base.save(sample.data, path, create_directories=True)

    def load(self, path):
        key = self.recover_key_from_path(path)
        # because we are checkpointing, we return a DelayedSample
        # instead of a normal (preloaded) sample. This allows the next
        # phase to avoid loading it would it be unnecessary (e.g. next
        # phase is already check-pointed)
        return DelayedSample(functools.partial(bob.io.base.load, path), key=key)

    def load_model(self):
        if _is_estimator_stateless(self):
            return self
        with open(self.model_path, "rb") as f:
            return pickle.load(f)

    def save_model(self):
        if _is_estimator_stateless(self) or self.model_path is None:
            return self
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self, f)
        return self


class SampleFunctionTransformer(SampleMixin, FunctionTransformer):
    """Mixin class that transforms Scikit learn FunctionTransformer (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
    work with :any:`Sample`-based pipelines.
    """

    pass


class CheckpointSampleFunctionTransformer(
    CheckpointMixin, SampleMixin, FunctionTransformer
):
    """Mixin class that transforms Scikit learn FunctionTransformer (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
    work with :any:`Sample`-based pipelines.

    Furthermore, it makes it checkpointable
    """

    pass


from sklearn.base import BaseEstimator
class NonPicklableWrapper:
    """Class that wraps estimators that are not picklable

    Example
    -------
        >>> from bob.pipelines.processor import NonPicklableWrapper
        >>> wrapper = NonPicklableWrapper(my_non_picklable_class_callable)

    Example
    -------
        >>> from bob.pipelines.processor import NonPicklableWrapper
        >>> import functools
        >>> wrapper = NonPicklableWrapper(functools.partial(MyNonPicklableClass, arg1, arg2))


    Parameters
    ----------
      callable: callable
         Calleble function that instantiates the scikit estimator

    """

    def __init__(self, callable):
        self.callable = callable
        self.instance = None


    def fit(self, X, y=None, **fit_params):        
        # Instantiates and do the "real" fit
        if self.instance is None:
            self.instance = self.callable()
        return self.instance.fit(X, y=y, **fit_params)


    def transform(self, X):
        
        # Instantiates and do the "real" transform
        if self.instance is None:
            self.instance = self.callable()
        return self.instance.transform(X)


from dask import delayed
class DaskEstimatorMixin:
    """Wraps Scikit estimators into Daskable objects
    """

    def fit(self, X, y=None, **fit_params):
        return delayed(super().fit)(X, y, **fit_params)




def _is_estimator_stateless(estimator):
    if not hasattr(estimator, "_get_tags"):
        return False
    return estimator._get_tags()["stateless"]
