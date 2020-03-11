#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ..sample import Sample, DelayedSample
import os
import cloudpickle
import functools
import bob.io.base
from sklearn.preprocessing import FunctionTransformer
import six
from sklearn.pipeline import Pipeline


def dask_it(o):
    """
    Mix up any :py:class:`sklearn.pipeline.Pipeline` or :py:class:`sklearn.estimator.Base with
    :py:class`DaskEstimatorMixin`
    """

    return mix_me_up(DaskEstimatorMixin, o)


def mix_me_up(bases, o):
    """
    Dynamically creates a new class from :any:`object` or :any:`class` using `cls` a base classes.
    For instance, mix_me_up((class_A, classB), class_c) is equal to `class ABC(A,B,C) pass:`
    
    Example
    -------

       >>> my_mixed_class = mix_me_up([MixInA, MixInB], OriginalClass)
       >>> mixed_object = my_mixed_class(*args)

    It's also possible to mix up an instance:

    Example
    -------

       >>> instance = OriginalClass()
       >>> mixed_object = mix_me_up([MixInA, MixInB], instance)       

    It's also possible to mix up a :py:class:`sklearn.pipeline.Pipeline`.
    In this case, every estimator inside of :py:meth:`sklearn.pipeline.Pipeline.steps`
    will be mixed up


    Parameters
    ----------
      bases:  or :any:`tuple` 
        Base classes to be mixed in
      o: :any:`class`, :any:`object` or :py:class:`sklearn.pipeline.Pipeline`
        Base element to be extended

    """
 
    def _mix(bases, o):
        bases = bases if isinstance(bases, tuple) else tuple([bases])
        class_name = ''.join([c.__name__ for c in bases])
        if isinstance(o, six.class_types):            
            # If it's a class, just merge them
            class_name += o.__name__
            new_type = type(class_name, bases+tuple([o]), {})
        else:            
            # If it's an object, creates a new class and copy the state of the current object
            class_name += o.__class__.__name__
            new_type = type(
                class_name, bases+tuple([o.__class__]), o.__dict__
            )()
            # new_type.__dict__ is made in the descending order of the classes
            # so the values of o.__dict__ are overwritten by the lower ones
            # here we are copying them back
            for k in o.__dict__:
                new_type.__dict__[k] = o.__dict__[k]
        return new_type

    # If it is a scikit pipeline, mixIN everything inside of
    # Pipeline.steps
    if isinstance(o, Pipeline):
        # mixing all pipelines
        for i in range(len(o.steps)):
            o.steps[i] = (str(i), dask_it(o.steps[i][1]))
        return o
    else:
        return _mix(bases, o)

    

    #def _mix_all(head):
    #    final_cls = head
    #    # We want the classed to be integrated in the reversed order
    #    # Because this how it's done here: class AB(A,B) pass:
    #    for c in cls[::-1]:
    #        final_cls = _mix_2(c, final_cls)
    #    return final_cls



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
            raise ValueError("`features_dir` is not in %s" % CheckpointMixin.__name__)

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
            return cloudpickle.load(f)

    def save_model(self):
        if _is_estimator_stateless(self) or self.model_path is None:
            return self
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            cloudpickle.dump(self, f)
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

    def __init__(self, callable=None):
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

    Parameters
    ----------

       fit_resource: str
           Mark the delayed(self.fit) with this value. This can be used in 
           a future delayed(self.fit).compute(resources=resource_tape) so
           dask scheduler can place this task in a particular resource
           (e.g GPU)

       transform_resource: str
           Mark the delayed(self.transform) with this value. This can be used in 
           a future delayed(self.transform).compute(resources=resource_tape) so
           dask scheduler can place this task in a particular resource
           (e.g GPU)

    """

    def __init__(self, fit_resource=None, transform_resource=None, **kwargs):
        super().__init__(**kwargs)
        self._dask_state = self
        self.resource_tape = dict()
        self.fit_resource = fit_resource
        self.transform_resource = transform_resource


    def fit(self, X, y=None, **fit_params):
        self._dask_state = delayed(super().fit)(X, y, **fit_params)
        if self.fit_resource is not None:
            self.resource_tape[self._dask_state] = self.fit_resource

        return self


    def transform(self, X):
        def _transf(X_line, dask_state):
            # return dask_state.transform(X_line)
            return super(DaskEstimatorMixin, dask_state).transform(X_line)

        map_partitions = X.map_partitions(_transf, self._dask_state)
        if self.transform_resource is not None:
            self.resource_tape[map_partitions] = self.transform_resource

        return map_partitions


def _is_estimator_stateless(estimator):
    if not hasattr(estimator, "_get_tags"):
        return False
    return estimator._get_tags()["stateless"]
