# vim: set fileencoding=utf-8 :

from .sample import Sample, DelayedSample, SampleSet
import os
import types
import cloudpickle
import functools
import bob.io.base
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from dask import delayed
import dask.bag


def estimator_dask_it(
    o, fit_tag=None, transform_tag=None, npartitions=None,
):
    """
    Mix up any :py:class:`sklearn.pipeline.Pipeline` or :py:class:`sklearn.estimator.Base` with
    :py:class`DaskEstimatorMixin`

    Parameters
    ----------

      o: :py:class:`sklearn.pipeline.Pipeline` or :py:class:`sklearn.estimator.Base`
        Any :py:class:`sklearn.pipeline.Pipeline` or :py:class:`sklearn.estimator.Base` to be dask mixed

      fit_tag: list(tuple()) or "str"
         Tag the `fit` method. This is useful to tag dask tasks to run in specific workers https://distributed.dask.org/en/latest/resources.html
         If `o` is :py:class:`sklearn.pipeline.Pipeline`, this parameter should contain a list of tuples
         containing the pipeline.step index and the `str` tag for `fit`.
         If `o` is :py:class:`sklearn.estimator.Base`, this parameter should contain just the tag for `fit`


      transform_tag: list(tuple()) or "str"
         Tag the `fit` method. This is useful to tag dask tasks to run in specific workers https://distributed.dask.org/en/latest/resources.html
         If `o` is :py:class:`sklearn.pipeline.Pipeline`, this parameter should contain a list of tuples
         containing the pipeline.step index and the `str` tag for `transform`.
         If `o` is :py:class:`sklearn.estimator.Base`, this parameter should contain just the tag for `transform`


    Examples
    --------

      Vanilla example

      >>> pipeline = estimator_dask_it(pipeline) # Take some pipeline and make the methods `fit`and `transform` run over dask
      >>> pipeline.fit(samples).compute()


      In this example we will "mark" the fit method with a particular tag
      Hence, we can set the `dask.delayed.compute` method to place some
      delayeds to be executed in particular resources

      >>> pipeline = estimator_dask_it(pipeline, fit_tag=[(1, "GPU")]) # Take some pipeline and make the methods `fit`and `transform` run over dask
      >>> fit = pipeline.fit(samples)
      >>> fit.compute(resources=pipeline.dask_tags())

      Taging estimator
      >>> estimator = estimator_dask_it(estimator)
      >>> transf = estimator.transform(samples)
      >>> transf.compute(resources=estimator.dask_tags())

    """

    def _fetch_resource_tape(self):
        """
        Get all the resources take
        """
        resource_tags = dict()
        if isinstance(self, Pipeline):
            for i in range(1, len(self.steps)):
                resource_tags.update(o[i].resource_tags)
        else:
            resource_tags.update(self.resource_tags)

        return resource_tags

    if isinstance(o, Pipeline):
        # Adding a daskbag in the tail of the pipeline
        o.steps.insert(0, ("0", DaskBagMixin(npartitions=npartitions)))

    # Patching dask_resources
    dasked = mix_me_up([DaskEstimatorMixin], o,)

    # Tagging each element in a pipeline
    if isinstance(o, Pipeline):

        # Tagging each element for fitting and transforming
        if fit_tag is not None:
            for index, tag in fit_tag:
                o.steps[index][1].fit_tag = tag
        else:
            for estimator in o.steps:
                estimator[1].fit_tag = fit_tag

        if transform_tag is not None:
            for index, tag in transform_tag:
                o.steps[index][1].transform_tag = tag
        else:
            for estimator in o.steps:
                estimator[1].transform_tag = transform_tag

        for estimator in o.steps:
            estimator[1].resource_tags = dict()
    else:
        dasked.fit_tag = fit_tag
        dasked.transform_tag = transform_tag
        dasked.resource_tags = dict()

    # Bounding the method
    dasked.dask_tags = types.MethodType(_fetch_resource_tape, dasked)

    return dasked


def mix_me_up(bases, o):
    """
    Dynamically creates a new class from :any:`object` or :any:`class`.
    For instance, mix_me_up((A,B), class_c) is equal to `class ABC(A,B,C) pass:`

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
        bases = tuple(bases)
        class_name = "".join([c.__name__ for c in bases])
        if isinstance(o, types.ClassType):
            # If it's a class, just merge them
            class_name += o.__name__
            new_type = type(class_name, bases + tuple([o]), {})
        else:
            # If it's an object, creates a new class and copy the state of the current object
            class_name += o.__class__.__name__
            new_type = type(class_name, bases + tuple([o.__class__]), o.__dict__)
            new_type = new_type.__new__(new_type)
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
            # checking if it's not the bag transformer
            if isinstance(o.steps[i][1], DaskBagMixin):
                continue
            o.steps[i] = (str(i), _mix(bases, o.steps[i][1]))
        return o
    else:
        return _mix(bases, o)


def _is_estimator_stateless(estimator):
    if not hasattr(estimator, "_get_tags"):
        return False
    return estimator._get_tags()["stateless"]


def _make_kwargs_from_samples(samples, arg_attr_list):
    kwargs = {arg: [getattr(s, attr) for s in samples] for arg, attr in arg_attr_list}
    return kwargs


class SampleMixin:
    """Mixin class to make scikit-learn estimators work in :any:`Sample`-based
    pipelines.
    Do not use this class except for scikit-learn estimators.

    .. todo::

        Also implement ``predict``, ``predict_proba``, and ``score``. See:
        https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects

    Attributes
    ----------
    fit_extra_arguments : [tuple], optional
        Use this option if you want to pass extra arguments to the fit method of the
        mixed instance. The format is a list of two value tuples. The first value in
        tuples is the name of the argument that fit accepts, like ``y``, and the second
        value is the name of the attribute that samples carry. For example, if you are
        passing samples to the fit method and want to pass ``subject`` attributes of
        samples as the ``y`` argument to the fit method, you can provide ``[("y",
        "subject")]`` as the value for this attribute.
    transform_extra_arguments : [tuple], optional
        Similar to ``fit_extra_arguments`` but for the transform method.
    """

    def __init__(
        self, transform_extra_arguments=None, fit_extra_arguments=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.transform_extra_arguments = transform_extra_arguments or tuple()
        self.fit_extra_arguments = fit_extra_arguments or tuple()

    def transform(self, samples):

        # Transform either samples or samplesets
        if isinstance(samples[0], Sample) or isinstance(samples[0], DelayedSample):
            kwargs = _make_kwargs_from_samples(samples, self.transform_extra_arguments)
            features = super().transform([s.data for s in samples], **kwargs)
            new_samples = [Sample(data, parent=s) for data, s in zip(features, samples)]
            return new_samples
        elif isinstance(samples[0], SampleSet):
            return [
                SampleSet(self.transform(sset.samples), parent=sset) for sset in samples
            ]
        else:
            raise ValueError("Type for sample not supported %s" % type(samples))

    def fit(self, samples, y=None):

        # See: https://scikit-learn.org/stable/developers/develop.html
        # if the estimator does not require fit or is stateless don't call fit
        tags = self._get_tags()
        if tags["stateless"] or not tags["requires_fit"]:
            return self

        # if the estimator needs to be fitted.
        kwargs = _make_kwargs_from_samples(samples, self.fit_extra_arguments)
        X = [s.data for s in samples]
        return super().fit(X, **kwargs)


class CheckpointMixin:
    """Mixin class that allows :any:`Sample`-based estimators save their results into
    disk."""

    def __init__(
        self,
        model_path=None,
        features_dir=None,
        extension=".h5",
        save_func=None,
        load_func=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.features_dir = features_dir
        self.extension = extension
        self.save_func = save_func or bob.io.base.save
        self.load_func = load_func or bob.io.base.load

    def transform_one_sample(self, sample):

        # Check if the sample is already processed.
        path = self.make_path(sample)
        if path is None or not os.path.isfile(path):
            new_sample = super().transform([sample])[0]
            # save the new sample
            self.save(new_sample)
        else:
            # Setting the solved path to the sample
            sample.path = path
            new_sample = self.load(sample)

        return new_sample

    def transform_one_sample_set(self, sample_set):
        samples = [self.transform_one_sample(s) for s in sample_set.samples]
        return SampleSet(samples, parent=sample_set)

    def transform(self, samples):
        if not isinstance(samples, list):
            raise ValueError("It's expected a list, not %s" % type(samples))

        if isinstance(samples[0], Sample) or isinstance(samples[0], DelayedSample):
            return [self.transform_one_sample(s) for s in samples]
        elif isinstance(samples[0], SampleSet):
            return [self.transform_one_sample_set(s) for s in samples]
        else:
            raise ValueError("Type not allowed %s" % type(samples[0]))

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

        return os.path.join(self.features_dir, str(sample.key) + self.extension)

    def save(self, sample):
        if isinstance(sample, Sample):
            path = self.make_path(sample)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return self.save_func(sample.data, path)
        elif isinstance(sample, SampleSet):
            for s in sample.samples:
                path = self.make_path(s)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                return self.save_func(s.data, path)
        else:
            raise ValueError("Type for sample not supported %s" % type(sample))

    def load(self, sample):
        # because we are checkpointing, we return a DelayedSample
        # instead of a normal (preloaded) sample. This allows the next
        # phase to avoid loading it would it be unnecessary (e.g. next
        # phase is already check-pointed)
        return DelayedSample(
            functools.partial(self.load_func, sample.path), parent=sample
        )

    def load_model(self):
        if _is_estimator_stateless(self):
            return self
        with open(self.model_path, "rb") as f:
            model = cloudpickle.load(f)
            self.__dict__.update(model.__dict__)
            return model

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


class NonPicklableMixin:
    """Class that wraps estimators that are not picklable

    Example
    -------
        >>> from bob.pipelines.processor import NonPicklableMixin
        >>> wrapper = NonPicklableMixin(my_non_picklable_class_callable)

    Example
    -------
        >>> from bob.pipelines.processor import NonPicklableMixin
        >>> import functools
        >>> wrapper = NonPicklableMixin(functools.partial(MyNonPicklableClass, arg1, arg2))


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

    def __init__(self, fit_tag=None, transform_tag=None, **kwargs):
        super().__init__(**kwargs)
        self._dask_state = self
        self.resource_tags = dict()
        self.fit_tag = fit_tag
        self.transform_tag = transform_tag

    def fit(self, X, y=None, **fit_params):
        self._dask_state = delayed(super().fit)(X, y, **fit_params)
        if self.fit_tag is not None:
            self.resource_tags[self._dask_state] = self.fit_tag

        return self

    def transform(self, X):
        def _transf(X_line, dask_state):
            return super(DaskEstimatorMixin, dask_state).transform(X_line)

        map_partitions = X.map_partitions(_transf, self._dask_state)
        if self.transform_tag is not None:
            self.resource_tags[map_partitions] = self.transform_tag

        return map_partitions


class DaskBagMixin(TransformerMixin):
    """Transform an arbitrary iterator into a :py:class:`dask.bag`


    Paramters
    ---------

      npartitions: int
        Number of partitions used it :py:meth:`dask.bag.npartitions`


    Example
    -------

    >>> transformer = DaskBagMixin()
    >>> dask_bag = transformer.transform([1,2,3])
    >>> dask_bag.map_partitions.....

    """

    def __init__(self, npartitions=None, **kwargs):
        super().__init__(**kwargs)
        self.npartitions = npartitions

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, **kwargs):
        return dask.bag.from_sequence(X, npartitions=self.npartitions)
