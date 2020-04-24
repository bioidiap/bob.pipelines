# vim: set fileencoding=utf-8 :

from .sample import DelayedSample, SampleSet
import os
import types
import cloudpickle
from functools import partial
from collections import defaultdict
import bob.io.base
from sklearn.pipeline import Pipeline
from dask import delayed
import logging

logger = logging.getLogger(__name__)


def estimator_dask_it(
    o, fit_tag=None, transform_tag=None, npartitions=None,
):
    """
    Mix up any :py:class:`sklearn.pipeline.Pipeline` or :py:class:`sklearn.estimator.Base` with
    :py:class`DaskMixin`

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
        o.steps.insert(0, ("0", ToDaskBag(npartitions=npartitions)))

    # Patching dask_resources
    dasked = mix_me_up([DaskMixin], o,)

    # Tagging each element in a pipeline
    if isinstance(o, Pipeline):
        # Tagging each element for fitting and transforming
        for estimator in o.steps:
            estimator[1].fit_tag = None
            estimator[1].transform_tag = None
            estimator[1].resource_tags = dict()
            estimator[1]._dask_state = estimator[1]

        if fit_tag is not None:
            for index, tag in fit_tag:
                o.steps[index][1].fit_tag = tag

        if transform_tag is not None:
            for index, tag in transform_tag:
                o.steps[index][1].transform_tag = tag
    else:
        dasked.fit_tag = fit_tag
        dasked.transform_tag = transform_tag
        dasked.resource_tags = dict()
        dasked._dask_state = dasked

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
        class_name = "".join([c.__name__.replace("Mixin", "") for c in bases])
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
            if isinstance(o.steps[i][1], ToDaskBag):
                continue
            o.steps[i] = (str(i), _mix(bases, o.steps[i][1]))
        return o
    else:
        return _mix(bases, o)


def _is_estimator_stateless(estimator):
    if not hasattr(estimator, "_get_tags"):
        raise ValueError(
            f"Passed estimator: {estimator} does not have the _get_tags method."
        )
    # See: https://scikit-learn.org/stable/developers/develop.html
    # if the estimator does not require fit or is stateless don't call fit
    tags = estimator._get_tags()
    if tags["stateless"] or not tags["requires_fit"]:
        return True
    return False


def _make_kwargs_from_samples(samples, arg_attr_list):
    kwargs = {arg: [getattr(s, attr) for s in samples] for arg, attr in arg_attr_list}
    return kwargs


def _check_n_input_output(samples, output, func_name):
    ls, lo = len(samples), len(output)
    if ls != lo:
        raise RuntimeError(f"{func_name} got {ls} samples but returned {lo} features!")


class DelayedSamplesCall:
    def __init__(self, func, func_name, samples, data_is_np_array=False, **kwargs):
        super().__init__(**kwargs)
        self.func = func
        self.func_name = func_name
        self.samples = samples
        self.output = None
        self.data_is_np_array = data_is_np_array

    def __call__(self, index):
        if self.output is None:
            if self.data_is_np_array:
                X = bob.io.base.vstack_features(
                    lambda s: s.data, self.samples, same_size=True
                )
            else:
                X = [s.data for s in self.samples]
            self.output = self.func(X)
            _check_n_input_output(self.samples, self.output, self.func_name)
        return self.output[index]


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
        self,
        transform_extra_arguments=None,
        fit_extra_arguments=None,
        data_is_np_array=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transform_extra_arguments = transform_extra_arguments or tuple()
        self.fit_extra_arguments = fit_extra_arguments or tuple()
        self.data_is_np_array = data_is_np_array

    @staticmethod
    def _samples_transform(
        method, func_name, samples, transform_extra_arguments, data_is_np_array
    ):
        # Transform either samples or samplesets

        if isinstance(samples[0], SampleSet):
            return [
                SampleSet(
                    SampleMixin._samples_transform(
                        method, sset.samples, transform_extra_arguments
                    ),
                    parent=sset,
                )
                for sset in samples
            ]
        else:
            kwargs = _make_kwargs_from_samples(samples, transform_extra_arguments)
            delayed = DelayedSamplesCall(
                partial(method, **kwargs),
                func_name,
                samples,
                data_is_np_array=data_is_np_array,
            )
            new_samples = [
                DelayedSample(partial(delayed, index=i), parent=s)
                for i, s in enumerate(samples)
            ]
            return new_samples

    def transform(self, samples):
        method_name = "transform"
        method = getattr(super(), method_name)
        logger.info(f"Calling {self.__class__.__name__}.{method_name} from SampleMixin")
        func_name = f"{self}.{method_name}"
        return self._samples_transform(
            method=method,
            func_name=func_name,
            samples=samples,
            transform_extra_arguments=self.transform_extra_arguments,
            data_is_np_array=self.data_is_np_array,
        )

    def decision_function(self, samples):
        method_name = "decision_function"
        method = getattr(super(), method_name)
        logger.info(f"Calling {self.__class__.__name__}.{method_name} from SampleMixin")
        func_name = f"{self}.{method_name}"
        return self._samples_transform(
            method=method,
            func_name=func_name,
            samples=samples,
            transform_extra_arguments=self.transform_extra_arguments,
            data_is_np_array=self.data_is_np_array,
        )

    def fit(self, samples, y=None):

        if _is_estimator_stateless(self):
            return self

        # if the estimator needs to be fitted.
        logger.info(f"Calling {self.__class__.__name__}.fit from SampleMixin")
        kwargs = _make_kwargs_from_samples(samples, self.fit_extra_arguments)

        if self.data_is_np_array:
            X = bob.io.base.vstack_features(lambda s: s.data, samples, same_size=True)
        else:
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.features_dir = features_dir
        self.extension = extension
        self.save_func = save_func or bob.io.base.save
        self.load_func = load_func or bob.io.base.load

    @staticmethod
    def _checkpoint_transform(
        method, samples, make_path, save, load, do_checkpoint,
    ):
        # if features_dir is None, just transform all samples at once
        if not do_checkpoint:
            return method(samples)

        def _transform_samples(samples):
            paths = [make_path(s) for s in samples]
            should_compute_list = [p is None or not os.path.isfile(p) for p in paths]
            # call method on non-checkpointed samples
            non_existing_samples = [
                s
                for s, should_compute in zip(samples, should_compute_list)
                if should_compute
            ]
            # non_existing_samples could be empty
            computed_features = []
            if non_existing_samples:
                computed_features = method(non_existing_samples)
            _check_n_input_output(non_existing_samples, computed_features, method)
            # return computed features and checkpointed features
            features, com_feat_index = [], 0
            for s, p, should_compute in zip(samples, paths, should_compute_list):
                if should_compute:
                    feat = computed_features[com_feat_index]
                    features.append(feat)
                    com_feat_index += 1
                    # save the computed feature
                    if p is not None:
                        save(feat)
                else:
                    features.append(load(s, p))
            return features

        if isinstance(samples[0], SampleSet):
            return [SampleSet(_transform_samples(s.samples), parent=s) for s in samples]
        else:
            return _transform_samples(samples)

    def transform(self, samples):
        method_name = "transform"
        method = getattr(super(), method_name)
        logger.info(
            f"Calling {self.__class__.__name__}.{method_name} from CheckpointMixin"
        )
        return self._checkpoint_transform(
            method=method,
            samples=samples,
            make_path=self.make_path,
            save=self.save,
            load=self.load,
            do_checkpoint=self.features_dir is not None,
        )

    def decision_function(self, samples):
        method_name = "decision_function"
        method = getattr(super(), method_name)
        logger.info(
            f"Calling {self.__class__.__name__}.{method_name} from CheckpointMixin"
        )
        return self._checkpoint_transform(
            method=method,
            samples=samples,
            make_path=self.make_path,
            save=self.save,
            load=self.load,
            do_checkpoint=self.features_dir is not None,
        )

    def fit(self, samples, y=None):

        if _is_estimator_stateless(self):
            return self

        if self.model_path is not None and os.path.isfile(self.model_path):
            logger.info(
                f"{self.__class__.__name__}: Found a checkpoint for model. Loading ..."
            )
            return self.load_model()

        logger.info(f"Calling {self.__class__.__name__}.fit from CheckpointMixin")
        super().fit(samples, y=y)
        return self.save_model()

    def fit_transform(self, samples, y=None):
        return self.fit(samples, y=y).transform(samples)

    def make_path(self, sample):
        if self.features_dir is None:
            return None

        return os.path.join(self.features_dir, str(sample.key) + self.extension)

    def save(self, sample):
        path = self.make_path(sample)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return self.save_func(sample.data, path)

    def load(self, sample, path):
        # because we are checkpointing, we return a DelayedSample
        # instead of a normal (preloaded) sample. This allows the next
        # phase to avoid loading it would it be unnecessary (e.g. next
        # phase is already check-pointed)
        return DelayedSample(partial(self.load_func, path), parent=sample)

    def load_model(self):
        if _is_estimator_stateless(self):
            return self
        with open(self.model_path, "rb") as f:
            model = cloudpickle.load(f)
            self.__dict__.update(model.__dict__)
            return self

    def save_model(self):
        if _is_estimator_stateless(self) or self.model_path is None:
            return self
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            cloudpickle.dump(self, f)
        return self


class NonPicklableMixin:
    """Class that wraps objects that are not picklable

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
         Calleble function that instantiates the non-pickalbe function
    """

    def __init__(self, callable, **kwargs):
        super().__init__(**kwargs)
        self.callable = callable
        self._instance = None

    @property
    def instance(self):
        if self._instance is None:
            self._instance = self.callable()
        return self._instance

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_instance")
        d["_NonPicklableMixin_instance_was_None"] = self._instance is None
        return d

    def __setstate__(self, d):
        instance_was_None = d.pop("_NonPicklableMixin_instance_was_None")
        self.__dict__ = d
        self._instance = None
        if not instance_was_None:
            # access self.instance to create the instance
            self.instance


class DaskMixin:
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

    def __init__(
        self, fit_tag=None, transform_tag=None, **kwargs,
    ):
        super().__init__(**kwargs)
        self._dask_state = self
        self.resource_tags = dict()
        self.fit_tag = fit_tag
        self.transform_tag = transform_tag

    def fit(self, X, y=None, **fit_params):
        if _is_estimator_stateless(self):
            return self

        logger.info(f"Calling {self.__class__.__name__}.fit from DaskMixin")

        # change the name to have a better name in dask graphs
        fit_func = super().fit

        def _fit(X, y, **fit_params):
            return fit_func(X, y, **fit_params)

        _fit.__name__ = f"{self.__class__.__name__}.fit"
        self._dask_state = delayed(_fit)(X, y, **fit_params)
        if self.fit_tag is not None:
            self.resource_tags[self._dask_state] = self.fit_tag

        return self

    def fit_transform(self, samples, y=None):
        return self.fit(samples, y=y).transform(samples)

    @staticmethod
    def _dask_transform(
        X, _dask_state, method_name, transform_tag, resource_tags, graph_name
    ):
        def _transf(X_line, dask_state):
            return getattr(super(DaskMixin, dask_state), method_name)(X_line)

        # change the name to have a better name in dask graphs
        _transf.__name__ = graph_name
        map_partitions = X.map_partitions(_transf, _dask_state)
        if transform_tag is not None:
            resource_tags[map_partitions] = transform_tag

        return map_partitions

    def transform(self, X):
        method_name = "transform"
        graph_name = f"{self.__class__.__name__}.{method_name}"
        logger.info(f"Calling {self.__class__.__name__}.{method_name} from DaskMixin")
        return self._dask_transform(
            X,
            self._dask_state,
            method_name,
            self.transform_tag,
            self.resource_tags,
            graph_name,
        )

    def decision_function(self, X):
        method_name = "decision_function"
        graph_name = f"{self.__class__.__name__}.{method_name}"
        logger.info(f"Calling {self.__class__.__name__}.{method_name} from DaskMixin")
        return self._dask_transform(
            X,
            self._dask_state,
            method_name,
            self.transform_tag,
            self.resource_tags,
            graph_name,
        )


# import at the end of the file to prevent import loops
from .transformers import ToDaskBag
