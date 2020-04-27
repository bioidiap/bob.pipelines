"""Scikit-learn Estimator Wrappers"""
from .sample import DelayedSample, SampleSet
from .utils import is_estimator_stateless, samples_to_np_array
import dask.bag
from sklearn.base import TransformerMixin, BaseEstimator
import os
import cloudpickle
from functools import partial
import bob.io.base
from dask import delayed
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)


def _frmt(estimator, limit=40):
    def _n(e):
        return e.__class__.__name__.replace("Wrapper", "")

    name = ""
    while hasattr(estimator, "estimator"):
        name += f"{_n(estimator)}|"
        estimator = estimator.estimator
    name += str(estimator)
    name = f"{name:.{limit}}"
    return name


class BaseWrapper(BaseEstimator):
    def _more_tags(self):
        return self.estimator._more_tags()


def _make_kwargs_from_samples(samples, arg_attr_list):
    kwargs = {arg: [getattr(s, attr) for s in samples] for arg, attr in arg_attr_list}
    return kwargs


def _check_n_input_output(samples, output, func_name):
    ls, lo = len(samples), len(output)
    if ls != lo:
        raise RuntimeError(f"{func_name} got {ls} samples but returned {lo} features!")


class DelayedSamplesCall:
    def __init__(self, func, func_name, samples, input_is_np_array=False, **kwargs):
        super().__init__(**kwargs)
        self.func = func
        self.func_name = func_name
        self.samples = samples
        self.output = None
        self.input_is_np_array = input_is_np_array

    def __call__(self, index):
        if self.output is None:
            if self.input_is_np_array:
                X = samples_to_np_array(self.samples)
            else:
                X = [s.data for s in self.samples]
            self.output = self.func(X)
            _check_n_input_output(self.samples, self.output, self.func_name)
        return self.output[index]


class SampleWrapper(BaseWrapper, TransformerMixin):
    """Wraps scikit-learn estimators to work with :any:`Sample`-based pipelines.

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
        Similar to ``fit_extra_arguments`` but for the transform and other similar methods.
    """

    def __init__(
        self,
        estimator,
        transform_extra_arguments=None,
        fit_extra_arguments=None,
        input_is_np_array=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.estimator = estimator
        self.transform_extra_arguments = transform_extra_arguments or tuple()
        self.fit_extra_arguments = fit_extra_arguments or tuple()
        self.input_is_np_array = input_is_np_array

    def _samples_transform(self, samples, method_name):
        # Transform either samples or samplesets
        method = getattr(self.estimator, method_name)
        logger.debug(f"{_frmt(self)}.{method_name}")
        func_name = f"{self}.{method_name}"

        if isinstance(samples[0], SampleSet):
            return [
                SampleSet(
                    self._samples_transform(sset.samples, method_name), parent=sset,
                )
                for sset in samples
            ]
        else:
            kwargs = _make_kwargs_from_samples(samples, self.transform_extra_arguments)
            delayed = DelayedSamplesCall(
                partial(method, **kwargs),
                func_name,
                samples,
                input_is_np_array=self.input_is_np_array,
            )
            new_samples = [
                DelayedSample(partial(delayed, index=i), parent=s)
                for i, s in enumerate(samples)
            ]
            return new_samples

    def transform(self, samples):
        return self._samples_transform(samples, "transform")

    def decision_function(self, samples):
        return self._samples_transform(samples, "decision_function")

    def predict(self, samples):
        return self._samples_transform(samples, "predict")

    def predict_proba(self, samples):
        return self._samples_transform(samples, "predict_proba")

    def score(self, samples):
        return self._samples_transform(samples, "score")

    def fit(self, samples, y=None):

        if is_estimator_stateless(self):
            return self

        # if the estimator needs to be fitted.
        logger.debug(f"{_frmt(self)}.fit")
        kwargs = _make_kwargs_from_samples(samples, self.fit_extra_arguments)

        if self.input_is_np_array:
            X = samples_to_np_array(samples)
        else:
            X = [s.data for s in samples]

        self.estimator = self.estimator.fit(X, **kwargs)
        return self


class CheckpointWrapper(BaseWrapper, TransformerMixin):
    """Wraps :any:`Sample`-based estimators so the results are saved in disk."""

    def __init__(
        self,
        estimator,
        model_path=None,
        features_dir=None,
        extension=".h5",
        save_func=None,
        load_func=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.estimator = estimator
        self.model_path = model_path
        self.features_dir = features_dir
        self.extension = extension
        self.save_func = save_func or bob.io.base.save
        self.load_func = load_func or bob.io.base.load
        if model_path is None and features_dir is None:
            logger.warning(
                "Both model_path and features_dir are None. "
                f"Nothing will be checkpointed. From: {self}"
            )

    def _checkpoint_transform(self, samples, method_name):
        # Transform either samples or samplesets
        method = getattr(self.estimator, method_name)
        logger.debug(f"{_frmt(self)}.{method_name}")

        # if features_dir is None, just transform all samples at once
        if self.features_dir is None:
            return method(samples)

        def _transform_samples(samples):
            paths = [self.make_path(s) for s in samples]
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
                        self.save(feat)
                else:
                    features.append(self.load(s, p))
            return features

        if isinstance(samples[0], SampleSet):
            return [SampleSet(_transform_samples(s.samples), parent=s) for s in samples]
        else:
            return _transform_samples(samples)

    def transform(self, samples):
        return self._checkpoint_transform(samples, "transform")

    def decision_function(self, samples):
        return self._checkpoint_transform(samples, "decision_function")

    def predict(self, samples):
        return self._checkpoint_transform(samples, "predict")

    def predict_proba(self, samples):
        return self._checkpoint_transform(samples, "predict_proba")

    def score(self, samples):
        return self._checkpoint_transform(samples, "score")

    def fit(self, samples, y=None):

        if is_estimator_stateless(self):
            return self

        # if the estimator needs to be fitted.
        logger.debug(f"{_frmt(self)}.fit")

        if self.model_path is not None and os.path.isfile(self.model_path):
            logger.info("Found a checkpoint for model. Loading ...")
            return self.load_model()

        self.estimator = self.estimator.fit(samples, y=y)
        return self.save_model()

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
        if is_estimator_stateless(self):
            return self
        with open(self.model_path, "rb") as f:
            model = cloudpickle.load(f)
            self.__dict__.update(model.__dict__)
            return self

    def save_model(self):
        if is_estimator_stateless(self) or self.model_path is None:
            return self
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            cloudpickle.dump(self, f)
        return self


class DaskWrapper(BaseWrapper, TransformerMixin):
    """Wraps Scikit estimators to handle Dask Bags as input

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
        self, estimator, fit_tag=None, transform_tag=None, **kwargs,
    ):
        super().__init__(**kwargs)
        self.estimator = estimator
        self._dask_state = estimator
        self.resource_tags = dict()
        self.fit_tag = fit_tag
        self.transform_tag = transform_tag

    def _dask_transform(self, X, method_name):
        graph_name = f"{_frmt(self)}.{method_name}"
        logger.debug(graph_name)

        def _transf(X_line, dask_state):
            return getattr(dask_state, method_name)(X_line)

        # change the name to have a better name in dask graphs
        _transf.__name__ = graph_name
        map_partitions = X.map_partitions(_transf, self._dask_state)
        if self.transform_tag is not None:
            self.resource_tags[map_partitions] = self.transform_tag

        return map_partitions

    def transform(self, samples):
        return self._dask_transform(samples, "transform")

    def decision_function(self, samples):
        return self._dask_transform(samples, "decision_function")

    def predict(self, samples):
        return self._dask_transform(samples, "predict")

    def predict_proba(self, samples):
        return self._dask_transform(samples, "predict_proba")

    def score(self, samples):
        return self._dask_transform(samples, "score")

    def fit(self, X, y=None, **fit_params):
        if is_estimator_stateless(self):
            return self

        logger.debug(f"{_frmt(self)}.fit")

        def _fit(X, y, **fit_params):
            return self.estimator.fit(X, y, **fit_params)

        # change the name to have a better name in dask graphs
        _fit.__name__ = f"{_frmt(self)}.fit"
        self._dask_state = delayed(_fit)(X, y,)
        if self.fit_tag is not None:
            self.resource_tags[self._dask_state] = self.fit_tag

        return self


class ToDaskBag(TransformerMixin, BaseEstimator):
    """Transform an arbitrary iterator into a :py:class:`dask.bag`

    Paramters
    ---------

      npartitions: int
        Number of partitions used in :py:meth:`dask.bag.from_sequence`


    Example
    -------

    >>> transformer = DaskBagMixin()
    >>> dask_bag = transformer.transform([1,2,3])
    >>> dask_bag.map_partitions.....

    """

    def __init__(self, npartitions=None, **kwargs):
        super().__init__(**kwargs)
        self.npartitions = npartitions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.debug(f"{_frmt(self)}.transform")
        return dask.bag.from_sequence(X, npartitions=self.npartitions)

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}


def wrap(bases, estimator=None, **kwargs):
    """Wraps several estimators inside each other.

    Example
    -------

    Parameters
    ----------
    bases : list
        A list of classes to be used
    estimator : None, optional
        An initial estimator to be wrapped inside other wrappers. If None, the first class will be used to initialize the estimator.
    **kwargs
        Extra parameters passed to the init of classes.

    Returns
    -------
    object
        The wrapped estimator

    Raises
    ------
    ValueError
        If not all kwargs are consumed.
    """
    # if wrappers are passed as strings convert them to classes
    for i, w in enumerate(bases):
        if isinstance(w, str):
            bases[i] = {
                "sample": SampleWrapper,
                "checkpoint": CheckpointWrapper,
                "dask": DaskWrapper,
            }[w.lower()]

    def _wrap(estimator, **kwargs):
        # wrap the object and pass the kwargs
        for w_class in bases:
            valid_params = w_class._get_param_names()
            params = {k: kwargs.pop(k) for k in valid_params if k in kwargs}
            if estimator is None:
                estimator = w_class(**params)
            else:
                estimator = w_class(estimator, **params)
        return estimator, kwargs

    # if the estimator is a pipeline, wrap its steps instead.
    # We don't look for pipelines recursively because most of the time we
    # don't want the inner pipeline's steps to be wrapped.
    if isinstance(estimator, Pipeline):
        # wrap inner steps
        for idx, name, trans in estimator._iter():
            trans, leftover = _wrap(trans, **kwargs)
            estimator.steps[idx] = (name, trans)

        # if being wrapped with DaskWrapper, add ToDaskBag to the steps
        if DaskWrapper in bases:
            valid_params = ToDaskBag._get_param_names()
            params = {k: kwargs.pop(k) for k in valid_params if k in kwargs}
            dask_bag = ToDaskBag(**params)
            estimator.steps.insert(0, ("ToDaskBag", dask_bag))

    else:
        estimator, leftover = _wrap(estimator, **kwargs)

    if leftover:
        raise ValueError(f"Got extra kwargs that were not consumed: {kwargs}")

    return estimator


def dask_tags(estimator):
    """
    Recursively collects resource_tags in dasked estimators
    """
    tags = {}

    if hasattr(estimator, "estimator"):
        tags.update(dask_tags(estimator.estimator))

    if isinstance(estimator, Pipeline):
        for idx, name, trans in estimator._iter():
            tags.update(dask_tags(trans))

    if hasattr(estimator, "resource_tags"):
        tags.update(estimator.resource_tags)

    return tags
