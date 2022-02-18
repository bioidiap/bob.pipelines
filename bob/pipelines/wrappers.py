"""Scikit-learn Estimator Wrappers."""
import logging
import os
import tempfile

from functools import partial
from pathlib import Path

import cloudpickle
import dask.array as da
import dask.bag
import numpy as np

from dask import delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import bob.io.base

from .sample import DelayedSample, SampleBatch, SampleSet
from .utils import is_estimator_stateless, isinstance_nested

logger = logging.getLogger(__name__)


def _frmt(estimator, limit=30):
    # default value of limit is chosen so the log can be seen in dask graphs
    def _n(e):
        return e.__class__.__name__.replace("Wrapper", "")

    name = ""
    while hasattr(estimator, "estimator"):
        name += f"{_n(estimator)}|"
        estimator = estimator.estimator

    if (
        isinstance(estimator, FunctionTransformer)
        and type(estimator) is FunctionTransformer
    ):
        name += str(estimator.func.__name__)
    else:
        name += str(estimator)

    name = f"{name:.{limit}}"
    return name


def copy_learned_attributes(from_estimator, to_estimator):
    attrs = {k: v for k, v in vars(from_estimator).items() if k.endswith("_")}

    for k, v in attrs.items():
        setattr(to_estimator, k, v)


def get_bob_tags(estimator=None, force_tags=None):
    """Returns the default tags of a Transformer unless forced or specified.

    Relies on the tags API of sklearn to set and retrieve the tags.

    Specify an estimator tag values with ``estimator._more_tags``:

    ```
    class My_annotator_transformer(sklearn.base.BaseEstimator):
        def _more_tags(self):
            return {"bob_output": "annotations"}
    ```

    The returned tags will take their value with the following priority:

    1. key:value in `force_tags`, if it is present;
    2. key:value in `estimator` tags (set with `estimator._more_tags()`) if it exists;
    3. the default value for that tag if none of the previous exist.

    Tags format
    -----------
    bob_input: str
        The Sample attribute passed to the first argument of the fit or transform method.
        Example:
            `{"bob_input": ("annotations")}`
        will result in:
            `estimator.transform(sample.annotations)`
        Default:
            `{"bob_input": "data"}`
    bob_transform_extra_input: tuple of str
        Each element of the tuple is a str representing an attribute of a Sample
        object. Each attribute of the sample will be passed as argument to the transform
        method in that order.
        Example:
            `{"bob_transform_extra_input": (("arg_0","data"), ("arg_1","annotations"))}`
        will result in:
            `estimator.transform(sample.data, arg_0=sample.data, arg_1=sample.annotations)`
        Default:
            `{"bob_transform_extra_input": tuple()}`
    bob_fit_extra_input: tuple of str
        Each element of the tuple is a str representing an attribute of a Sample
        object. Each attribute of the sample will be passed as argument to the fit
        method in that order.
        Example:
            `{"bob_fit_extra_input": (("y", "annotations"), ("extra "metadata"))}`
        will result in:
            `estimator.fit(sample.data, y=sample.annotations, extra=sample.metadata)`
        Default:
            `{"bob_fit_extra_input": tuple()}`
    bob_output: str
        The Sample attribute in which the output of the transform is stored.
        Default:
            `{"bob_output": "data"}`
    bob_checkpoint_extension: str
        The extension of each checkpoint file.
        Default:
            `{"bob_checkpoint_extension": ".h5"}`
    bob_features_save_fn: func
        The function used to save each checkpoint file.
        Default:
            `{"bob_features_save_fn": bob.io.base.save}`
    bob_features_load_fn: func
        The function used to load each checkpoint file.
        Default:
            `{"bob_features_load_fn": bob.io.base.load}`
    bob_fit_supports_dask_array: bool
        Indicates that the fit method of that estimator accepts dask arrays as input.
        Default:
            `{"bob_fit_supports_dask_array": False}`

    Parameters
    ----------
    estimator: sklearn.BaseEstimator or None
        An estimator class with tags that will overwrite the default values. Setting to
        None will return the default values of every tags.
    force_tags: dict[str, Any] or None
        Tags with a non-default value that will overwrite the default and the estimator
        tags.

    Returns
    -------
    dict[str, Any]
        The resulting tags with a value (either specified in the estimator, forced by
        the arguments, or default)
    """
    force_tags = force_tags or {}
    default_tags = {
        "bob_input": "data",
        "bob_transform_extra_input": tuple(),
        "bob_fit_extra_input": tuple(),
        "bob_output": "data",
        "bob_checkpoint_extension": ".h5",
        "bob_features_save_fn": bob.io.base.save,
        "bob_features_load_fn": bob.io.base.load,
        "bob_fit_supports_dask_array": False,
    }
    estimator_tags = estimator._get_tags() if estimator is not None else {}
    return {**default_tags, **estimator_tags, **force_tags}


class BaseWrapper(MetaEstimatorMixin, BaseEstimator):
    """The base class for all wrappers."""

    def _more_tags(self):
        return self.estimator._more_tags()


def _make_kwargs_from_samples(samples, arg_attr_list):
    kwargs = {
        arg: [getattr(s, attr) for s in samples] for arg, attr in arg_attr_list
    }
    return kwargs


def _check_n_input_output(samples, output, func_name):
    ls, lo = len(samples), len(output)
    if ls != lo:
        raise RuntimeError(
            f"{func_name} got {ls} samples but returned {lo} samples!"
        )


class DelayedSamplesCall:
    def __init__(
        self, func, func_name, samples, sample_attribute="data", **kwargs
    ):
        super().__init__(**kwargs)
        self.func = func
        self.func_name = func_name
        self.samples = samples
        self.output = None
        self.sample_attribute = sample_attribute

    def __call__(self, index):
        if self.output is None:
            # Isolate invalid samples (when previous transformers returned None)
            invalid_ids = [
                i for i, s in enumerate(self.samples) if s.data is None
            ]
            valid_samples = [s for s in self.samples if s.data is not None]
            # Process only the valid samples
            if len(valid_samples) > 0:
                X = SampleBatch(
                    valid_samples, sample_attribute=self.sample_attribute
                )
                self.output = self.func(X)
                _check_n_input_output(
                    valid_samples, self.output, self.func_name
                )
            if self.output is None:
                self.output = [None] * len(valid_samples)
            # Rebuild the full batch of samples (include the previously failed)
            if len(invalid_ids) > 0:
                self.output = list(self.output)
                for i in invalid_ids:
                    self.output.insert(i, None)
        return self.output[index]


class SampleWrapper(BaseWrapper, TransformerMixin):
    """Wraps scikit-learn estimators to work with :any:`Sample`-based
    pipelines.

    Do not use this class except for scikit-learn estimators.

    Attributes
    ----------
    estimator
        The scikit-learn estimator that is wrapped.
    fit_extra_arguments : [tuple]
        Use this option if you want to pass extra arguments to the fit method of the
        mixed instance. The format is a list of two value tuples. The first value in
        tuples is the name of the argument that fit accepts, like ``y``, and the second
        value is the name of the attribute that samples carry. For example, if you are
        passing samples to the fit method and want to pass ``subject`` attributes of
        samples as the ``y`` argument to the fit method, you can provide ``[("y",
        "subject")]`` as the value for this attribute.
    output_attribute : str
        The name of a Sample attribute where the output of the estimator will be
        saved to [Default is ``data``]. For example, if ``output_attribute`` is
        ``"annotations"``, then ``sample.annotations`` will contain the output of
        the estimator.
    transform_extra_arguments : [tuple]
        Similar to ``fit_extra_arguments`` but for the transform and other similar
        methods.
    """

    def __init__(
        self,
        estimator,
        transform_extra_arguments=None,
        fit_extra_arguments=None,
        output_attribute=None,
        input_attribute=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.estimator = estimator

        bob_tags = get_bob_tags(self.estimator)
        self.input_attribute = input_attribute or bob_tags["bob_input"]
        self.transform_extra_arguments = (
            transform_extra_arguments or bob_tags["bob_transform_extra_input"]
        )
        self.fit_extra_arguments = (
            fit_extra_arguments or bob_tags["bob_fit_extra_input"]
        )
        self.output_attribute = output_attribute or bob_tags["bob_output"]

    def _samples_transform(self, samples, method_name):
        # Transform either samples or samplesets
        method = getattr(self.estimator, method_name)
        logger.debug(f"{_frmt(self)}.{method_name}")
        func_name = f"{self}.{method_name}"

        if isinstance(samples[0], SampleSet):
            return [
                SampleSet(
                    self._samples_transform(sset.samples, method_name),
                    parent=sset,
                )
                for sset in samples
            ]
        else:
            kwargs = _make_kwargs_from_samples(
                samples, self.transform_extra_arguments
            )
            delayed = DelayedSamplesCall(
                partial(method, **kwargs),
                func_name,
                samples,
                sample_attribute=self.input_attribute,
            )
            if self.output_attribute != "data":
                # Edit the sample.<output_attribute> instead of data
                for i, s in enumerate(samples):
                    setattr(s, self.output_attribute, delayed(i))
                new_samples = samples
            else:
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
        if y is not None:
            raise TypeError(
                "We don't accept `y` in fit arguments because `y` should be part of "
                "the sample. To pass `y` to the wrapped estimator, use "
                "`fit_extra_arguments` tag."
            )

        if is_estimator_stateless(self.estimator):
            return self

        # if the estimator needs to be fitted.
        logger.debug(f"{_frmt(self)}.fit")
        kwargs = _make_kwargs_from_samples(samples, self.fit_extra_arguments)

        X = SampleBatch(samples, sample_attribute=self.input_attribute)

        self.estimator = self.estimator.fit(X, **kwargs)
        copy_learned_attributes(self.estimator, self)
        return self


class CheckpointWrapper(BaseWrapper, TransformerMixin):
    """Wraps :any:`Sample`-based estimators so the results are saved in
    disk.

    Parameters
    ----------

    estimator
       The scikit-learn estimator to be wrapped.

    model_path: str
       Saves the estimator state in this directory if the `estimator` is stateful

    features_dir: str
       Saves the transformed data in this directory

    extension: str
       Default extension of the transformed features.
       If None, will use the ``bob_checkpoint_extension`` tag in the estimator, or
       default to ``.h5``.

    save_func
       Pointer to a customized function that saves transformed features to disk.
       If None, will use the ``bob_feature_save_fn`` tag in the estimator, or default
       to ``bob.io.base.save``.

    load_func
       Pointer to a customized function that loads transformed features from disk.
       If None, will use the ``bob_feature_load_fn`` tag in the estimator, or default
       to ``bob.io.base.load``.

    sample_attribute: str
       Defines the payload attribute of the sample.
       If None, will use the ``bob_output`` tag in the estimator, or default to
       ``data``.

    hash_fn
       Pointer to a hash function. This hash function maps
       `sample.key` to a hash code and this hash code corresponds a relative directory
       where a single `sample` will be checkpointed.
       This is useful when is desirable file directories with less than
       a certain number of files.

    attempts
       Number of checkpoint attempts. Sometimes, because of network/disk issues
       files can't be saved. This argument sets the maximum number of attempts
       to checkpoint a sample.

    force: bool
        If True, will recompute the checkpoints even if they exists

    """

    def __init__(
        self,
        estimator,
        model_path=None,
        features_dir=None,
        extension=None,
        save_func=None,
        load_func=None,
        sample_attribute=None,
        hash_fn=None,
        attempts=10,
        force=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.force = force
        self.estimator = estimator
        self.model_path = model_path
        self.features_dir = features_dir
        self.hash_fn = hash_fn
        self.attempts = attempts

        bob_tags = get_bob_tags(self.estimator)
        self.extension = extension or bob_tags["bob_checkpoint_extension"]
        self.save_func = save_func or bob_tags["bob_features_save_fn"]
        self.load_func = load_func or bob_tags["bob_features_load_fn"]
        self.sample_attribute = sample_attribute or bob_tags["bob_output"]

        # Paths check
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
            should_compute_list = [
                self.force or p is None or not os.path.isfile(p) for p in paths
            ]
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
            _check_n_input_output(
                non_existing_samples, computed_features, method
            )
            # return computed features and checkpointed features
            features, com_feat_index = [], 0

            for s, p, should_compute in zip(
                samples, paths, should_compute_list
            ):
                if should_compute:
                    feat = computed_features[com_feat_index]
                    com_feat_index += 1
                    # save the computed feature when valid (not None)
                    if (
                        p is not None
                        and getattr(feat, self.sample_attribute) is not None
                    ):
                        self.save(feat)
                        feat = self.load(s, p)
                    features.append(feat)
                else:
                    features.append(self.load(s, p))
            return features

        if isinstance(samples[0], SampleSet):
            return [
                SampleSet(_transform_samples(s.samples), parent=s)
                for s in samples
            ]
        else:
            return _transform_samples(samples)

    def transform(self, samples):
        return self._checkpoint_transform(samples, "transform")

    def decision_function(self, samples):
        return self.estimator.decision_function(samples)

    def predict(self, samples):
        return self.estimator.predict(samples)

    def predict_proba(self, samples):
        return self.estimator.predict_proba(samples)

    def score(self, samples):
        return self.estimator.score(samples)

    def fit(self, samples, y=None):

        if is_estimator_stateless(self.estimator):
            return self

        # if the estimator needs to be fitted.
        logger.debug(f"{_frmt(self)}.fit")

        if self.model_path is not None and os.path.isfile(self.model_path):
            logger.info("Found a checkpoint for model. Loading ...")
            return self.load_model()

        self.estimator = self.estimator.fit(samples, y=y)
        copy_learned_attributes(self.estimator, self)
        return self.save_model()

    def make_path(self, sample):

        if self.features_dir is None:
            return None

        key = str(sample.key)
        if key.startswith(os.sep) or ".." in key:
            raise ValueError(
                "Sample.key values should be relative paths with no "
                f"reference to upper folders. Got: {key}"
            )

        hash_dir_name = self.hash_fn(key) if self.hash_fn is not None else ""

        return os.path.join(
            self.features_dir, hash_dir_name, key + self.extension
        )

    def save(self, sample):
        path = self.make_path(sample)
        # Gets sample.data or sample.<sample_attribute> if specified
        to_save = getattr(sample, self.sample_attribute)
        for _ in range(self.attempts):
            try:

                dirname = os.path.dirname(path)
                os.makedirs(dirname, exist_ok=True)

                # Atomic writing
                extension = "".join(Path(path).suffixes)
                with tempfile.NamedTemporaryFile(
                    dir=dirname, delete=False, suffix=extension
                ) as f:
                    self.save_func(to_save, f.name)
                    os.replace(f.name, path)

                # test loading
                self.load_func(path)
                break
            except Exception:
                pass
        else:
            raise RuntimeError(f"Could not save {to_save} doing {self}.save")

    def load(self, sample, path):
        # because we are checkpointing, we return a DelayedSample
        # instead of a normal (preloaded) sample. This allows the next
        # phase to avoid loading it would it be unnecessary (e.g. next
        # phase is already check-pointed)
        if self.sample_attribute == "data":
            return DelayedSample(partial(self.load_func, path), parent=sample)
        else:
            loaded = self.load_func(path)
            setattr(sample, self.sample_attribute, loaded)
            return sample

    def load_model(self):
        if is_estimator_stateless(self.estimator):
            return self
        with open(self.model_path, "rb") as f:
            model = cloudpickle.load(f)
            self.__dict__.update(model.__dict__)
            return self

    def save_model(self):
        if is_estimator_stateless(self.estimator) or self.model_path is None:
            return self
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            cloudpickle.dump(self, f)
        return self


def is_checkpointed(estimator):
    return isinstance_nested(estimator, "estimator", CheckpointWrapper)


def getattr_nested(estimator, attr):
    if hasattr(estimator, attr):
        return getattr(estimator, attr)
    elif hasattr(estimator, "estimator"):
        return getattr_nested(estimator.estimator, attr)
    return None


def _array_from_sample_bags(X, attribute):
    delayeds = X.to_delayed()
    lengths = X.map_partitions(lambda samples: [len(samples)]).compute()
    shapes = X.map_partitions(
        lambda samples: [[getattr(s, attribute).shape for s in samples]]
    ).compute()
    dtype, X = None, []
    for length_, shape_, delayed_samples_list in zip(lengths, shapes, delayeds):
        delayed_samples_list._length = length_
        for shape, delayed_sample in zip(shape_, delayed_samples_list):
            if dtype is None:
                dtype = np.array(
                    getattr(delayed_sample, attribute).compute()
                ).dtype
            darray = da.from_delayed(
                getattr(delayed_sample, attribute),
                shape,
                dtype=dtype,
                name=False,
            )
            X.append(darray)
    return X


class DaskWrapper(BaseWrapper, TransformerMixin):
    """Wraps Scikit estimators to handle Dask Bags as input.

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
        self,
        estimator,
        fit_tag=None,
        transform_tag=None,
        fit_supports_dask_array=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.estimator = estimator
        self._dask_state = estimator
        self.resource_tags = dict()
        self.fit_tag = fit_tag
        self.transform_tag = transform_tag
        self.fit_supports_dask_array = (
            fit_supports_dask_array
            or get_bob_tags(self.estimator)["bob_fit_supports_dask_array"]
        )

    def _make_dask_resource_tag(self, tag):
        return {tag: 1}

    def _dask_transform(self, X, method_name):
        graph_name = f"{_frmt(self)}.{method_name}"
        logger.debug(graph_name)

        def _transf(X_line, dask_state):
            return getattr(dask_state, method_name)(X_line)

        # change the name to have a better name in dask graphs
        _transf.__name__ = graph_name
        map_partitions = X.map_partitions(_transf, self._dask_state)
        if self.transform_tag:
            self.resource_tags[
                tuple(map_partitions.dask.keys())
            ] = self._make_dask_resource_tag(self.transform_tag)

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

    def _get_fit_params_from_sample_bags(self, bags):
        logger.debug("Converting dask bag to dask array")

        input_attribute = getattr_nested(self, "input_attribute")
        fit_extra_arguments = getattr_nested(self, "fit_extra_arguments")

        # convert X which is a dask bag to a dask array
        bags = bags.persist()
        X = _array_from_sample_bags(bags, input_attribute)
        kwargs = dict()
        for arg, attr in fit_extra_arguments:
            kwargs[arg] = _array_from_sample_bags(bags, attr)
        return X, kwargs

    def fit(self, X, y=None, **fit_params):
        if is_estimator_stateless(self.estimator):
            return self

        logger.debug(f"{_frmt(self)}.fit")

        if self.fit_supports_dask_array:
            if y is not None or fit_params:
                raise ValueError(
                    "y or fit_params should be passed through fit_extra_arguments of the SampleWrapper"
                )

            model_path = None
            if is_checkpointed(self):
                model_path = getattr_nested(self, "model_path")
            model_path = model_path or ""
            if not os.path.isfile(model_path):
                X, fit_params = self._get_fit_params_from_sample_bags(X)
                # the estimators are supposed to be dask (self) | [checkpoint] | sample | estimator
                estimator = self.estimator.estimator
                if is_checkpointed(self):
                    estimator = estimator.estimator
                estimator.fit(X, **fit_params)
                # if the estimator is checkpointed, we need to save the model
                if is_checkpointed(self):
                    self.estimator.save_model()
                return self
            else:
                logger.info(
                    "Ignoring conversion to dask array (checkpoint detected)"
                )

        def _fit(X, y, **fit_params):
            try:
                self.estimator = self.estimator.fit(X, y, **fit_params)
            except Exception as e:
                raise RuntimeError(
                    f"Something went wrong when fitting {self.estimator} "
                    f"from {self}"
                ) from e
            copy_learned_attributes(self.estimator, self)
            return self.estimator

        # change the name to have a better name in dask graphs
        _fit.__name__ = f"{_frmt(self)}.fit"

        self._dask_state = delayed(_fit)(X, y)

        if self.fit_tag is not None:
            # If you do `delayed(_fit)(X, y)`, two tasks are generated;
            # the `finalize-TASK` and `TASK`. With this, we make sure
            # that the two are annotated
            self.resource_tags[
                tuple(
                    [
                        f"{k}{str(self._dask_state.key)}"
                        for k in ["", "finalize-"]
                    ]
                )
            ] = self._make_dask_resource_tag(self.fit_tag)

        return self


class ToDaskBag(TransformerMixin, BaseEstimator):
    """Transform an arbitrary iterator into a :any:`dask.bag.Bag`

    Example
    -------
    >>> import bob.pipelines as mario
    >>> transformer = mario.ToDaskBag()
    >>> dask_bag = transformer.transform([1,2,3])
    >>> # dask_bag.map_partitions(...)

    Attributes
    ----------
    npartitions : int
        Number of partitions used in :any:`dask.bag.from_sequence`
    """

    def __init__(self, npartitions=None, partition_size=None, **kwargs):
        super().__init__(**kwargs)
        self.npartitions = npartitions
        self.partition_size = partition_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.debug(f"{_frmt(self)}.transform")
        if self.partition_size is None:
            return dask.bag.from_sequence(X, npartitions=self.npartitions)
        else:
            return dask.bag.from_sequence(X, partition_size=self.partition_size)

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}


def wrap(bases, estimator=None, **kwargs):
    """Wraps several estimators inside each other.

    If ``estimator`` is a pipeline, the estimators in that pipeline are wrapped.

    Parameters
    ----------
    bases : list
        A list of classes to be used to wrap ``estimator``.
    estimator : :any:`object`, optional
        An initial estimator to be wrapped inside other wrappers.
        If None, the first class will be used to initialize the estimator.
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

            # when checkpointing a pipeline, checkpoint each transformer in its own folder
            new_kwargs = dict(kwargs)
            features_dir, model_path = (
                kwargs.get("features_dir"),
                kwargs.get("model_path"),
            )
            if features_dir is not None:
                new_kwargs["features_dir"] = os.path.join(features_dir, name)
            if model_path is not None:
                new_kwargs["model_path"] = os.path.join(
                    model_path, f"{name}.pkl"
                )

            trans, leftover = _wrap(trans, **new_kwargs)
            estimator.steps[idx] = (name, trans)

        # if being wrapped with DaskWrapper, add ToDaskBag to the steps
        if DaskWrapper in bases:
            valid_params = ToDaskBag._get_param_names()
            params = {k: leftover.pop(k) for k in valid_params if k in leftover}
            dask_bag = ToDaskBag(**params)
            estimator.steps.insert(0, ("ToDaskBag", dask_bag))
    else:
        estimator, leftover = _wrap(estimator, **kwargs)

    if leftover:
        raise ValueError(f"Got extra kwargs that were not consumed: {leftover}")

    return estimator


def dask_tags(estimator):
    """Recursively collects resource_tags in dasked estimators."""
    tags = {}

    if hasattr(estimator, "estimator"):
        tags.update(dask_tags(estimator.estimator))

    if isinstance(estimator, Pipeline):
        for idx, name, trans in estimator._iter():
            tags.update(dask_tags(trans))

    if hasattr(estimator, "resource_tags"):
        tags.update(estimator.resource_tags)

    return tags
