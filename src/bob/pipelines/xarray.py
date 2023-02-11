import logging
import os
import random
import string

from functools import partial

import cloudpickle
import dask
import h5py
import numpy as np
import xarray as xr

from sklearn.base import BaseEstimator
from sklearn.pipeline import _name_estimators
from sklearn.utils.metaestimators import _BaseComposition

from .sample import SAMPLE_DATA_ATTRS, _ReprMixin
from .wrappers import estimator_requires_fit

logger = logging.getLogger(__name__)


def save(data, path):
    array = np.require(data, requirements=("C_CONTIGUOUS", "ALIGNED"))
    with h5py.File(path, "w") as f:
        f.create_dataset("array", data=array)


def load(path):
    with h5py.File(path, "r") as f:
        data = f["array"][()]
    return data


def _load_fn_to_xarray(load_fn, meta=None):
    if meta is None:
        meta = np.array(load_fn())

    da = dask.array.from_delayed(
        dask.delayed(load_fn)(), meta.shape, dtype=meta.dtype, name=False
    )
    try:
        dims = meta.dims
    except Exception:
        dims = None

    xa = xr.DataArray(da, dims=dims)
    return xa, meta


def _one_sample_to_dataset(sample, meta=None):
    dataset = {}
    delayed_attributes = getattr(sample, "_delayed_attributes", None) or {}
    for k in sample.__dict__:
        if (
            k in SAMPLE_DATA_ATTRS
            or k in delayed_attributes
            or k.startswith("_")
        ):
            continue
        dataset[k] = getattr(sample, k)

    meta = meta or {}

    for k in ["data"] + list(delayed_attributes.keys()):
        attr_meta = meta.get(k)
        attr_array, attr_meta = _load_fn_to_xarray(
            partial(getattr, sample, k), meta=attr_meta
        )
        meta[k] = attr_meta
        dataset[k] = attr_array

    return xr.Dataset(dataset).chunk(), meta


def samples_to_dataset(samples, meta=None, npartitions=48, shuffle=False):
    """Converts a list of samples to a dataset.

    See :ref:`bob.pipelines.dataset_pipeline`.

    Parameters
    ----------
    samples : list
        A list of :any:`Sample` or :any:`DelayedSample` objects.
    meta : ``xarray.DataArray``, optional
        An xarray.DataArray to be used as a template for data inside samples.
    npartitions : :obj:`int`, optional
        The number of partitions to partition the samples.
    shuffle : :obj:`bool`, optional
        If True, shuffles the samples (in-place) before constructing the dataset.

    Returns
    -------
    ``xarray.Dataset``
        The constructed dataset with at least a ``data`` variable.
    """
    if meta is not None and not isinstance(meta, dict):
        meta = dict(data=meta)

    delayed_attributes = getattr(samples[0], "delayed_attributes", None) or {}
    if meta is None or not all(
        k in meta for k in ["data"] + list(delayed_attributes.keys())
    ):
        dataset, meta = _one_sample_to_dataset(samples[0])

    if shuffle:
        random.shuffle(samples)

    dataset = xr.concat(
        [_one_sample_to_dataset(s, meta=meta)[0] for s in samples], dim="sample"
    )
    if npartitions is not None:
        dataset = dataset.chunk({"sample": max(1, len(samples) // npartitions)})
    return dataset


class Block(_ReprMixin):
    """A block representation in a graph.
    This class is meant to be used with :any:`DatasetPipeline`.

    Attributes
    ----------
    dataset_map : ``callable``
        A callable that transforms the input dataset into another dataset.
    estimator : object
        A scikit-learn estimator
    estimator_name : str
        Name of the estimator
    extension : str
        The extension of checkpointed features.
    features_dir : str
        The directory to save the features.
    fit_input : str or list
        A str or list of str of column names of the dataset to be given to the ``.fit``
        method.
    fit_kwargs : None or dict
        A dict of ``fit_kwargs`` to be passed to the ``.fit`` method of the estimator.
    input_dask_array : bool
        Whether the estimator takes dask arrays in its fit method or not.
    load_func : ``callable``
        A function to save the features. Defaults to ``np.load``.
    model_path : str or None
        If given, the estimator will be pickled here.
    output_dims : list
        A list of ``(dim_name, dim_size)`` tuples. If ``dim_name`` is ``None``, a new
        name is automatically generated, otherwise it should be a string. ``dim_size``
        should be a positive integer or nan for new dimensions or ``None`` for existing
        dimensions.
    output_dtype : object
        The dtype of the output of the transformer. Defaults to ``float``.
    save_func : ``callable``
        A function to save the features. Defaults to ``np.save`` with ``allow_pickle``
        set to ``False``.
    transform_input : str or list
        A str or list of str of column names of the dataset to be given to the
        ``.transform`` method.
    """

    def __init__(
        self,
        estimator=None,
        output_dtype=float,
        output_dims=((None, np.nan),),
        fit_input="data",
        transform_input="data",
        estimator_name=None,
        model_path=None,
        features_dir=None,
        extension=".hdf5",
        save_func=None,
        load_func=None,
        dataset_map=None,
        input_dask_array=False,
        fit_kwargs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.estimator = estimator
        self.output_dtype = output_dtype
        if not all(len(d) == 2 for d in output_dims):
            raise ValueError(
                "output_dims must be an iterable of size 2 tuples "
                f"(dim_name, dim_size), not {output_dims}"
            )
        self.output_dims = output_dims
        self.fit_input = fit_input
        self.transform_input = transform_input
        if estimator_name is None:
            estimator_name = _name_estimators([estimator])[0][0]
        self.estimator_name = estimator_name
        self.model_path = model_path
        self.features_dir = features_dir
        self.extension = extension
        estimator_save_fn = (
            None
            if estimator is None
            else estimator._get_tags().get("bob_features_save_fn")
        )
        estimator_load_fn = (
            None
            if estimator is None
            else estimator._get_tags().get("bob_features_load_fn")
        )
        self.save_func = save_func or estimator_save_fn or save
        self.load_func = load_func or estimator_load_fn or load
        self.dataset_map = dataset_map
        self.input_dask_array = input_dask_array
        self.fit_kwargs = fit_kwargs or {}

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @property
    def output_ndim(self):
        return len(self.output_dims) + 1

    def make_path(self, key):
        key = str(key)
        if key.startswith(os.sep) or ".." in key:
            raise ValueError(
                "Sample.key values should be relative paths with no "
                f"reference to upper folders. Got: {key}"
            )
        return os.path.join(self.features_dir, key + self.extension)

    def save(self, key, data):
        path = self.make_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # this should be save_func(data, path) so it's compatible with bob.io.base.save
        return self.save_func(data, path)

    def load(self, key):
        path = self.make_path(key)
        return self.load_func(path)


def _fit(*args, block):
    logger.info(f"Calling {block.estimator_name}.fit")
    block.estimator.fit(*args, **block.fit_kwargs)
    if block.model_path is not None:
        logger.info(f"Saving {block.estimator_name} in {block.model_path}")
        os.makedirs(os.path.dirname(block.model_path), exist_ok=True)
        with open(block.model_path, "wb") as f:
            cloudpickle.dump(block.estimator, f)
    return block.estimator


class _TokenStableTransform:
    def __init__(self, block, method_name=None, input_has_keys=False, **kwargs):
        super().__init__(**kwargs)
        self.block = block
        self.method_name = method_name or "transform"
        self.input_has_keys = input_has_keys

    def __dask_tokenize__(self):
        return (self.method_name, self.block.features_dir)

    def __call__(self, *args, estimator):
        block, method_name = self.block, self.method_name
        logger.info(f"Calling {block.estimator_name}.{method_name}")

        input_args = args[:-1] if self.input_has_keys else args
        try:
            features = getattr(estimator, self.method_name)(*input_args)
        except Exception as e:
            raise RuntimeError(
                f"Failed to transform data: {estimator}.{self.method_name}(*{input_args})"
            ) from e

        # if keys are provided, checkpoint features
        if self.input_has_keys:
            data = args[0]
            key = args[-1]

            l1, l2 = len(data), len(features)
            if l1 != l2:
                raise ValueError(
                    f"Got {l2} features from processing {l1} samples!"
                )

            # save computed_features
            logger.info(f"Saving {l2} features in {block.features_dir}")
            for feat, k in zip(features, key):
                block.save(k, feat)

        return features


def _populate_graph(graph):
    new_graph = []
    for block in graph:
        if isinstance(block, BaseEstimator):
            block = {"estimator": block}
        if isinstance(block, dict):
            block = Block(**block)
        new_graph.append(block)
    return new_graph


def _get_dask_args_from_ds(ds, columns):
    if isinstance(columns, str):
        args = [(ds[columns].data, ds[columns].dims)]
    else:
        args = []
        for c in columns:
            args.extend(_get_dask_args_from_ds(ds, c))
        args = tuple(args)
    return args


def _blockwise_with_block_args(args, block, method_name=None):
    meta = []
    for _ in range(1, block.output_ndim):
        meta = [meta]
    meta = np.array(meta, dtype=block.output_dtype)

    ascii_letters = list(string.ascii_lowercase)
    dim_map = {}

    input_arg_pairs = []
    for array, dims in args:
        dim_name = []
        for dim, dim_size in zip(dims, array.shape):
            if dim not in dim_map:
                dim_map[dim] = (ascii_letters.pop(0), dim_size)
            dim_name.append(dim_map[dim][0])
        input_arg_pairs.extend((array, "".join(dim_name)))

    # the sample dimension is always kept the same
    output_dim_name = f"{input_arg_pairs[1][0]}"
    new_axes = dict()
    for dim_name, dim_size in block.output_dims:
        if dim_name in dim_map:
            output_dim_name += dim_map[dim_name][0]
        else:
            try:
                dim_size = float(dim_size)
            except Exception:
                raise ValueError(
                    "Expected a float dim_size (positive integers or nan) for new "
                    f"dimension: {dim_name} but got: {dim_size}"
                )

            new_letter = ascii_letters.pop(0)
            if dim_name is None:
                dim_name = new_letter
            dim_map[dim_name] = (new_letter, dim_size)
            output_dim_name += new_letter
            new_axes[new_letter] = dim_size

    dims = []
    inv_map = {v[0]: k for k, v in dim_map.items()}
    for dim_name in output_dim_name:
        dims.append(inv_map[dim_name])

    output_shape = [dim_map[d][1] for d in dims]

    return output_dim_name, new_axes, input_arg_pairs, dims, meta, output_shape


def _blockwise_with_block(args, block, method_name=None, input_has_keys=False):
    (
        output_dim_name,
        new_axes,
        input_arg_pairs,
        dims,
        meta,
        _,
    ) = _blockwise_with_block_args(args, block, method_name=None)
    transform_func = _TokenStableTransform(
        block, method_name, input_has_keys=input_has_keys
    )
    transform_func.__name__ = f"{block.estimator_name}.{method_name}"

    data = dask.array.blockwise(
        transform_func,
        output_dim_name,
        *input_arg_pairs,
        meta=meta,
        new_axes=new_axes,
        concatenate=True,
        estimator=block.estimator_,
    )

    return dims, data


def _load_estimator(block):
    logger.info(f"Loading {block.estimator_name} from {block.model_path}")
    with open(block.model_path, "rb") as f:
        block.estimator = cloudpickle.load(f)
    return block.estimator


def _transform_or_load(block, ds, input_columns, mn):
    if isinstance(input_columns, str):
        input_columns = [input_columns]
    input_columns = list(input_columns) + ["key"]

    # filter dataset based on existing checkpoints
    key = np.asarray(ds["key"])
    paths = [block.make_path(k) for k in key]
    saved_samples = np.asarray([os.path.isfile(p) for p in paths])
    # compute/load features per chunk
    chunksize = ds.data.data.chunksize[0]
    for i in range(0, len(saved_samples), chunksize):
        if not np.all(saved_samples[i : i + chunksize]):
            saved_samples[i : i + chunksize] = False

    nonsaved_samples = np.logical_not(saved_samples)
    total_samples_n, saved_samples_n = len(key), saved_samples.sum()
    saved_ds = ds.sel({"sample": saved_samples})
    nonsaved_ds = ds.sel({"sample": nonsaved_samples})

    computed_data = loaded_data = None
    # compute non-saved data
    if total_samples_n - saved_samples_n > 0:
        args = _get_dask_args_from_ds(nonsaved_ds, input_columns)
        dims, computed_data = _blockwise_with_block(
            args, block, mn, input_has_keys=True
        )

    # load saved data
    if saved_samples_n > 0:
        logger.info(
            f"Might load {saved_samples_n} features of {block.estimator_name}.{mn} from disk."
        )
        args = _get_dask_args_from_ds(saved_ds, input_columns)
        dims, meta, shape = _blockwise_with_block_args(args, block, mn)[-3:]
        loaded_data = [
            dask.array.from_delayed(
                dask.delayed(block.load)(k),
                shape=shape[1:],
                meta=meta,
                name=False,
            )[None, ...]
            for k in key[saved_samples]
        ]
        loaded_data = dask.array.concatenate(loaded_data, axis=0)

    # merge loaded and computed data
    if computed_data is None:
        data = loaded_data
    elif loaded_data is None:
        data = computed_data
    else:
        # merge data chunk-based
        data = []
        i, j = 0, 0
        for k in range(0, len(saved_samples), chunksize):
            saved = saved_samples[k]
            if saved:
                pick = loaded_data[j : j + chunksize]
                j += chunksize
            else:
                pick = computed_data[i : i + chunksize]
                i += chunksize
            data.append(pick)
        data = dask.array.concatenate(data, axis=0)

    data = dask.array.rechunk(data, {0: chunksize})
    return dims, data


class DatasetPipeline(_BaseComposition):
    """A dataset-based scikit-learn pipeline.
    See :ref:`bob.pipelines.dataset_pipeline`.

    Attributes
    ----------
    graph : list
        A list of :any:`Block`'s to be applied on input dataset.
    """

    def __init__(self, graph, **kwargs):
        super().__init__(**kwargs)
        self.graph = _populate_graph(graph)

    def _transform(self, ds, do_fit=False, method_name=None):
        for i, block in enumerate(self.graph):
            if block.dataset_map is not None:
                try:
                    ds = block.dataset_map(ds)
                except Exception as e:
                    raise RuntimeError(
                        f"Could not map ds {ds}\n with {block.dataset_map}"
                    ) from e
                continue

            if do_fit:
                args = _get_dask_args_from_ds(ds, block.fit_input)
                args = [d for d, dims in args]
                estimator = block.estimator
                if not estimator_requires_fit(estimator):
                    block.estimator_ = estimator
                elif block.model_path is not None and os.path.isfile(
                    block.model_path
                ):
                    _load_estimator.__name__ = f"load_{block.estimator_name}"
                    block.estimator_ = dask.delayed(_load_estimator)(block)
                elif block.input_dask_array:
                    ds = ds.persist()
                    args = _get_dask_args_from_ds(ds, block.fit_input)
                    args = [d for d, dims in args]
                    block.estimator_ = _fit(*args, block=block)
                else:
                    _fit.__name__ = f"{block.estimator_name}.fit"
                    block.estimator_ = dask.delayed(_fit)(
                        *args,
                        block=block,
                    )

            mn = "transform"
            if i == len(self.graph) - 1:
                if do_fit:
                    break
                mn = method_name

            if block.features_dir is None:
                args = _get_dask_args_from_ds(ds, block.transform_input)
                dims, data = _blockwise_with_block(
                    args, block, mn, input_has_keys=False
                )
            else:
                dims, data = _transform_or_load(
                    block, ds, block.transform_input, mn
                )

            # replace data inside dataset
            ds = ds.copy(deep=False)
            del ds["data"]
            persisted = False
            if not np.all(np.isfinite(data.shape)):
                block.estimator_, data = dask.persist(block.estimator_, data)
                data = data.compute_chunk_sizes()
                persisted = True
            ds["data"] = (dims, data)
            if persisted:
                ds = ds.persist()

        return ds

    def fit(self, ds, y=None):
        if y is not None:
            raise ValueError()
        self._transform(ds, do_fit=True)
        return self

    def transform(self, ds):
        return self._transform(ds, method_name="transform")

    def decision_function(self, ds):
        return self._transform(ds, method_name="decision_function")

    def predict(self, ds):
        return self._transform(ds, method_name="predict")

    def predict_proba(self, ds):
        return self._transform(ds, method_name="predict_proba")

    def score(self, ds):
        return self._transform(ds, method_name="score")
