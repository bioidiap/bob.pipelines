"""Base definition of sample."""

from collections.abc import MutableSequence
from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np

from bob.io.base import vstack_features

SAMPLE_DATA_ATTRS = ("data", "samples")


def _copy_attributes(sample, parent, kwargs):
    """Copies attributes from a dictionary to self."""
    if parent is not None:
        for key in parent.__dict__:
            if key.startswith("_") or key in SAMPLE_DATA_ATTRS:
                continue

            setattr(sample, key, getattr(parent, key))

    for key, value in kwargs.items():
        if key.startswith("_") or key in SAMPLE_DATA_ATTRS:
            continue

        setattr(sample, key, value)


class _ReprMixin:
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + ", ".join(
                f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_")
            )
            + ")"
        )

    def __eq__(self, other):
        sorted_self = {
            k: v for k, v in sorted(self.__dict__.items(), key=lambda item: item[0])
        }
        sorted_other = {
            k: v for k, v in sorted(other.__dict__.items(), key=lambda item: item[0])
        }

        for s, o in zip(sorted_self, sorted_other):
            # Checking keys
            if s != o:
                return False

            # Checking values
            if isinstance(sorted_self[s], np.ndarray) and isinstance(
                sorted_self[o], np.ndarray
            ):
                if not np.allclose(sorted_self[s], sorted_other[o]):
                    return False
            else:
                if sorted_self[s] != sorted_other[o]:
                    return False

        return True


class Sample(_ReprMixin):
    """Representation of sample. A Sample is a simple container that wraps a
    data-point (see :ref:`bob.pipelines.sample`)

    Each sample must have the following attributes:

        * attribute ``data``: Contains the data for this sample


    Parameters
    ----------

        data : object
            Object representing the data to initialize this sample with.

        parent : object
            A parent object from which to inherit all other attributes (except
            ``data``)
    """

    def __init__(self, data, parent=None, **kwargs):
        self.data = data
        _copy_attributes(self, parent, kwargs)


class DelayedSample(_ReprMixin):
    """Representation of sample that can be loaded via a callable.

    The optional ``**kwargs`` argument allows you to attach more attributes to
    this sample instance.


    Parameters
    ----------

        load
            A python function that can be called parameterlessly, to load the
            sample in question from whatever medium

        parent : :any:`DelayedSample`, :any:`Sample`, None
            If passed, consider this as a parent of this sample, to copy
            information

        delayed_attributes : dict or None
            A dictionary of name : load_fn pairs that will be used to create
            attributes of name : load_fn() in this class. Use this to option
            to create more delayed attributes than just ``sample.data``.

        kwargs : dict
            Further attributes of this sample, to be stored and eventually
            transmitted to transformed versions of the sample
    """

    def __init__(self, load, parent=None, delayed_attributes=None, **kwargs):
        self.__running_init__ = True
        self._delayed_attributes = delayed_attributes
        # create the delayed attributes but leave the their values as None for now.
        if delayed_attributes is not None:
            kwargs.update({k: None for k in delayed_attributes})
        _copy_attributes(self, parent, kwargs)
        self._load = load
        del self.__running_init__

    def __getattribute__(self, name: str) -> Any:
        try:
            delayed_attributes = super().__getattribute__("_delayed_attributes")
        except AttributeError:
            delayed_attributes = None
        if delayed_attributes is None or name not in delayed_attributes:
            return super().__getattribute__(name)
        return delayed_attributes[name]()

    def __setattr__(self, name: str, value: Any) -> None:
        if name != "delayed_attributes" and "__running_init__" not in self.__dict__:
            delayed_attributes = getattr(self, "_delayed_attributes", None)
            # if setting an attribute which was delayed, remove it from delayed_attributes
            if delayed_attributes is not None and name in delayed_attributes:
                del delayed_attributes[name]

        super().__setattr__(name, value)

    @property
    def data(self):
        """Loads the data from the disk file."""
        return self._load()


class SampleSet(MutableSequence, _ReprMixin):
    """A set of samples with extra attributes"""

    def __init__(self, samples, parent=None, **kwargs):
        self.samples = samples
        _copy_attributes(self, parent, kwargs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples.__getitem__(item)

    def __setitem__(self, key, item):
        return self.samples.__setitem__(key, item)

    def __delitem__(self, item):
        return self.samples.__delitem__(item)

    def insert(self, index, item):
        # if not item in self.samples:
        self.samples.insert(index, item)


class DelayedSampleSet(SampleSet):
    """A set of samples with extra attributes"""

    def __init__(self, load, parent=None, **kwargs):
        self._load = load
        _copy_attributes(self, parent, kwargs)

    @property
    def samples(self):
        return self._load()


class DelayedSampleSetCached(DelayedSampleSet):
    """A cached version of DelayedSampleSet"""

    def __init__(self, load, parent=None, **kwargs):
        super().__init__(load, parent=parent, kwargs=kwargs)
        self._data = None
        _copy_attributes(self, parent, kwargs)

    @property
    def samples(self):
        if self._data is None:
            self._data = self._load()
        return self._data


class SampleBatch(Sequence, _ReprMixin):
    """A batch of samples that looks like [s.data for s in samples]

    However, when you call np.array(SampleBatch), it will construct a numpy array from
    sample.data attributes in a memory efficient way.
    """

    def __init__(self, samples, sample_attribute="data"):
        self.samples = samples
        self.sample_attribute = sample_attribute

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return getattr(self.samples[item], self.sample_attribute)

    def __array__(self, dtype=None, *args, **kwargs):
        def _reader(s):
            # adding one more dimension to data so they get stacked sample-wise
            return getattr(s, self.sample_attribute)[None, ...]

        arr = vstack_features(_reader, self.samples, dtype=dtype)
        return np.asarray(arr, dtype, *args, **kwargs)


def sample_to_hdf5(sample, hdf5):
    """
    Saves the content of sample to hdf5 file

    Parameters
    ----------

        sample: :any:`Sample` or :any:`DelayedSample` or :any:`list`
            Sample to be saved

        hdf5: `h5py.File`
            Pointer to a HDF5 file for writing
    """
    if isinstance(sample, list):
        for i, s in enumerate(sample):
            group = hdf5.create_group(str(i))
            sample_to_hdf5(s, group)
    else:
        for s in sample.__dict__:
            hdf5[s] = getattr(sample, s)


def hdf5_to_sample(hdf5):
    """
    Reads the content of a HDF5File and returns a :any:`Sample`

    Parameters
    ----------

        hdf5: `h5py.File`
            Pointer to a HDF5 file for reading
    """

    # Checking if it has groups
    has_groups = np.sum([isinstance(hdf5[k], h5py.Group) for k in hdf5.keys()]) > 0

    if has_groups:
        # If has groups, returns a list of Samples
        samples = []
        for k in hdf5.keys():
            group = hdf5[k]
            samples.append(hdf5_to_sample(group))
        return samples
    else:
        # If hasn't groups, returns a sample
        sample = Sample(None)
        for k in hdf5.keys():
            setattr(sample, k, hdf5[k].value)

        return sample
