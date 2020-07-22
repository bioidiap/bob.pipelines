"""Base definition of sample."""

from collections.abc import MutableSequence
from collections.abc import Sequence

import h5py
import numpy as np

from .utils import vstack_features

SAMPLE_DATA_ATTRS = ("data", "load", "samples", "_data")


def _copy_attributes(s, d):
    """Copies attributes from a dictionary to self."""
    s.__dict__.update(dict((k, v) for k, v in d.items() if k not in SAMPLE_DATA_ATTRS))


class _ReprMixin:
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
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
        if parent is not None:
            _copy_attributes(self, parent.__dict__)
        _copy_attributes(self, kwargs)


class DelayedSample(_ReprMixin):
    """Representation of sample that can be loaded via a callable.

    The optional ``**kwargs`` argument allows you to attach more attributes to
    this sample instance.


    Parameters
    ----------

        load:
            A python function that can be called parameterlessly, to load the
            sample in question from whatever medium

        parent : :any:`DelayedSample`, :any:`Sample`, None
            If passed, consider this as a parent of this sample, to copy
            information

        kwargs : dict
            Further attributes of this sample, to be stored and eventually
            transmitted to transformed versions of the sample
    """

    def __init__(self, load, parent=None, **kwargs):
        self.load = load
        if parent is not None:
            _copy_attributes(self, parent.__dict__)
        _copy_attributes(self, kwargs)
        self._data = None

    @property
    def data(self):
        """Loads the data from the disk file."""
        if self._data is None:
            self._data = self.load()
        return self._data


class SampleSet(MutableSequence, _ReprMixin):
    """A set of samples with extra attributes"""

    def __init__(self, samples, parent=None, **kwargs):
        self.samples = samples
        if parent is not None:
            _copy_attributes(self, parent.__dict__)
        _copy_attributes(self, kwargs)

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
        self._data = None
        self.load = load
        if parent is not None:
            _copy_attributes(self, parent.__dict__)
        _copy_attributes(self, kwargs)

    @property
    def samples(self):
        if self._data is None:
            self._data = self.load()
        return self._data


class SampleBatch(Sequence, _ReprMixin):
    """A batch of samples that looks like [s.data for s in samples]

    However, when you call np.array(SampleBatch), it will construct a numpy array from
    sample.data attributes in a memory efficient way.
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item].data

    def __array__(self, dtype=None, *args, **kwargs):
        def _reader(s):
            # adding one more dimension to data so they get stacked sample-wise
            return s.data[None, ...]

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
            hdf5[s] = sample.__dict__[s]


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
            sample.__dict__[k] = hdf5[k].value

        return sample
