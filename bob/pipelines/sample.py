"""Base definition of sample"""

from collections.abc import MutableSequence


def _copy_attributes(s, d):
    """Copies attributes from a dictionary to self
    """
    s.__dict__.update(
        dict(
            (k, v)
            for k, v in d.items()
            if k not in ("data", "load", "samples", "_data")
        )
    )


class _ReprMixin:
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
            + ")"
        )


class Sample(_ReprMixin):
    """Representation of sample that is sufficient for the blocks in this module

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
    """Representation of sample that can be loaded via a callable

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
        """Loads the data from the disk file"""
        if self._data is None:
            self._data = self.load()
        return self._data


class SampleSet(MutableSequence, _ReprMixin):
    """A set of samples with extra attributes
    https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes
    """

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
