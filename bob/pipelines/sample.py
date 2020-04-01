"""Base definition of sample"""


def _copy_attributes(s, d):
    """Copies attributes from a dictionary to self
    """
    s.__dict__.update(
        dict([k, v] for k, v in d.items() if k not in ("data", "load", "samples"))
    )


class DelayedSample:
    """Representation of sample that can be loaded via a callable

    The optional ``**kwargs`` argument allows you to attach more attributes to
    this sample instance.


    Parameters
    ----------

        load : function
            A python function that can be called parameterlessly, to load the
            sample in question from whatever medium

        parent : :py:class:`DelayedSample`, :py:class:`Sample`, None
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

    @property
    def data(self):
        """Loads the data from the disk file"""
        return self.load()


class Sample:
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


from collections.abc import MutableSet


class SampleSet(MutableSet):
    """A set of samples with extra attributes
    https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes
    """

    def __init__(self, samples, parent=None, **kwargs):
        self.samples = set(samples)
        if parent is not None:
            _copy_attributes(self, parent.__dict__)
        _copy_attributes(self, kwargs)

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return self.samples.__iter__()

    def __contains__(self, item):
        return str(item) in [str(sample.key) for sample in self]

    def add(self, item):
        if not isinstance(item, Sample):
            raise ValueError(f"item should be of type Sample, not {item}")

        if not item in self.samples:
            self.samples.add(item)

    def discard(self, item):

        if isinstance(item, Sample):
            self.samples.discard(item)

        selected_sample = None
        for sample in self:
            if str(item) == str(sample.key):
                selected_sample = sample
                break

        self.samples.discard(selected_sample)
