"""Base definition of sample."""

from collections.abc import MutableSequence, Sequence
from .utils import vstack_features
import numpy as np
from distributed.protocol.serialize import (
    serialize,
    deserialize,
    dask_serialize,
    dask_deserialize,
    register_generic,
)
import cloudpickle

import logging

logger = logging.getLogger(__name__)


def _copy_attributes(s, d):
    """Copies attributes from a dictionary to self."""
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

    #def __getstate__(self):        
    #    d = dict(self.__dict__)
    #    d.pop("_data", None)
    #    return d

    #def __setstate__(self, d):
    #    self._data = d.pop("_data", None)
    #    self.__dict__.update(d)


class SampleSet(MutableSequence, _ReprMixin):
    """A set of samples with extra attributes"""

    def __init__(self, samples, parent=None, **kwargs):
        self.samples = samples
        if parent is not None:
            _copy_attributes(self, parent.__dict__)
        _copy_attributes(self, kwargs)

    def _load(self):
        if isinstance(self.samples, DelayedSample):
            self.samples = self.samples.data

    def __len__(self):
        self._load()
        return len(self.samples)

    def __getitem__(self, item):
        self._load()
        return self.samples.__getitem__(item)

    def __setitem__(self, key, item):
        self._load()
        return self.samples.__setitem__(key, item)

    def __delitem__(self, item):
        self._load()
        return self.samples.__delitem__(item)

    def insert(self, index, item):
        self._load()
        # if not item in self.samples:
        self.samples.insert(index, item)


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


def get_serialized_sample_header(sample):
    
    sample_header = dict(
        (k, v)
        for k, v in sample.__dict__.items()
        if k not in ("data", "load", "samples", "_data")
    )

    return cloudpickle.dumps(sample_header)

def deserialize_sample_header(sample):
    return cloudpickle.loads(sample)

@dask_serialize.register(SampleSet)
def serialize_sampleset(sampleset):
    
    def serialize_delayed_sample(delayed_sample):        
        header_sample = get_serialized_sample_header(delayed_sample)
        frame_sample = cloudpickle.dumps(delayed_sample)
        return header_sample, frame_sample

    header = dict()

    # Ship the header of the sampleset
    # in the header of the message
    key = sampleset.key
    header["sampleset_header"] = get_serialized_sample_header(sampleset)

    header["sample_header"] = []
    frames = []
    
    # Checking first if our sampleset.samples are shipped as DelayedSample
    if isinstance(sampleset.samples, DelayedSample):
        header_sample, frame_sample = serialize_delayed_sample(sampleset.samples)
        frames += [frame_sample]        
        header["sample_header"].append(header_sample)
        header["sample_type"] = "DelayedSampleList"
    else:        
        for sample in sampleset.samples:
            if isinstance(sample, DelayedSample):
                header_sample, frame_sample = serialize_delayed_sample(sample)
                frame_sample = [frame_sample]
            else:                
                header_sample, frame_sample = serialize(sample)
            frames += frame_sample
            header["sample_header"].append(header_sample)

        header["sample_type"] = "DelayedSample" if isinstance(sample, DelayedSample) else "Sample"

    return header, frames


@dask_deserialize.register(SampleSet)
def deserialize_sampleset(header, frames):

    if not "sample_header" in header:
        raise ValueError("Problem with SampleSet serialization. `_sample_header` not found")

    sampleset_header = deserialize_sample_header(header["sampleset_header"])
    sampleset = SampleSet([], **sampleset_header)
    

    if header["sample_type"]=="DelayedSampleList":
        sampleset.samples = cloudpickle.loads(frames[0])
        return sampleset
  
    for h, f in zip(header["sample_header"], frames):        
        if header["sample_type"] == "Sample":            
            data = dask_deserialize.dispatch(Sample)(h, [f])
            sampleset.samples.append(data)
        else:
            sampleset.samples.append( cloudpickle.loads(f) )

    return sampleset


@dask_serialize.register(Sample)
def serialize_sample(sample):

    header_sample = get_serialized_sample_header(sample)

    # If data is numpy array, uses the dask serializer
    header, frames = serialize(sample.data)
    header["sample"] = header_sample

    return header, frames


@dask_deserialize.register(Sample)
def deserialize_sample(header, frames):    

    try:
        data = dask_deserialize.dispatch(np.ndarray)(header, frames)
    except KeyError:
        data = cloudpickle.loads(frames)

    sample_header = deserialize_sample_header(header["sample"])
    sample = Sample(data, parent=None, **sample_header)
    return sample
