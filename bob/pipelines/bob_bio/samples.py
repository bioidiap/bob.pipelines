#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import os
import collections

import numpy

import bob.io.base


class _BaseSample:
    """Very basic sample (internal use only)"""

    def __init__(self, data, path):

        self.data = data
        self.path = path


## TODO: This design only takes into consideration the very basic use-case.  It
## is possible to extend it so each sample carries more information that can be
## used in more sophisticated implementations of each algorithm block.  For
## example, it would be nice to have support for "annotated" samples.


class Sample(_BaseSample):
    """Representation of sample that is sufficient for our pipelines

    Each sample must respond to the following methods and attributes:

        * method ``load()``: Loads the data for this sample, which should be an
          iterable of :py:class:`numpy.ndarray`.
        * attribute ``data``: Contains the data for this sample.  This field
          may be set to ``None`` upon initialization.  It is used internally to
          store and transmit pre-loaded and transformed data between different
          processing stages.  It is also an iterable which may contain elements
          of different nature than those returned by ``load()``, but respect
          the same ordering.  E.g., the first entry of data corresponds to a
          transformed version of the first array returned by ``load()``
        * attribute ``path``: This is a unique path that leads either to a
          cached version of this sample, or its original raw data
        * attribute ``cache``: This is a unique path to be used for eventually
          storing a cached version of this sample.

    The optional ``**kwargs`` argument allows you to attach more attributes to
    this sample instance.


    Parameters
    ----------

        data : object
            Object representing the data to initialize this sample with.
            Typically, it is ``None``

        path : str
            A path to the raw data pertaining to this sample

        directory : str
            Base directory where to look for the raw data

        extension : str
            Extension of the file in the ``directory`` to be loaded

    """

    def __init__(self, data, path, directory, extension, **kwargs):
        super(Sample, self).__init__(data=data, path=path)
        self.directory = directory
        self.extension = extension
        for k, v in kwargs.items():
            setattr(self, k, v)

    def load(self):
        """Loads the sample from disk, if not already loaded

        Returns
        -------

            data : object
                An object representing the data for this sample.  It can be any
                combination of data that represents the sample.

        """

        return self.data or [
            bob.io.base.load(
                os.path.join(self.directory, (self.path + self.extension))
            )
        ]


class Reference(_BaseSample):
    """Representation of biometric reference that is sufficient for our
    pipelines

    This is the same as :py:class:`Sample`, except it accomodates loading of
    multiple different subsamples within and also stores the subject identifier
    for the references.

    The optional ``**kwargs`` argument allows you to attach more attributes to
    this reference instanceRawData.

    """

    def __init__(self, data, path, samples, subject, id, **kwargs):
        super(Reference, self).__init__(data=data, path=path)
        self.samples = samples
        self.subject = subject
        self.id = id
        for k, v in kwargs.items():
            setattr(self, k, v)

    def load(self):
        if self.data: return self.data
        l = [k.load() for k in self.samples]
        return [item for sublist in l for item in sublist]  ##flatten


class Probe(Reference):
    """Representation of probe that is sufficient for our pipelines

    This is (functionally) the same as :py:class:`Reference`, except it
    contains a list of biometric reference identifiers it need to be
    cross-checked against, during scoring.

    """

    def __init__(self, data, path, samples, subject, id, references, **kwargs):
        super(Probe, self).__init__(
            data=data, path=path, samples=samples, subject=subject, id=id
        )
        self.references = references
        for k, v in kwargs.items():
            setattr(self, k, v)

    def load(self):
        return super(Probe, self).load()


Score = collections.namedtuple("Score", ["probe", "reference", "data"])
