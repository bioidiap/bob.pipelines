"""Base definition of sample"""


def samplesets_to_samples(samplesets):
    """
    Given a list of :py:class:`SampleSet` break them in to a list of :py:class:`Sample` with its 
    corresponding key

    This is supposed to fit the :py:meth:`sklearn.estimator.BaseEstimator.fit` where X and y are the inputs
    Check here https://scikit-learn.org/stable/developers/develop.html for more info

    Parameters
    ----------
      samplesets: list
         List of :py:class:`SampleSet


    Return 
    ------
       X and y used in :py:meth:`sklearn.estimator.BaseEstimator.fit`

    """

    # TODO: Is there a way to make this operation more efficient? numpy.arrays?
    X = []
    y= []

    for s in samplesets:
        X += s.samples
        y += [s.key]

    return X, y


def transform_sample_sets(transformer, sample_sets):
    return [
        SampleSet(transformer.transform(sset.samples), parent=sset)
        for sset in sample_sets
    ]



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


class SampleSet:
    """A set of samples with extra attributes"""

    def __init__(self, samples, parent=None, **kwargs):
        self.samples = samples
        if parent is not None:
            _copy_attributes(self, parent.__dict__)
        _copy_attributes(self, kwargs)
