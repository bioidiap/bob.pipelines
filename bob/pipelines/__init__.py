from . import utils
from .sample import Sample, DelayedSample, SampleSet, sample_to_hdf5, hdf5_to_sample
from .wrappers import (
    BaseWrapper,
    DelayedSamplesCall,
    SampleWrapper,
    CheckpointWrapper,
    DaskWrapper,
    ToDaskBag,
    wrap,
    dask_tags,    
)
from . import distributed
from . import transformers


def __appropriate__(*args):
    """Says object was actually declared here, and not in the import module.
    Fixing sphinx warnings of not being able to find classes, when path is
    shortened.

    Parameters
    ----------
    *args
        The objects that you want sphinx to beleive that are defined here.

    Resolves `Sphinx referencing issues <https//github.com/sphinx-
    doc/sphinx/issues/3048>`
    """

    for obj in args:
        obj.__module__ = __name__


__appropriate__(
    Sample,
    DelayedSample,
    SampleSet,
    BaseWrapper,
    DelayedSamplesCall,
    SampleWrapper,
    CheckpointWrapper,
    DaskWrapper,
    ToDaskBag,
)

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith("_")]
