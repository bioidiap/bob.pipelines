from . import distributed  # noqa
from . import transformers  # noqa
from . import utils  # noqa
from . import xarray as xr  # noqa
from .sample import DelayedSample
from .sample import DelayedSampleSet
from .sample import Sample
from .sample import SampleSet
from .sample import SampleBatch
from .sample import hdf5_to_sample  # noqa
from .sample import sample_to_hdf5  # noqa
from .wrappers import BaseWrapper
from .wrappers import CheckpointWrapper
from .wrappers import DaskWrapper
from .wrappers import DelayedSamplesCall
from .wrappers import SampleWrapper
from .wrappers import ToDaskBag
from .wrappers import dask_tags  # noqa
from .wrappers import wrap  # noqa


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
    DelayedSampleSet,
    SampleBatch,
    BaseWrapper,
    DelayedSamplesCall,
    SampleWrapper,
    CheckpointWrapper,
    DaskWrapper,
    ToDaskBag,
)

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith("_")]
