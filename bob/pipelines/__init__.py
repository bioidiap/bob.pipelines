from . import distributed  # noqa: F401
from . import transformers  # noqa: F401
from . import utils  # noqa: F401
from . import xarray as xr  # noqa: F401
from .sample import DelayedSample
from .sample import DelayedSampleSet
from .sample import DelayedSampleSetCached
from .sample import Sample
from .sample import SampleBatch
from .sample import SampleSet
from .sample import hdf5_to_sample  # noqa: F401
from .sample import sample_to_hdf5  # noqa: F401
from .wrappers import BaseWrapper
from .wrappers import CheckpointWrapper
from .wrappers import DaskWrapper
from .wrappers import DelayedSamplesCall
from .wrappers import SampleWrapper
from .wrappers import ToDaskBag
from .wrappers import dask_tags  # noqa: F401
from .wrappers import wrap  # noqa: F401


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
    DelayedSampleSetCached,
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
