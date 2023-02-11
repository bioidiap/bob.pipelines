# isort: skip_file
from . import distributed  # noqa: F401
from . import transformers  # noqa: F401
from . import utils  # noqa: F401
from . import xarray as xr  # noqa: F401
from .sample import (
    DelayedSample,
    DelayedSampleSet,
    DelayedSampleSetCached,
    Sample,
    SampleBatch,
    SampleSet,
)
from .utils import (  # noqa: F401
    assert_picklable,
    check_parameter_for_validity,
    check_parameters_for_validity,
    flatten_samplesets,
    hash_string,
    is_picklable,
)
from .wrappers import wrap  # noqa: F401
from .wrappers import (  # noqa: F401
    BaseWrapper,
    CheckpointWrapper,
    DaskWrapper,
    DelayedSamplesCall,
    SampleWrapper,
    ToDaskBag,
    dask_tags,
    estimator_requires_fit,
    get_bob_tags,
    getattr_nested,
    is_instance_nested,
    is_pipeline_wrapped,
)
from .dataset import FileListToSamples, CSVToSamples, FileListDatabase


def __appropriate__(*args):
    """Says object was actually declared here, and not in the import module.
    Fixing sphinx warnings of not being able to find classes, when path is
    shortened.

    Parameters
    ----------
    *args
        The objects that you want sphinx to believe that are defined here.

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
    FileListToSamples,
    CSVToSamples,
    FileListDatabase,
)

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith("_")]
