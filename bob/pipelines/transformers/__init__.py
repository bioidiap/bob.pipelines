from .file_loader import FileLoader
from .function import CheckpointSampleFunctionTransformer
from .function import SampleFunctionTransformer
from .function import StatelessPipeline
from .linearize import CheckpointSampleLinearize
from .linearize import Linearize
from .linearize import SampleLinearize
from .pca import CheckpointSamplePCA
from .pca import SamplePCA
from .str_to_types import Str_To_Types  # noqa: F401
from .str_to_types import str_to_bool  # noqa: F401


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
    Linearize,
    SampleLinearize,
    CheckpointSampleLinearize,
    CheckpointSamplePCA,
    SamplePCA,
    SampleFunctionTransformer,
    CheckpointSampleFunctionTransformer,
    StatelessPipeline,
    FileLoader,
)

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith("_")]
