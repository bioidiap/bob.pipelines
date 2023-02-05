"""Functionalities related to datasets processing."""

from .database import CSVToSamples, FileListDatabase, FileListToSamples
from .protocols import (  # noqa: F401
    download_protocol_definition,
    list_group_names,
    list_protocol_names,
    open_definition_file,
)


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
    FileListToSamples,
    CSVToSamples,
    FileListDatabase,
)
