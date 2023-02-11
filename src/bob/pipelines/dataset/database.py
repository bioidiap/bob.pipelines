"""
The principles of this module are:

* one csv file -> one set
* one row -> one sample
* csv files could exist in a tarball or inside a folder
* scikit-learn transformers are used to further transform samples
* several csv files (sets) compose a protocol
* several protocols compose a database
"""
import csv
import itertools
import os

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, TextIO, Union

import sklearn.pipeline

from bob.pipelines.dataset.protocols.retrieve import (
    list_group_names,
    list_protocol_names,
    open_definition_file,
    retrieve_protocols,
)

from ..sample import Sample
from ..utils import check_parameter_for_validity, check_parameters_for_validity


def _maybe_open_file(path, **kwargs):
    if isinstance(path, (str, bytes, Path)):
        path = open(path, **kwargs)
    return path


class FileListToSamples(Iterable):
    """Converts a list of paths and metadata to a list of samples.

    This class reads a file containing paths and optionally metadata and returns a list
    of :py:class:`bob.pipelines.Sample`\\ s when called.

    A separator character can be set (defaults is space) to split the rows.
    No escaping is done (no quotes).

    A Transformer can be given to apply a transform on each sample. (Keep in mind this
    will not be distributed on Dask; Prefer applying Transformer in a
    ``bob.pipelines.Pipeline``.)
    """

    def __init__(
        self,
        list_file: str,
        separator: str = " ",
        transformer: Optional[sklearn.pipeline.Pipeline] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.list_file = list_file
        self.transformer = transformer
        self.separator = separator

    def __iter__(self):
        for row_dict in self.rows:
            sample = Sample(None, **row_dict)
            if self.transformer is not None:
                # The transformer might convert one sample to several samples
                for s in self.transformer.transform([sample]):
                    yield s
            else:
                yield sample

    @property
    def rows(self) -> dict[str, Any]:
        with open(self.list_file, "rt") as f:
            for line in f:
                yield dict(line.split(self.separator))


class CSVToSamples(FileListToSamples):
    """Converts a csv file to a list of samples"""

    def __init__(
        self,
        list_file: str,
        transformer: Optional[sklearn.pipeline.Pipeline] = None,
        dict_reader_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        list_file = _maybe_open_file(list_file, newline="")
        super().__init__(
            list_file=list_file,
            transformer=transformer,
            **kwargs,
        )
        self.dict_reader_kwargs = dict_reader_kwargs

    @property
    def rows(self):
        self.list_file.seek(0)
        kw = self.dict_reader_kwargs or {}
        reader = csv.DictReader(self.list_file, **kw)
        return reader


class FileListDatabase:
    """A generic database interface.
    Use this class to convert csv files to a database that outputs samples. The
    format is simple, the files must be inside a folder (or a compressed
    tarball) with the following format::

        dataset_protocols_path/<protocol>/<group>.csv

    The top folders are the name of the protocols (if you only have one, you may
    name it ``default``). Inside each protocol folder, there are `<group>.csv`
    files where the name of the file specifies the name of the group. We
    recommend using the names ``train``, ``dev``, ``eval`` for your typical
    training, development, and test sets.

    """

    def __init__(
        self,
        *,
        name: str,
        protocol: str,
        dataset_protocols_path: Union[os.PathLike[str], str, None] = None,
        reader_cls: Iterable = CSVToSamples,
        transformer: Optional[sklearn.pipeline.Pipeline] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        dataset_protocols_path
            Path to a folder or a tarball where the csv protocol files are located.
        protocol
            The name of the protocol to be used for samples. If None, the first
            protocol found will be used.
        reader_cls
            An iterable that returns created Sample objects from a list file.
        transformer
            A scikit-learn transformer that further changes the samples.

        Raises
        ------
        ValueError
            If the dataset_protocols_path does not exist.
        """

        # Tricksy trick to make protocols non-classmethod when instantiated
        self.protocols = self._instance_protocols

        if getattr(self, "name", None) is None:
            self.name = name

        if dataset_protocols_path is None:
            dataset_protocols_path = self.retrieve_dataset_protocols()

        self.dataset_protocols_path = Path(dataset_protocols_path)

        if len(self.protocols()) < 1:
            raise ValueError(
                f"No protocols found at `{dataset_protocols_path}`!"
            )
        self.reader_cls = reader_cls
        self._transformer = transformer
        self.readers: dict[str, Iterable] = {}
        self._protocol = None
        self.protocol = protocol
        super().__init__(**kwargs)

    @property
    def protocol(self) -> str:
        return self._protocol

    @protocol.setter
    def protocol(self, value: str):
        value = check_parameter_for_validity(
            value, "protocol", self.protocols(), self.protocols()[0]
        )
        self._protocol = value

    @property
    def transformer(self) -> sklearn.pipeline.Pipeline:
        return self._transformer

    @transformer.setter
    def transformer(self, value: sklearn.pipeline.Pipeline):
        self._transformer = value
        for reader in self.readers.values():
            reader.transformer = value

    def groups(self) -> list[str]:
        """Returns all the available groups."""
        return list_group_names(
            database_name=self.name,
            protocol=self.protocol,
            database_filename=self.dataset_protocols_path.name,
            base_dir=self.dataset_protocols_path.parent,
            subdir=".",
        )

    def _instance_protocols(self) -> list[str]:
        """Returns all the available protocols."""
        return list_protocol_names(
            database_name=self.name,
            database_filename=self.dataset_protocols_path.name,
            base_dir=self.dataset_protocols_path.parent,
            subdir=".",
        )

    @classmethod
    def protocols(cls) -> list[str]:  # pylint: disable=method-hidden
        """Returns all the available protocols."""
        # Ensure the definition file exists locally
        loc = cls.retrieve_dataset_protocols()
        if not hasattr(cls, "name"):
            raise ValueError(f"{cls} has no attribute 'name'.")
        return list_protocol_names(
            database_name=getattr(cls, "name"),
            database_filename=loc.name,
            base_dir=loc.parent,
            subdir=".",
        )

    @classmethod
    def retrieve_dataset_protocols(cls) -> Path:
        """Return a path to the protocols definition files.

        If the files are not present locally in ``bob_data/<subdir>/<category>``, they
        will be downloaded.

        The class inheriting from CSVDatabase must have a ``name`` and an
        ``dataset_protocols_urls`` attributes.

        A ``checksum`` attribute can be used to verify the file and ensure the correct
        version is used.
        """

        # When the path is specified, just return it.
        if getattr(cls, "dataset_protocols_path", None) is not None:
            return getattr(cls, "dataset_protocols_path")

        # Save to bob_data/protocols, or if present, in a category sub directory.
        subdir = Path("protocols")
        if hasattr(cls, "category"):
            subdir = subdir / getattr(cls, "category")

        # Retrieve the file from the server (or use the local version).
        return retrieve_protocols(
            urls=getattr(cls, "dataset_protocols_urls"),
            destination_filename=getattr(cls, "dataset_protocols_name", None),
            base_dir=None,
            subdir=subdir,
            checksum=getattr(cls, "dataset_protocols_checksum", None),
        )

    def list_file(self, group: str) -> TextIO:
        """Returns the corresponding definition file of a group."""
        list_file = open_definition_file(
            search_pattern=group + ".csv",
            database_name=self.name,
            protocol=self.protocol,
            database_filename=self.dataset_protocols_path.name,
            base_dir=self.dataset_protocols_path.parent,
            subdir=".",
        )
        return list_file

    def get_reader(self, group: str) -> Iterable:
        """Returns an :any:`Iterable` of :any:`Sample` objects."""
        key = (self.protocol, group)
        if key not in self.readers:
            self.readers[key] = self.reader_cls(
                list_file=self.list_file(group), transformer=self.transformer
            )

        reader = self.readers[key]
        return reader

    def samples(self, groups=None):
        """Get samples of a certain group

        Parameters
        ----------
        groups : :obj:`str`, optional
            A str or list of str to be used for filtering samples, by default None

        Returns
        -------
        list
            A list containing the samples loaded from csv files.
        """

        groups = check_parameters_for_validity(
            groups, "groups", self.groups(), self.groups()
        )
        all_samples = []
        for grp in groups:
            for sample in self.get_reader(grp):
                all_samples.append(sample)

        return all_samples

    @staticmethod
    def sort(samples: list[Sample], unique: bool = True):
        """Sorts samples and removes duplicates by default."""

        def key_func(x):
            return x.key

        samples = sorted(samples, key=key_func)

        if unique:
            samples = [
                next(iter(v))
                for _, v in itertools.groupby(samples, key=key_func)
            ]

        return samples
