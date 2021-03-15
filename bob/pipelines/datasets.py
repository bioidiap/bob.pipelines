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
import pathlib

from collections.abc import Iterable

from bob.db.base.utils import check_parameter_for_validity
from bob.db.base.utils import check_parameters_for_validity
from bob.extension.download import list_dir
from bob.extension.download import search_file

from .sample import Sample


def _maybe_open_file(path, **kwargs):
    if isinstance(path, (str, bytes, pathlib.Path)):
        path = open(path, **kwargs)
    return path


class FileListToSamples(Iterable):
    """Converts a list of files to a list of samples."""

    def __init__(self, list_file, transformer=None, **kwargs):
        super().__init__(**kwargs)
        self.list_file = list_file
        self.transformer = transformer

    def __iter__(self):
        for row_dict in self.rows:
            sample = Sample(None, **row_dict)
            if self.transformer is not None:
                # the transofmer might convert one sample to several samples
                for s in self.transformer.transform([sample]):
                    yield s
            else:
                yield sample


class CSVToSamples(FileListToSamples):
    """Converts a csv file to a list of samples"""

    def __init__(
        self,
        list_file,
        transformer=None,
        fieldnames=None,
        dict_reader_kwargs=None,
        **kwargs,
    ):
        list_file = _maybe_open_file(list_file, newline="")
        super().__init__(list_file=list_file, transformer=transformer, **kwargs)
        self.fieldnames = fieldnames
        self.dict_reader_kwargs = dict_reader_kwargs

    @property
    def rows(self):
        self.list_file.seek(0)
        kw = self.dict_reader_kwargs or {}
        reader = csv.DictReader(self.list_file, fieldnames=self.fieldnames, **kw)
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
        dataset_protocols_path,
        protocol,
        reader_cls=CSVToSamples,
        transformer=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        dataset_protocols_path : str
            Path to a folder or a tarball where the csv protocol files are located.
        protocol : str
            The name of the protocol to be used for samples. If None, the first
            protocol will be used.
        reader_cls : object
            A callable that will initialize the CSVToSamples reader, by default CSVToSamples
        transformer : object
            A scikit-learn transformer that further changes the samples

        Raises
        ------
        ValueError
            If the dataset_protocols_path does not exist.
        """
        super().__init__(**kwargs)
        if not os.path.exists(dataset_protocols_path):
            raise ValueError(f"The path `{dataset_protocols_path}` was not found")
        self.dataset_protocols_path = dataset_protocols_path
        self.reader_cls = reader_cls
        self._transformer = transformer
        self.readers = dict()
        self._protocol = None
        self.protocol = protocol

    @property
    def protocol(self):
        return self._protocol

    @protocol.setter
    def protocol(self, value):
        value = check_parameter_for_validity(
            value, "protocol", self.protocols(), self.protocols()[0]
        )
        self._protocol = value

    @property
    def transformer(self):
        return self._transformer

    @transformer.setter
    def transformer(self, value):
        self._transformer = value
        for reader in self.readers.values():
            reader.transformer = value

    def groups(self):
        names = list_dir(self.dataset_protocols_path, self.protocol, folders=False)
        names = [os.path.splitext(n)[0] for n in names]
        return names

    def protocols(self):
        return list_dir(self.dataset_protocols_path, files=False)

    def list_file(self, group):
        list_file = search_file(
            self.dataset_protocols_path,
            os.path.join(self.protocol, group + ".csv"),
        )
        return list_file

    def get_reader(self, group):
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
    def sort(samples, unique=True):
        """Sorts samples and removes duplicates by default."""

        def key_func(x):
            return x.key

        samples = sorted(samples, key=key_func)

        if unique:
            samples = [
                next(iter(v)) for _, v in itertools.groupby(samples, key=key_func)
            ]

        return samples
