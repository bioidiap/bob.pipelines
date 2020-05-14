#!/usr/bin/env python
# coding=utf-8

import os
import csv
import pathlib

import logging

logger = logging.getLogger(__name__)


class CSVDataset:
    """
    Generic multi-subset filelist dataset that yields samples

    To create a new dataset, you only need to provide a CSV formatted filelist
    using any separator (e.g. comma, space, semi-colon) with the following
    information:

    .. code-block:: text

       value1,value2,value3
       value4,value5,value6
       ...

    Notice that all rows must have the same number of entries.

    Parameters
    ----------

    subsets : list, dict
        Paths to one or more CSV formatted files containing the various subsets
        to be recognized by this dataset, or a dictionary, mapping subset names
        to paths (or opened file objects) of CSV files.  Internally, we save a
        dictionary where keys default to the basename of paths (list input).

    fieldnames : list, tuple
        An iterable over the field names (strings) to assign to each column in
        the CSV file.  It should have as many items as fields in each row of
        the CSV file(s).

    loader : object
        A function that receives as input, a context dictionary (with, at
        least, a "subset" key indicating which subset is being served), and a
        dictionary with ``{key: path}`` entries, and returns a dictionary with
        the loaded data.

    """

    def __init__(self, subsets, fieldnames, loader):

        if isinstance(subsets, dict):
            self._subsets = subsets
        else:
            self._subsets = dict(
                (os.path.splitext(os.path.basename(k))[0], k) for k in subsets
            )
        self.fieldnames = fieldnames
        self._loader = loader

    def check(self, limit=0):
        """For each subset, check if all data can be correctly accessed

        This function assumes each sample has a ``data`` and a ``key``
        attribute.  The ``key`` attribute should be a string, or representable
        as such.


        Parameters
        ----------

        limit : int
            Maximum number of samples to check (in each protocol/subset
            combination) in this dataset.  If set to zero, then check
            everything.


        Returns
        -------

        errors : int
            Number of errors found

        """

        logger.info(f"Checking dataset...")
        errors = 0
        for name in self._subsets.keys():
            logger.info(f"Checking subset '{name}'...")
            samples = self.samples(name)
            if limit:
                logger.info(f"Checking at most first '{limit}' samples...")
                samples = samples[:limit]
            for pos, sample in enumerate(samples):
                try:
                    sample.data  # may trigger data loading
                    logger.info(f"{sample.key}: OK")
                except Exception as e:
                    logger.error(
                        f"Found error loading entry {pos} in subset {name} "
                        f"from file '{self._subsets[name]}': {e}"
                        )
                    errors += 1
        return errors

    def subsets(self):
        """Returns all available subsets at once

        Returns
        -------

        subsets : dict
            A dictionary mapping subset names to lists of objects (respecting
            the ``key``, ``data`` interface).

        """

        return dict((k, self.samples(k)) for k in self._subsets.keys())

    def samples(self, subset):
        """Returns all samples in a subset

        This method will load CSV information for a given subset and return
        all samples of the given subset after passing each entry through the
        loading function.


        Parameters
        ----------

        subset : str
            Name of the subset data to load


        Returns
        -------

        subset : list
            A lists of objects (respecting the ``key``, ``data`` interface).

        """

        fileobj = self._subsets[subset]
        if isinstance(fileobj, (str, bytes, pathlib.Path)):
            with open(self._subsets[subset], newline="") as f:
                cf = csv.reader(f)
                samples = [k for k in cf]
        else:
            cf = csv.reader(fileobj)
            samples = [k for k in cf]
            fileobj.seek(0)

        return [
            self._loader(
                dict(subset=subset, order=n), dict(zip(self.fieldnames, k))
            )
            for n, k in enumerate(samples)
        ]
