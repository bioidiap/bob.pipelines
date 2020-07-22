#!/usr/bin/env python
# coding=utf-8

import json
import logging
import os
import pathlib

logger = logging.getLogger(__name__)


class JSONDataset:
    """
    Generic multi-protocol/subset filelist dataset that yields samples

    To create a new dataset, you need to provide one or more JSON formatted
    filelists (one per protocol) with the following contents:

    .. code-block:: json

       {
           "subset1": [
               [
                   "value1",
                   "value2",
                   "value3"
               ],
               [
                   "value4",
                   "value5",
                   "value6"
               ]
           ],
           "subset2": [
           ]
       }

    Your dataset many contain any number of subsets, but all sample entries
    must contain the same number of fields.


    Parameters
    ----------

    protocols : list, dict
        Paths to one or more JSON formatted files containing the various
        protocols to be recognized by this dataset, or a dictionary, mapping
        protocol names to paths (or opened file objects) of CSV files.
        Internally, we save a dictionary where keys default to the basename of
        paths (list input).

    fieldnames : list, tuple
        An iterable over the field names (strings) to assign to each entry in
        the JSON file.  It should have as many items as fields in each entry of
        the JSON file.

    loader : object
        A function that receives as input, a context dictionary (with at least
        a "protocol" and "subset" keys indicating which protocol and subset are
        being served), and a dictionary with ``{fieldname: value}`` entries,
        and returns an object with at least 2 attributes:

        * ``key``: which must be a unique string for every sample across
          subsets in a protocol, and
        * ``data``: which contains the data associated witht this sample

    """

    def __init__(self, protocols, fieldnames, loader):

        if isinstance(protocols, dict):
            self._protocols = protocols
        else:
            self._protocols = dict(
                (os.path.splitext(os.path.basename(k))[0], k) for k in protocols
            )
        self.fieldnames = fieldnames
        self._loader = loader

    def check(self, limit=0):
        """For each protocol, check if all data can be correctly accessed

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

        logger.info("Checking dataset...")
        errors = 0
        for proto in self._protocols:
            logger.info(f"Checking protocol '{proto}'...")
            for name, samples in self.subsets(proto).items():
                logger.info(f"Checking subset '{name}'...")
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
                            f"of protocol {proto} from file "
                            f"'{self._protocols[proto]}': {e}"
                        )
                        errors += 1
                    except Exception as e:
                        logger.error(f"{sample.key}: {e}")
                        errors += 1
        return errors

    def subsets(self, protocol):
        """Returns all subsets in a protocol

        This method will load JSON information for a given protocol and return
        all subsets of the given protocol after converting each entry through
        the loader function.

        Parameters
        ----------

        protocol : str
            Name of the protocol data to load


        Returns
        -------

        subsets : dict
            A dictionary mapping subset names to lists of objects (respecting
            the ``key``, ``data`` interface).

        """

        fileobj = self._protocols[protocol]
        if isinstance(fileobj, (str, bytes, pathlib.Path)):
            with open(self._protocols[protocol], "r") as f:
                data = json.load(f)
        else:
            data = json.load(f)
            fileobj.seek(0)

        retval = {}
        for subset, samples in data.items():
            retval[subset] = [
                self._loader(
                    dict(protocol=protocol, subset=subset, order=n),
                    dict(zip(self.fieldnames, k)),
                )
                for n, k in enumerate(samples)
            ]

        return retval
