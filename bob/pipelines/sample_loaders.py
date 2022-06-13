#!/usr/bin/env python
# coding=utf-8


"""
Base mechanism that converts CSV lines to Samples
"""

import collections
import csv
import functools
import json
import logging
import os

from sklearn.base import BaseEstimator, TransformerMixin

from bob.extension.download import search_file
from bob.pipelines import DelayedSample

logger = logging.getLogger(__name__)


class CSVToSampleLoader(TransformerMixin, BaseEstimator):
    """
    Base class that converts the lines of a CSV file, like the one below to
    :any:`bob.pipelines.DelayedSample` or :any:`bob.pipelines.SampleSet`

    .. code-block:: text

       PATH,REFERENCE_ID
       path_1,reference_id_1
       path_2,reference_id_2
       path_i,reference_id_j
       ...

    .. note::
       This class should be extended, because the meaning of the lines depends on
       the final application where thoses CSV files are used.

    Parameters
    ----------

        data_loader:
            A python function that can be called parameterlessly, to load the
            sample in question from whatever medium

        dataset_original_directory: str
            Path of where data is stored

        extension: str
            Default file extension

    """

    def __init__(
        self,
        data_loader,
        dataset_original_directory="",
        extension="",
    ):
        self.data_loader = data_loader
        self.extension = extension
        self.dataset_original_directory = dataset_original_directory

    def fit(self, X, y=None):
        return self

    def _more_tags(self):
        return {
            "requires_fit": False,
        }

    def transform(self, X):
        """
        Transform one CVS line to ONE :any:`bob.pipelines.DelayedSample`

        Parameters
        ----------
        X:
            CSV File Object (open file)

        """
        X.seek(0)
        reader = csv.reader(X)
        header = next(reader)

        self.check_header(header)
        return [self.convert_row_to_sample(row, header) for row in reader]

    def check_header(self, header):
        """
        A header should have at least "reference_id" AND "PATH"
        """
        header = [h.lower() for h in header]
        if "reference_id" not in header:
            raise ValueError(
                "The field `reference_id` is not available in your dataset."
            )

        if "path" not in header:
            raise ValueError(
                "The field `path` is not available in your dataset."
            )

    def convert_row_to_sample(self, row, header):
        path = row[0]
        reference_id = row[1]

        kwargs = dict(
            [[str(h).lower(), r] for h, r in zip(header[2:], row[2:])]
        )

        return DelayedSample(
            functools.partial(
                self.data_loader,
                os.path.join(
                    self.dataset_original_directory, path + self.extension
                ),
            ),
            key=path,
            reference_id=reference_id,
            **kwargs,
        )


class AnnotationsLoader(TransformerMixin, BaseEstimator):
    """
    Metadata loader that loads annotations in the Idiap format using the function
    :any:`read_annotation_file`

    Parameters
    ----------

    annotation_directory: str
        Path where the annotations are store

    annotation_extension: str
        Extension of the annotations

    annotation_type: str
        Annotations type

    """

    def __init__(
        self,
        annotation_directory=None,
        annotation_extension=".json",
        annotation_type="json",
    ):
        self.annotation_directory = annotation_directory
        self.annotation_extension = annotation_extension
        self.annotation_type = annotation_type

    def transform(self, X):
        if self.annotation_directory is None:
            return None

        annotated_samples = []
        for x in X:

            # since the file id is equal to the file name, we can simply use it
            annotation_file = os.path.join(
                self.annotation_directory, x.key + self.annotation_extension
            )

            annotated_samples.append(
                DelayedSample.from_sample(
                    x,
                    delayed_attributes=dict(
                        annotations=lambda: read_annotation_file(
                            annotation_file, self.annotation_type
                        )
                    ),
                )
            )

        return annotated_samples

    def fit(self, X, y=None):
        return self

    def _more_tags(self):
        return {
            "requires_fit": False,
        }


def read_annotation_file(file_name, annotation_type):
    """This function provides default functionality to read annotation files.

    Parameters
    ----------
    file_name : str
            The full path of the annotation file to read. The path can also be like
            ``base_path:relative_path`` where the base_path can be both a directory or
            a tarball. This allows you to read annotations from inside a tarball.
    annotation_type : str
            The type of the annotation file that should be read. The following
            annotation_types are supported:

                * ``json``: The file contains annotations of any format, dumped in a
                    text json file.

    Returns
    -------
    dict
            A python dictionary with the keypoint name as key and the
            position ``(y,x)`` as value, and maybe some additional annotations.

    Raises
    ------
    IOError
            If the annotation file is not found.
    ValueError
            If the annotation type is not known.
    """
    if not file_name:
        return None

    if annotation_type != "json":
        raise ValueError(
            f"The annotation type {annotation_type} is not supported."
        )

    if ":" in file_name:
        base_path, tail = file_name.split(":", maxsplit=1)
        f = search_file(base_path, [tail])
    else:
        if not os.path.exists(file_name):
            raise IOError("The annotation file '%s' was not found" % file_name)
        f = open(file_name)

    annotations = {}

    try:
        annotations = json.load(f, object_pairs_hook=collections.OrderedDict)
    finally:
        f.close()

    if (
        annotations is not None
        and "leye" in annotations
        and "reye" in annotations
        and annotations["leye"][1] < annotations["reye"][1]
    ):
        logger.warn(
            "The eye annotations in file '%s' might be exchanged!" % file_name
        )

    return annotations
