#!/usr/bin/env python
# coding=utf-8


"""
Base mechanism that converts CSV lines to Samples
"""

import csv
import functools
import os

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

import bob.db.base

from bob.pipelines import DelayedSample


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
            "stateless": True,
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
            raise ValueError("The field `path` is not available in your dataset.")

    def convert_row_to_sample(self, row, header):
        path = row[0]
        reference_id = row[1]

        kwargs = dict([[str(h).lower(), r] for h, r in zip(header[2:], row[2:])])

        return DelayedSample(
            functools.partial(
                self.data_loader,
                os.path.join(self.dataset_original_directory, path + self.extension),
            ),
            key=path,
            reference_id=reference_id,
            **kwargs
        )


class AnnotationsLoader(TransformerMixin, BaseEstimator):
    """
    Metadata loader that loads annotations in the Idiap format using the function
    :any:`bob.db.base.read_annotation_file`

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
                DelayedSample(
                    x._load,
                    parent=x,
                    delayed_attributes=dict(
                        annotations=lambda: bob.db.base.read_annotation_file(
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
            "stateless": True,
            "requires_fit": False,
        }
