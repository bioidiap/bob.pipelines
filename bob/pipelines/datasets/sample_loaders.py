#!/usr/bin/env python
# coding=utf-8


"""
Base mechanism that converts CSV lines to Samples
"""

from bob.extension.download import find_element_in_tarball
from bob.pipelines import DelayedSample, Sample, SampleSet
import os
from abc import ABCMeta, abstractmethod


class CSVBaseSampleLoader(metaclass=ABCMeta):
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

        metadata_loader:
            A python function that transforms parts of a `row` in a more complex object (e.g. convert eyes annotations embedded in the CSV file to a python dict)

        dataset_original_directory: str
            Path of where data is stored
        
        extension: str
            Default file extension

    """

    def __init__(
        self,
        data_loader,
        metadata_loader=None,
        dataset_original_directory="",
        extension="",
    ):
        self.data_loader = data_loader
        self.extension = extension
        self.dataset_original_directory = dataset_original_directory
        self.metadata_loader = metadata_loader

    @abstractmethod
    def __call__(self, filename):
        pass

    @abstractmethod
    def convert_row_to_sample(self, row, header):
        pass

    def convert_samples_to_samplesets(
        self, samples, group_by_reference_id=True, references=None
    ):
        if group_by_reference_id:

            # Grouping sample sets
            sample_sets = dict()
            for s in samples:
                if s.reference_id not in sample_sets:
                    sample_sets[s.reference_id] = (
                        SampleSet([s], parent=s)
                        if references is None
                        else SampleSet([s], parent=s, references=references)
                    )
                else:
                    sample_sets[s.reference_id].append(s)
            return list(sample_sets.values())

        else:
            return (
                [SampleSet([s], parent=s) for s in samples]
                if references is None
                else [SampleSet([s], parent=s, references=references) for s in samples]
            )

