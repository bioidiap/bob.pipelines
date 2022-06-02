
============================
Python API for bob.pipelines
============================

Summary
=======

Sample's API
------------
.. autosummary::
    bob.pipelines.Sample
    bob.pipelines.DelayedSample
    bob.pipelines.SampleSet
    bob.pipelines.DelayedSampleSet
    bob.pipelines.DelayedSampleSetCached
    bob.pipelines.SampleBatch

Wrapper's API
-------------
.. autosummary::
    bob.pipelines.wrap
    bob.pipelines.BaseWrapper
    bob.pipelines.SampleWrapper
    bob.pipelines.CheckpointWrapper
    bob.pipelines.DaskWrapper
    bob.pipelines.ToDaskBag
    bob.pipelines.DelayedSamplesCall

Database's API
--------------
.. autosummary::
    bob.pipelines.FileListDatabase
    bob.pipelines.FileListToSamples
    bob.pipelines.CSVToSamples

Transformers' API
-----------------
.. autosummary::
    bob.pipelines.transformers.Str_To_Types
    bob.pipelines.transformers.str_to_bool

Xarray's API
------------
.. autosummary::
    bob.pipelines.xarray.samples_to_dataset
    bob.pipelines.xarray.DatasetPipeline
    bob.pipelines.xarray.Block

Utilities
---------
.. autosummary::
    bob.pipelines.assert_picklable
    bob.pipelines.check_parameter_for_validity
    bob.pipelines.check_parameters_for_validity
    bob.pipelines.dask_tags
    bob.pipelines.estimator_requires_fit
    bob.pipelines.flatten_samplesets
    bob.pipelines.get_bob_tags
    bob.pipelines.hash_string
    bob.pipelines.is_instance_nested
    bob.pipelines.is_picklable
    bob.pipelines.is_pipeline_wrapped

Main module
===========
.. automodule:: bob.pipelines

Heterogeneous SGE
=================
.. automodule:: bob.pipelines.distributed.sge

Transformers
============
.. automodule:: bob.pipelines.transformers

xarray Wrapper
==============
.. automodule:: bob.pipelines.xarray
