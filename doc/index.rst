.. -*- coding: utf-8 -*-

.. _bob.pipelines:

===============
 Bob Pipelines
===============

This is a temporary documentation on how this pipeline mechanism is going to work.
This work is heavily based on `dask <https://dask.org/>`_. Hence, have a look in its documentation so see how this works.
For now, we have standardized how:

 - Samples are defined
 - Algorithms are defined
 - Pipelines are defined
 - The relation with the grid (that's for free using dask)
 - The way to interface with the legacy databases and algorithms 


So far only biometric pipelines are working and this will be the base for everything else.



Design
======

The design of the new system enforces a clear separation between the definition of the pipeline, the type data each processing step can consume and the definition of the processing blocks themselves.
We leverage Python's weakly typed system to avoid concepts as base classes, leaving to the programmers' discretion how to concretely implement input data and processing algorithms.
We explain first the pipeline, then we approach the algorithms and, finally, how a checkpointing mechanism was put in place for a basic biometric use-case taken from the current bob.bio.base implementation (eigenfaces).
We hope this design and strategy is sufficient to cover all other existing use cases in that package.



Pipeline
--------

Implementation: `pipelines <bob/pipelines/bob_bio/pipelines.py#L15>`_

The basic biometric pipeline we implemented contains 3 processing steps:

 1. Background modelling
 2. Biometric reference enrollement
 3. Biometric probe-reference matching to generate scores

Each phase is implemented as a function that takes arguments and returns the result of processing.
These functions can be re-used in other pipelines to create more complex structures.
For example, in datasets with separate development and test data, phases 2 and 3 above could be repeated once for tackling the test data separately.

By the end of the processing steps above, the user is returned a list of scores they can use directly or save to disk.


Background modelling
....................

In this phase, a set of samples is load and optionally pre-processed (with a variable number of steps of choice) before being fed to an algorithm that generates an output background model.

 - Input: an iterable over `Sample` objects. Each `Sample` (implementation: `Sample <bob/pipelines/bob_bio/blocks.py#L61>`_ object defines **at least** a `data` attribute that contains the actual data that is available with such a sample.  `data` can be of any type, for as long as the algorithm for background modelling can consume it.
 - Output: an object representing the background model, of any type, for as long as enrollment and probing (i.e., the next processing phases) can use it.


Biometric enrollement
.....................

In this phase, a set of biometric references is provided.
For generality, each reference is composed of an iterable over (potentially) many Samples representing all data available to enroll a new reference.

 - Input: Various `SampleSet` objects, each representing a new biometric reference to be created.  Each `SampleSet` (implementation: `SampleSet <bob/pipelines/bob_bio/blocks.py#L88>`_ may be decorated with as much metadata as necessary (via Python's object-attribute attachment), that is copied to generated models (implementation: `attribute copying <bob/pipelines/bob_bio/blocks.py#L15>`_.
 - Output: Various objects with at least 1 attribute named `data` that represent each, one enrolled biometric reference.  The actual data type may be any supported by the next processing phases (i.e., probing).  Metadata for each enrolled model is copied from the parent's `SampleSet` to the generated using the attribute copying procedure.


Biometric probing
.................

In this phase, a set of biometric probes is provided. Each probe should indicate to which models they should be matched against. Each probe is composed of at least one, but possible many `Sample` objects. Such information will be used by the matching algorithm to filter down (if required) model-probe matching pairs.

 - Input: Various `SampleSet` objects, annotated with any amount of metadata (attributes)
 - Output: A number of score objects, which is the result of probing each `Sample` in the `SampleSet`for each probe object to all required references.  Each returned `SampleSet` is decorated with metadata concerning the original probe.  Each `Sample` in this set is decorated with metadata concerning the original reference.  By inspection, it is possible to associate references to probe identities, provided they are included in the original `Sample` attributes.


Pre-processing and Feature Extraction
-------------------------------------

The mechanism of choice for implementing these steps is embedded into a class called `SampleLoader <bob/pipelines/bob_bio/blocks.py#L274>`_.
It chains a `Sample` through a series of processing steps, generating another sample with the same metadata.
While the `SampleLoader` was originally intended to wrap sequences of bob.bio.base `Preprocessor` and `Extractor` types, it was simplified to host **any** number of processing steps.

A `SampleLoader` is initialized with a sequence of 2-tuples indicating a common name for the step and a callable, that transforms the sample `data`.
The callable should be called without any other parameters than the `data` object that represents the sample data.
If by any chance an existing function or method does not work this way, you should use `functools.partial` to create a representation that conforms to this API.
See usage examples `here <bob/pipelines/script/run.py#L68>`_.

In the existing pipeline implementation, the `SampleLoader` is called for every existing `Sample` or `SampleSet` in an homogeneous way (`see <bob/pipelines/bob_bio/pipelines.py#L15>`_.

.. note::
  If different loading needs to be created for each phase of the pipeline processing, then a new, more specialized pipeline may be created from the existing functionality in place.



Legacy databases
----------------


The legacy code of Bob low and high-level database interfaces most of the time do not respect this constraint.
To sort this out, we provided a `DatabaseConnector <bob/pipelines/bob_bio/blocks.py#L98>`_, whose responsibility is to provide the required iterables for each of our pipeline's 3 processing stages above, wrapping the original Bob loading methods and `File` objects so their `load()` methods can be called parameterlessly.
In case of doubt, check the implementation of that class.


DelayedSample, raw data loading and checkpointing
-------------------------------------------------


While our basic pipeline only controls what gets processed and when, it has no clue about how to handle data potentially stored in disk (or in a remote database) that are input to it.
To solve this problem transparently, we introduced the concept of a `DelayedSample <bob/pipelines/bob_bio/blocks.py#L658>`_.
A `DelayedSample` acts like a `Sample`, but its `data` attribute is implemented as a function that can load the respective data from its permanent storage representation.
To create a `DelayedSample`, you pass a `load()` function that must be called **parameterlessly** to load the required data (see implementation of the `data` attribute of this class).

.. warning::

   The way we implemented check-pointing is not written in stone. Each pipeline may re-implement in a different way. The current implementation only shows possibilities for this feature. 
   
The `DelayedSample` mechanism can be extended to provide optional check-pointing for the sample processing through the pipeline.
Check-pointing is a mechanism that allows one to save intermediate processing output of the various pipeline phases and sub-phases to permanent storage.

Checking-pointing is implemented for most of the processing phases in the example pipeline, but can be further extended.
The logic is simple: if a certain step should be check-pointed, a path is passed to the function implementing the step, indicating the name of directory where objects it is treating should be stored.
In that case, the result of the sample processing is stored by the processor itself on the provided path using the sample's own identity information.
It is up to the implementor of the processing phase to decide which meta information on the sample to use for creating a file representation of the sample.
The `SampleLoader` for example, uses the `path` attribute on samples to do so.
After storing the processed sample on disk, the processor returns a `DelayedSample` indicating to the next processing phases it may eventually need to load the data from disk.  The loading mechanism of this `DelayedSample` is now provided by the processor itself, which is the **only** agent that knows how to serialize and deserialize the processed samples.

In an eventual second pass of the same processing step (e.g. when the user re-executes the whole pipeline), each processed sample path is checked and the sample is only re-processed in case it is not yet available.
While this technique may save processing time, it does not prevent the pipeline checking every individual sample has been processed.
If a stored version of the processed sample exists, then it is not reprocessed and again a `DelayedSample` representing it is forwarded to the next processing step.

Notice that if no check-pointing is enabled, the processing step pass simple `Sample` objects to the next processing step.
Because `Sample` and `DelayedSample` are functionality equivalent, each processing step may be (or not) check-pointed independently.
For example, it is possible to check-point feature extraction in the `SampleLoader`, but omit checkpointing for the pre-processor.

The way to configure check-pointing for the whole pipeline is done via a nested dictionary structure.  Details can be found `in <bob/pipelines/bob_bio/pipelines.py#L66>`_.

Current implementation
----------------------

You will find a way to put all these elements together to run `existing pipelines <bob/pipelines/script/run.py>`_.



Reference Manual
==================


.. include:: links.rst
