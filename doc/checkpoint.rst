.. _checkpoint:

=======================
Checkpointing Samples
=======================

Mechanism that allows checkpointing of :py:class:`bob.pipelines.sample.Sample` during the processing of :py:class:`sklearn.pipeline.Pipeline` using `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ files.

Very often during the processing of :py:class:`sklearn.pipeline.Pipeline` with big chunks of data is useful to have checkpoints of some steps of the pipeline into the disk.
This is useful for several purposes:
  - Reuse samples that are expensive to be re-computed
  - Inspection of algorithms  


Scikit learn has a caching mechanism that allows the caching of `estimators <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline>`_ that can be used for such purpose.
Althought useful, such structure is not user friendly.

As in :ref:`sample`, this can be approached with the :py:class:`bob.pipelines.mixins.CheckpointMixin` mixin, where a new class can be created either dynamically with the :py:func:`bob.pipelines.mixings.mix_me_up` function:

.. code:: python

    >>> bob.pipelines.mixins import CheckpointMixin
    >>> MyTransformerCheckpoint = mix_me_up((CheckpointMixin,), MyTransformer)

or explicitly:

.. code:: python

    >>> bob.pipelines.mixins import CheckpointMixin
    >>> class MyTransformerCheckpoint(CheckpointMixin, MyTransformer):
    >>>     pass


Checkpointing a transformer
---------------------------

The code below is a repetition of the example from :ref:`sample`, but now `MyTransformer` is checkpointable once `MyTransformer.transform` is executed.

.. literalinclude:: ./python/pipeline_example_boosted_checkpoint.py
   :linenos:
   :emphasize-lines: 23, 28, 34, 38


.. warning::

    In line 28, samples are created with the keyword argument, `key`. The :py:class:`bob.pipelines.mixins.CheckpointMixin` uses this information for saving.


The keyword argument `features_dir` defined in lines 34 and 38 sets the absolute path where those samples will be saved


Checkpointing an statfull transformers
--------------------------------------

Statefull transformers, are transformers that implement the methods `fit` and `transform`.
Those can be checkpointed too as can be observed in the example below.

.. literalinclude:: ./python/pipeline_example_boosted_checkpoint_estimator.py
   :linenos:
   :emphasize-lines: 52-55


The keyword argument `features_dir` and `model_payh` defined in lines 52 to 55 sets the absolute path where samples and the model trained after fit will be saved



