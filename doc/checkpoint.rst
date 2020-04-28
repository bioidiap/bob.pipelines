.. _bob.pipelines.checkpoint:

=============
Checkpointing
=============

Very often during the processing of :any:`sklearn.pipeline.Pipeline` with big chunks of
data, it is useful to have checkpoints of some steps of the pipeline into the disk. This
is useful for several purposes, such as:

   - Reusing samples that are expensive to be re-computed.
   - Inspection of algorithms.


Scikit-learn has a caching mechanism that allows the caching of
:any:`sklearn.pipeline.Pipeline` that can be used for such purpose. Although useful,
such structure is not user friendly.

As we detailed in :ref:`bob.pipelines.sample`, sklearn estimators can be extended to
handle samples with metadata. Now, one metadata can be a unique identifier of each
sample. We will refer to this unique identifier as ``sample.key``. If we have that in
our samples, we can use that identifier to save and load samples from disk. This is what
we call checkpointing and to do this, all you need to do is to wrap your estimator with
:any:`CheckpointWrapper` and make sure your samples have the ``.key`` metadata.

Checkpointing samples
---------------------

Below, you will see an example on how checkpointing works. First, let's make a
transformer.

.. doctest::

   >>> # by convention, we import bob.pipelines as mario, because mario works with pipes ;)
   >>> import bob.pipelines as mario
   >>> import numpy as np
   >>> from sklearn.base import TransformerMixin, BaseEstimator
   >>>
   >>> class MyTransformer(TransformerMixin, BaseEstimator):
   ...     def transform(self, X, sample_specific_offsets):
   ...         print(f"Transforming {len(X)} samples ...")
   ...         return np.array(X) + np.array(sample_specific_offsets)
   ...
   ...     def fit(self, X):
   ...         print("Fit was called!")
   ...         return self

All checkpointing transformers must be able to handle :any:`Sample`'s.
For that, we can use :any:`SampleWrapper`:

.. doctest::

   >>> transform_extra_arguments=[("sample_specific_offsets", "offset")]
   >>> sample_transformer = mario.SampleWrapper(MyTransformer(), transform_extra_arguments)

Then, we wrap it with :any:`CheckpointWrapper`:

.. doctest::

   >>> # create some samples with ``key`` metadata
   >>> # Creating X: 3 samples, 2 features
   >>> X = np.zeros((3, 2))
   >>> # 3 offsets: one for each sample
   >>> offsets = np.arange(3).reshape((3, 1))
   >>> # key values must be string because they will be used to create file names.
   >>> samples = [mario.Sample(x, offset=o, key=str(i)) for i, (x, o) in enumerate(zip(X, offsets))]
   >>> samples[0]
   Sample(data=array([0., 0.]), offset=array([0]), key='0')

   >>> import tempfile
   >>> import os
   >>> # create a temporary directory to save checkpoints
   >>> with tempfile.TemporaryDirectory() as dir_name:
   ...    checkpointing_transformer = mario.CheckpointWrapper(
   ...        sample_transformer, features_dir=dir_name)
   ...
   ...    # transform samples
   ...    transformed_samples = checkpointing_transformer.transform(samples)
   ...
   ...    # Let's check the features directory
   ...    list(sorted(os.listdir(dir_name)))
   Transforming 3 samples ...
   ['0.h5', '1.h5', '2.h5']

.. note::

   By default, :any:`CheckpointWrapper` saves samples inside HDF5 files
   but you can change that. Refer to its documentation to see how.

If checkpoints for a sample already exists, it will not be recomputed but loaded from
disk:

.. doctest::

   >>> # create a temporary directory to save checkpoints
   >>> with tempfile.TemporaryDirectory() as dir_name:
   ...    checkpointing_transformer = mario.CheckpointWrapper(
   ...        sample_transformer, features_dir=dir_name)
   ...
   ...    # transform samples for the first time, it should print transforming 3 samples
   ...    transformed_samples1 = checkpointing_transformer.transform(samples)
   ...
   ...    # transform samples again. This time it should not print transforming 3
   ...    # samples
   ...    transformed_samples2 = checkpointing_transformer.transform(samples)
   ...
   ...    # It should print True
   ...    print(np.allclose(transformed_samples1[1].data, transformed_samples2[1].data))
   Transforming 3 samples ...
   True

.. note::

   :any:`SampleSet`'s can be checkpointed as well. The samples inside them
   should have the ``.key`` metadata.


Checkpointing estimators
------------------------

We can also checkpoint estimators after their training (``estimator.fit``). This allows
us to load the estimator from disk instead of training it if ``.fit`` is called and a
checkpoint exists.

.. doctest::

   >>> # create a temporary directory to save checkpoints
   >>> with tempfile.NamedTemporaryFile(prefix="model", suffix=".pkl") as f:
   ...    f.close()
   ...    checkpointing_transformer = mario.CheckpointWrapper(
   ...        sample_transformer, model_path=f.name)
   ...
   ...    # call .fit for the first time, it should print Fit was called!
   ...    __ = checkpointing_transformer.fit(samples)
   ...
   ...    # call .fit again. This time it should not print anything
   ...    __ = checkpointing_transformer.fit(samples)
   Fit was called!
