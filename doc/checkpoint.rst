.. _bob.pipelines.checkpoint:

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

   >>> import bob.pipelines
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
   >>> sample_transformer = bob.pipelines.SampleWrapper(MyTransformer(), transform_extra_arguments)

Then, we wrap it with :any:`CheckpointWrapper`:

.. doctest::

   >>> # create some samples with ``key`` metadata
   >>> # Creating X: 3 samples, 2 features
   >>> X = np.zeros((3, 2))
   >>> # 3 offsets: one for each sample
   >>> offsets = np.arange(3).reshape((3, 1))
   >>> # key values must be string because they will be used to create file names.
   >>> samples = [bob.pipelines.Sample(x, offset=o, key=str(i)) for i, (x, o) in enumerate(zip(X, offsets))]
   >>> samples[0]
   Sample(data=array([0., 0.]), offset=array([0]), key='0')

   >>> import tempfile
   >>> import os
   >>> # create a temporary directory to save checkpoints
   >>> with tempfile.TemporaryDirectory() as dir_name:
   ...    checkpointing_transformer = bob.pipelines.CheckpointWrapper(
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
   ...    checkpointing_transformer = bob.pipelines.CheckpointWrapper(
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
   ...    checkpointing_transformer = bob.pipelines.CheckpointWrapper(
   ...        sample_transformer, model_path=f.name)
   ...
   ...    # call .fit for the first time, it should print Fit was called!
   ...    __ = checkpointing_transformer.fit(samples)
   ...
   ...    # call .fit again. This time it should not print anything
   ...    __ = checkpointing_transformer.fit(samples)
   Fit was called!


.. _bob.pipelines.wrap:

Convenience wrapper function
----------------------------

We provide a :any:`wrap` function to wrap estimators in several layers easily. So far we
learned that we need to wrap our estimators with :any:`SampleWrapper` and
:any:`CheckpointWrapper`. There is also a Dask wrapper: :any:`DaskWrapper` which you'll
learn about in :ref:`bob.pipelines.dask`. Below, is an example on how to use it.
Instead of:

.. doctest::

   >>> transformer = MyTransformer()

   >>> transform_extra_arguments=[("sample_specific_offsets", "offset")]
   >>> transformer = bob.pipelines.SampleWrapper(transformer, transform_extra_arguments)

   >>> transformer = bob.pipelines.CheckpointWrapper(
   ...     transformer, features_dir="features", model_path="model.pkl")

   >>> transformer = bob.pipelines.DaskWrapper(transformer)

You can write:

.. doctest::

   >>> transformer = bob.pipelines.wrap(
   ...     [MyTransformer, "sample", "checkpoint", "dask"],
   ...     transform_extra_arguments=transform_extra_arguments,
   ...     features_dir="features",
   ...     model_path="model.pkl",
   ... )
   >>> # or if your estimator is already created.
   >>> transformer = bob.pipelines.wrap(
   ...     ["sample", "checkpoint", "dask"],
   ...     MyTransformer(),
   ...     transform_extra_arguments=transform_extra_arguments,
   ...     features_dir="features",
   ...     model_path="model.pkl",
   ... )

Much simpler, no? Internally ``"sample"`` string will be replaced by
:any:`SampleWrapper`. You provide a list of classes to wrap as the first argument,
optionally provide an estimator to be wrapped as the second argument. If the second
argument is missing, the first class will be used to create the estimator. Then, you
provide the ``__init__`` parameters of all classes as kwargs.
Internally, :any:`wrap` will pass kwargs to classes that accept it.

.. note::

   :any:`wrap` is a convenience function but it might be limited in what it can do. You
   can always use the wrapper classes directly.

:any:`wrap` recognizes :any:`sklearn.pipeline.Pipeline` objects and when pipelines are
passed, it wraps the steps inside them instead. For example, instead of:

.. doctest::

   >>> transformer1 = bob.pipelines.wrap(
   ...     [MyTransformer, "sample"],
   ...     transform_extra_arguments=transform_extra_arguments,
   ... )
   >>> transformer2 = bob.pipelines.wrap(
   ...     [MyTransformer, "sample"],
   ...     transform_extra_arguments=transform_extra_arguments,
   ... )
   >>> from sklearn.pipeline import make_pipeline
   >>> pipeline = make_pipeline(transformer1, transformer2)

you can write:

.. doctest::

   >>> pipeline = make_pipeline(MyTransformer(), MyTransformer())
   >>> pipeline = bob.pipelines.wrap(["sample"], pipeline, transform_extra_arguments=transform_extra_arguments)

It will pass ``transform_extra_arguments`` to all steps when wrapping them with the
:any:`SampleWrapper`. You cannot pass specific arguments to one of the steps. Wrapping
pipelines with :any:`wrap`, while limited, becomes useful when we are wrapping them
with Dask as we will see in :ref:`bob.pipelines.dask`.
