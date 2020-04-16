.. _sample:

==============================================================
Samples, a way to enhance scikit pipelines with metadata
==============================================================

Some tasks in pattern recognition demands the usage of metadata to support some processing (e.g. face cropping, audio segmentation).
To support scikit-learn based estimators with such requirement task, this package provides two mechanisms that:

    1. Wraps input data in a layer called `:py:class:Sample` that allows you to append some metadata to  your original input data.
    
    2. A mixin class (:py:class:`bob.pipelines.mixins.SampleMixin`) that interplay between :py:class:`bob.pipelines.sample.Sample` and your estimator.

What is a Sample ?
------------------

A :py:class:`bob.pipelines.sample.Sample` is simple container that wraps a datapoint.
The example below shows how this can be used to wrap a :py:class:`numpy.array`.

.. code:: python

   >>> import numpy
   >>> from bob.pipelines.sample import Sample
   >>> X = numpy.array([1,3])
   >>> sample = Sample(X)
   

Sample and metadata
-------------------

Metadata can be added as keyword arguments in :py:meth:`bob.pipelines.sample.Sample.__init__.py`, like:

.. code:: python

   >>> import numpy
   >>> from bob.pipelines.sample import Sample
   >>> X = numpy.array([1,3])
   >>> sample = Sample(X, gender="Male")
   >>> print(sample.gender)
   Male



Transforming Samples
--------------------

The example below shows a simple snippet on how to build a scikit learn transformer and use it in a Pipeline.

.. literalinclude:: ./python/pipeline_example.py
   :linenos:

As can be observed, `MyTransformer` supports one keyword argument called `metadata` that can't be used by :py:class:`scikit.pipeline.Pipeline`.


This can be approached with the mixing class :py:class:`bob.pipelines.mixins.SampleMixin`.
By extending `MyTransform` with this mixin class allows you to use instance of `Samples` in the methods `estimator.fit` and `estimator.transform` without having to touch the original implementation of `MyTranformer`.

The example below shows the same example, but now wrapping datapoints in to :py:class:`bob.pipelines.sample.Sample`

.. literalinclude:: ./python/pipeline_example_boosted.py
   :linenos:
   :emphasize-lines: 22

The magic happens in line 22, where `MyTransformer` is "mixed" with the function :py:class:`bob.pipelines.mixins.SampleMixin` to create a new class called `MyBoostedTransformer` that is able to:  i-) handle samples, ii-) handle the original operations of `MyTransformer`, and iii-) pass metadata through the pipeline.
This can be carried out at runtime by the function :py:func:`bob.pipelines.mixins.mix_me_up`.
Another possibility would be to carry this out at development time by explicitly create this new class, like in the example below.

.. code:: python

    >>> from bob.pipelines.mixins import SampleMixin
    >>> class MyBoostedTransformer(SampleMixin, MyTransformer):
    >>>     pass


Delayed Sample
--------------

Sometimes keeping several samples into memory and transfer them over the network can be very memory and band demanding.
For these cases, there is :py:class:`bob.pipelines.sample.DelayedSample`.

A `DelayedSample` acts like a `Sample`, but its `data` attribute is implemented as a function that can load the respective data from its permanent storage representation.
To create a `DelayedSample`, you pass a `load()` function that must be called **parameterlessly** to load the required data (see implementation of the `data` attribute of this class).

Below follow an example of how to use :py:class:`bob.pipelines.sample.DelayedSample`.

.. code:: python

    >>> from bob.pipelines.sample import DelayedSample
    >>> import pickle
    >>> import functools
    >>> f = open("my_sample_in_disk", "rb")
    >>> delayed_sample = DelayedSample(functools.partial(pickle.load, X), metadata=1)

This can be used transparently with the :py:class:`bob.pipelines.mixins.SampleMixin` as can be observed in the example below.


.. literalinclude:: ./python/pipeline_example_boosted_delayed.py
   :linenos:
   :emphasize-lines: 30

Observe in line 30, a :py:class:`bob.pipelines.sample.DelayedSample` is used instead of :py:class:`bob.pipelines.sample.Sample`.


Sample Set
----------

A :py:class:`bob.pipelines.sample.SampleSet`, as the name sugests, represents a set of samples.
Such set of samples can represent the samples that belongs to a class.

Below follow an snippet on how to use :py:class:`bob.pipelines.sample.SampleSet`.

.. code:: python

   >>> import numpy
   >>> from bob.pipelines.sample import Sample, SampleSet
   >>> X = numpy.array([1,3])
   >>> sample = SampletSet([Sample(X)], class_name="A")


As can be observed, :py:class:`bob.pipelines.sample.SampleSet` allows you to set any type of metadata (`class_name` in the example).


This can be used transparently with the :py:class:`bob.pipelines.mixins.SampleMixin` as can be observed in the example below.


.. literalinclude:: ./python/pipeline_example_boosted_sample_set.py
   :linenos:
   :emphasize-lines: 32