.. _bob.pipelines.sample:

Samples, a way to enhance scikit pipelines with metadata
=========================================================

Some tasks in pattern recognition demands the usage of metadata to support some processing (e.g. face cropping, audio segmentation).
To support scikit-learn based estimators with such requirement task, this package provides two mechanisms that:

    1. Wraps input data in a layer called :any:`Sample` that allows you to append some metadata to your original input data.

    2. A wrapper class (:any:`SampleWrapper`) that interplay between :any:`Sample` and your estimator.

What is a Sample ?
------------------

A :any:`Sample` is a simple container that wraps a data-point.
The example below shows how this can be used to wrap a :any:`numpy.array`.

.. doctest::

   >>> import bob.pipelines
   >>> import numpy as np
   >>> data = np.array([1, 3])
   >>> sample = bob.pipelines.Sample(data)
   >>> sample
   Sample(data=array([1, 3]))
   >>> sample.data is data
   True


Sample and metadata
-------------------

Metadata can be added as keyword arguments in :any:`Sample`, like:

.. doctest::

   >>> sample = bob.pipelines.Sample(data, gender="Male")
   >>> sample
   Sample(data=array([1, 3]), gender='Male')
   >>> sample.gender
   'Male'


Transforming Samples
--------------------

Imagine that we have the following transformer that requires some metadata to actually
work:

.. doctest::

   >>> from sklearn.base import TransformerMixin, BaseEstimator
   >>>
   >>> class MyTransformer(TransformerMixin, BaseEstimator):
   ...     def transform(self, X, sample_specific_offsets):
   ...         return np.array(X) + np.array(sample_specific_offsets)
   ...
   ...     def fit(self, X):
   ...         pass
   ...
   ...     def _more_tags(self):
   ...         return {"requires_fit": False}
   >>>
   >>>
   >>> # Creating X: 3 samples, 2 features
   >>> X = np.zeros((3, 2))
   >>> # 3 offsets: one for each sample
   >>> offsets = np.arange(3).reshape((3, 1))
   >>> transformer = MyTransformer()
   >>>
   >>> transformer.transform(X, offsets)
   array([[0., 0.],
          [1., 1.],
          [2., 2.]])

While this transformer works well by itself, it can't be used by
:any:`sklearn.pipeline.Pipeline`:

.. doctest::

   >>> from sklearn.pipeline import make_pipeline
   >>> pipeline = make_pipeline(transformer)
   >>> pipeline.transform(X, offsets)
   Traceback (most recent call last):
      ...
   TypeError: _transform() takes 2 positional arguments but 3 were given

To approach this issue, :any:`SampleWrapper` can be used. This class wraps
other estimators and accepts as input samples and passes the data with metadata inside
samples to the wrapped estimator:

.. doctest::

   >>> # construct a list of samples from the data we had before
   >>> samples = [bob.pipelines.Sample(x, offset=o) for x, o in zip(X, offsets)]
   >>> samples[1]
   Sample(data=array([0., 0.]), offset=array([1]))

Now we need to tell :any:`SampleWrapper` to pass the ``offset`` inside
samples as an extra argument to our transformer as ``sample_specific_offsets``. This is
accommodated by the ``transform_extra_arguments`` parameter. It accepts a list of tuples
that maps sample metadata to arguments of the transformer:

.. doctest::

   >>> transform_extra_arguments=[("sample_specific_offsets", "offset")]
   >>> sample_transformer = bob.pipelines.SampleWrapper(transformer, transform_extra_arguments)
   >>> transformed_samples = sample_transformer.transform(samples)
   >>> # transformed values will be stored in sample.data
   >>> np.array([s.data for s in transformed_samples])
   array([[0., 0.],
          [1., 1.],
          [2., 2.]])

Note that wrapped estimators accept samples as input and return samples. Also, they keep
the sample's metadata around in transformed samples.

.. doctest::

   >>> transformed_samples[1].data
   array([1., 1.])
   >>> transformed_samples[1].offset  # the `offset` metadata is available here too.
   array([1])

Now that our transformer is wrapped, we can also use it inside a pipeline:

.. doctest::

   >>> sample_pipeline = make_pipeline(sample_transformer)
   >>> np.array([s.data for s in sample_pipeline.transform(samples)])
   array([[0., 0.],
          [1., 1.],
          [2., 2.]])


Delayed Sample
--------------

Sometimes keeping several samples into memory and transferring them over the network can
be very memory and bandwidth demanding. For these cases, there is
:any:`DelayedSample`.

A :any:`DelayedSample` acts like a :any:`Sample`, but its `data` attribute is implemented as a
function that can load the respective data from its permanent storage representation. To
create a :any:`DelayedSample`, you pass a ``load()`` function that when called without any
parameter, it must load and return the required data.

Below, follow an example on how to use :any:`DelayedSample`.

.. doctest::

   >>> def load():
   ...     # load data (usually from disk) and return
   ...     print("Loading data from disk!")
   ...     return np.zeros((2,))
   >>> delayed_sample = bob.pipelines.DelayedSample(load, metadata=1)
   >>> delayed_sample
   DelayedSample(metadata=1)

As soon as you access the ``.data`` attribute, the data is loaded and returned:

.. doctest::

   >>> delayed_sample.data
   Loading data from disk!
   array([0., 0.])

:any:`DelayedSample` can be used instead of :any:`Sample`
transparently:

.. doctest::

   >>> from functools import partial
   >>> def load_ith_data(i):
   ...     return np.zeros((2,)) + i
   >>>
   >>> delayed_samples = [bob.pipelines.DelayedSample(partial(load_ith_data, i), offset=[i]) for i in range(3)]
   >>> np.array([s.data for s in sample_pipeline.transform(delayed_samples)])
   array([[0., 0.],
          [2., 2.],
          [4., 4.]])

.. note::

   Actually, :any:`SampleWrapper` always returns
   :any:`DelayedSample`'s. This becomes useful when the data returned
   is not used. We will see that happening in :ref:`bob.pipelines.checkpoint`.

Sample Set
----------

A :any:`SampleSet`, as the name suggests, represents a set of samples.
Such set of samples can represent the samples that belongs to a class.

Below, follow an snippet on how to use :any:`SampleSet`.

.. doctest::

   >>> sample_sets = [
   ...     bob.pipelines.SampleSet(samples, class_name="A"),
   ...     bob.pipelines.SampleSet(delayed_samples, class_name="B"),
   ... ]
   >>> sample_sets[0]
   SampleSet(samples=[Sample(data=array([0., 0.]), offset=array([0])), Sample(data=array([0., 0.]), offset=array([1])), Sample(data=array([0., 0.]), offset=array([2]))], class_name='A')


:any:`SampleWrapper` works transparently with :any:`SampleSet`'s as well. It will
transform each sample inside and returns the same SampleSets with new data.

.. doctest::

   >>> transformed_sample_sets = sample_pipeline.transform(sample_sets)
   >>> transformed_sample_sets[0].samples[1]
   DelayedSample(offset=array([1]))
   >>> transformed_sample_sets[0].samples[1].data
   array([1., 1.])
