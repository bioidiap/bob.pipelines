.. _bob.pipelines.pipeline:

Making pipelines
================

We rely on the :ref:`scikit-learn pipeline API<scikit-learn:pipeline>` to run our
pipelines.
Basically, a pipeline is a series of :py:class:`sklearn.base.BaseEstimator` objects.
When data is fed to the pipeline, the first Estimator receives it, processes it and what
is returned is then fed to the second Estimator in the pipeline, and so on until the
end of the steps. The result of the last Estimator is the result of the pipeline.

To make a pipeline, you can call :py:func:`sklearn.pipeline.make_pipeline` or directly
instantiate a Pipeline object with :py:class:`sklearn.pipeline.Pipeline`.

Below is a quick example on how to make a pipeline out of two Estimators:

.. doctest:: pipeline

    >>> import numpy
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.decomposition import PCA

    >>> pipeline = make_pipeline(StandardScaler(), PCA())
    >>> pipeline = pipeline.fit([[0, 0], [0, 0], [2, 2]])
    >>> pipeline.transform([[0, 1], [2, 1]])
    array([[-0.25,  0.75],
           [ 1.25, -0.75]])

The issue with scikit-learn pipelines
-------------------------------------

The previous example works fine as long as we only have one parameter to give to the
``transform`` or ``fit`` method of all the Estimators in the pipeline.

If we want to pass some metadata to the Estimators in a pipeline (like the ``offsets``
parameters from the previous page), we cannot include the Estimator in a pipeline.

This example takes the transformer with an ``offsets`` parameters defined in the
previous page and tries to make a pipeline out of it:

.. testsetup:: custom_estimator

    from sklearn.base import BaseEstimator
    import numpy as np
    class OffsetTransformerPerSample(BaseEstimator):
        """Demo Estimator to add a different offset to each sample array."""
        def transform(self, arrays: np.ndarray, offsets: np.ndarray) -> np.ndarray:
            """Add its offset to each array"""
            return [arr + o for arr, o in zip(arrays, offsets)]

    transformer_2 = OffsetTransformerPerSample()

.. doctest:: custom_estimator

   >>> from sklearn.pipeline import make_pipeline
   >>> pipeline = make_pipeline(transformer_2)
   >>> pipeline.transform([np.array([3,4,5]), np.array([1,2,3])], [1,2])
   Traceback (most recent call last):
      ...
   TypeError: _transform() takes 2 positional arguments but 3 were given

In order to include such an Estimator in a pipeline, we must add some logic to redirect
data and metadata through the pipeline. This is the goal of the :any:`Sample` and
:any:`SampleWrapper` presented in the next page.
