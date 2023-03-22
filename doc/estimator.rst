.. _bob.pipelines.estimator:

Scikit-learn Estimators
=======================

This is a short example showing the difference between an estimator taking one parameter
and one taking metadata for each sample in addition.

Example of a custom Estimator
-----------------------------

Let us make an Estimator that takes batches of arrays as input and applies a simple
function to each of them:

.. doctest:: custom_estimator

    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator
    >>> class OffsetTransformer(BaseEstimator):
    ...     """Demo Estimator to add an offset to arrays."""
    ...     def __init__(self, offset=1):
    ...         self.offset = offset
    ...
    ...     def transform(self, arrays: np.ndarray) -> np.ndarray:
    ...         """Add the offset to each array"""
    ...         return [arr + self.offset for arr in arrays]

    >>> transformer = OffsetTransformer(offset=2)
    >>> transformer.transform([np.array([1, 2]), np.array([2, 3])])
    [array([3, 4]), array([4, 5])]

.. note::

    The ``transform`` method accepts a series of data samples (it works on batches). If
    you work with 2D numpy arrays, your ``transform`` needs to handle 3D arrays (or you
    need to loop over the first dimension to handle sample individually).

Now let's edit the ``OffsetTransformer`` so that each sample can be offset by a
different value. This may be the case when applying preprocessing on some data with
annotations. Each will be different for each sample.
Here, we want to pass an offset for each sample given to ``transform``:

.. doctest:: custom_estimator

    >>> class OffsetTransformerPerSample(BaseEstimator):
    ...     """Demo Estimator to add a different offset to each sample array."""
    ...     def transform(self, arrays: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    ...         """Add its offset to each array"""
    ...         return [arr + o for arr, o in zip(arrays, offsets)]

    >>> transformer_2 = OffsetTransformerPerSample()

We need to pass two series of arrays to ``transform``, one for the samples data and one
containing the offsets:

.. doctest:: custom_estimator

    >>> transformer_2.transform(arrays=[np.array([3,4,5]), np.array([1,2,3])], offsets=[1,2])
    [array([4, 5, 6]), array([3, 4, 5])]


We will see how the second estimator (with multiple parameters in ``transform``) can
cause problems with the pipeline API in the next page.
