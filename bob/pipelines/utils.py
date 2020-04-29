import pickle
import nose
import numpy as np


def is_picklable(obj):
    """Test if an object is picklable or not."""
    try:
        pickle.dumps(obj)
    except TypeError:
        return False
    except pickle.PicklingError:
        return False

    return True


def assert_picklable(obj):
    """Test if an object is picklable or not."""

    string = pickle.dumps(obj)
    new_obj = pickle.loads(string)
    obj = obj.__dict__
    new_obj = new_obj.__dict__
    assert len(obj) == len(new_obj)
    nose.tools.assert_equal(list(obj.keys()), list(new_obj.keys()))
    for k, v in obj.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_equal(v, new_obj[k])
        else:
            nose.tools.assert_equal(v, new_obj[k])


def is_estimator_stateless(estimator):
    if not hasattr(estimator, "_get_tags"):
        raise ValueError(
            f"Passed estimator: {estimator} does not have the _get_tags method."
        )
    # See: https://scikit-learn.org/stable/developers/develop.html
    # if the estimator does not require fit or is stateless don't call fit
    tags = estimator._get_tags()
    if tags["stateless"] or not tags["requires_fit"]:
        return True
    return False


def _generate_features(reader, paths, same_size=False):
    """Load and stack features in a memory efficient way. This function is
    meant to be used inside :py:func:`vstack_features`.

    Parameters
    ----------
    reader : ``collections.Callable``
      See the documentation of :py:func:`vstack_features`.
    paths : ``collections.Iterable``
      See the documentation of :py:func:`vstack_features`.
    same_size : :obj:`bool`, optional
      See the documentation of :py:func:`vstack_features`.

    Yields
    ------
    object
      The first object returned is a tuple of :py:class:`numpy.dtype` of
      features and the shape of the first feature. The rest of objects are
      the actual values in features. The features are returned in C order.
    """

    shape_determined = False
    for i, path in enumerate(paths):

        feature = np.atleast_2d(reader(path))
        feature = np.ascontiguousarray(feature)
        if not shape_determined:
            shape_determined = True
            dtype = feature.dtype
            shape = list(feature.shape)
            yield (dtype, shape)
        else:
            # make sure all features have the same shape and dtype
            if same_size:
                assert shape == list(feature.shape)
            else:
                assert shape[1:] == list(feature.shape[1:])
            assert dtype == feature.dtype

        for value in feature.flat:
            yield value


def vstack_features(reader, paths, same_size=False):
    """Stacks all features in a memory efficient way.

    Parameters
    ----------
    reader : ``collections.Callable``
      The function to load the features. The function should only take one
      argument ``path`` and return loaded features. Use :any:`functools.partial`
      to accommodate your reader to this format.
      The features returned by ``reader`` are expected to have the same
      :py:class:`numpy.dtype` and the same shape except for their first
      dimension. First dimension should correspond to the number of samples.
    paths : ``collections.Iterable``
      An iterable of paths to iterate on. Whatever is inside path is given to
      ``reader`` so they do not need to be necessarily paths to actual files.
      If ``same_size`` is ``True``, ``len(paths)`` must be valid.
    same_size : :obj:`bool`, optional
      If ``True``, it assumes that arrays inside all the paths are the same
      shape. If you know the features are the same size in all paths, set this
      to ``True`` to improve the performance.

    Returns
    -------
    numpy.ndarray
      The read features with the shape ``(n_samples, *features_shape[1:])``.

    Examples
    --------
    This function in a simple way is equivalent to calling
    ``numpy.vstack(reader(p) for p in paths)``.

    >>> import numpy
    >>> from bob.io.base import vstack_features
    >>> def reader(path):
    ...     # in each file, there are 5 samples and features are 2 dimensional.
    ...     return numpy.arange(10).reshape(5,2)
    >>> paths = ['path1', 'path2']
    >>> all_features = vstack_features(reader, paths)
    >>> numpy.allclose(all_features, numpy.array(
    ...     [[0, 1],
    ...      [2, 3],
    ...      [4, 5],
    ...      [6, 7],
    ...      [8, 9],
    ...      [0, 1],
    ...      [2, 3],
    ...      [4, 5],
    ...      [6, 7],
    ...      [8, 9]]))
    True
    >>> all_features_with_more_memory = numpy.vstack(reader(p) for p in paths)
    >>> numpy.allclose(all_features, all_features_with_more_memory)
    True

    You can allocate the array at once to improve the performance if you know
    that all features in paths have the same shape and you know the total number
    of the paths:

    >>> all_features = vstack_features(reader, paths, same_size=True)
    >>> numpy.allclose(all_features, numpy.array(
    ...     [[0, 1],
    ...      [2, 3],
    ...      [4, 5],
    ...      [6, 7],
    ...      [8, 9],
    ...      [0, 1],
    ...      [2, 3],
    ...      [4, 5],
    ...      [6, 7],
    ...      [8, 9]]))
    True

    .. note::

      This function runs very slowly. Only use it when RAM is precious.
    """
    iterable = _generate_features(reader, paths, same_size)
    dtype, shape = next(iterable)
    if same_size:
        total_size = int(len(paths) * np.prod(shape))
        all_features = np.fromiter(iterable, dtype, total_size)
    else:
        all_features = np.fromiter(iterable, dtype)

    # the shape is assumed to be (n_samples, ...) it can be (5, 2) or (5, 3, 4).
    shape = list(shape)
    shape[0] = -1
    return np.reshape(all_features, shape, order="C")


def samples_to_np_array(samples, same_size=True):
    return vstack_features(lambda s: s.data, samples, same_size=same_size)
