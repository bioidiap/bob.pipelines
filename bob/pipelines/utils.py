import pickle

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
    assert list(obj.keys()) == list(new_obj.keys())
    for k, v in obj.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_equal(v, new_obj[k])
        else:
            assert v == new_obj[k]


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


def isinstance_nested(instance, attribute, isinstance_of):
    """
    Check if an object and its nested objects is an instance of a class.

    This is useful while using aggregation and it's necessary to check if some
    functionally was aggregated

    Parameters
    ----------
        instance:
           Object to be searched

        attribute:
           Attribute name to be recursively searched

        isinstance_of:
            Instance class to be searched

    """

    if not hasattr(instance, attribute):
        return False

    # Checking the current object and its immediate nested
    if isinstance(instance, isinstance_of) or isinstance(
        getattr(instance, attribute), isinstance_of
    ):
        return True
    else:
        # Recursive search
        return isinstance_nested(getattr(instance, attribute), attribute, isinstance_of)


def hash_string(key, bucket_size=1000):
    """
    Generates a hash code given a string.
    The have is given by the `sum(ord([string])) mod bucket_size`

    Parameters
    ----------

    key: str
      Input string to be hashed

    bucket_size: int
      Size of the hash table.

    """
    return str(sum([ord(i) for i in (key)]) % bucket_size)
