import pickle
import nose
import numpy as np

def is_picklable(obj):
    """
    Test if an object is picklable or not
    """
    try:
        pickle.dumps(obj)
    except TypeError:
        return False
    except pickle.PicklingError:
        return False

    return True


def assert_picklable(obj):
    """
    Test if an object is picklable or not
    """

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
