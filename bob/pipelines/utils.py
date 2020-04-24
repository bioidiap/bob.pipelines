#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

def assert_picklable(obj):
    """
    Test if an object is picklable or not
    """
    import pickle

    try:
        pickle.dumps(obj)
    except TypeError:
        return False
    except pickle.PicklingError:
        return False

    return True
