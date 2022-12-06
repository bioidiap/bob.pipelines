import copy
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


def flatten_samplesets(samplesets):
    """
    Takes a list of SampleSets (with one or multiple samples in each SampleSet)
    and returns a list of SampleSets (with one sample in each SampleSet)

    Parameters
    ----------

    samplesets: list of :obj:`bob.pipelines.SampleSet`
      Input list of SampleSets (with one or multiple samples in each SampleSet

    """
    new_samplesets = []

    # Iterating over the samplesets
    for sset in samplesets:
        # Iterating over the samples, and deep copying each sampleset
        # for each sample
        for i, s in enumerate(sset):
            new_sset = copy.deepcopy(sset)
            new_sset.samples = [s]
            # Very important step
            # We need to redo the keys
            new_sset.key = f"{new_sset.key}-{i}"

            new_samplesets.append(new_sset)

    return new_samplesets


def check_parameters_for_validity(
    parameters, parameter_description, valid_parameters, default_parameters=None
):
    """Checks the given parameters for validity.

    Checks a given parameter is in the set of valid parameters. It also
    assures that the returned parameters form a list.  If parameters is
    'None' or empty, the default_parameters will be returned (if
    default_parameters is omitted, all valid_parameters are returned).

    This function will return a list of parameters, or raise a
    ValueError.


    Parameters
    ----------
    parameters : str or list of :obj:`str` or None
            The parameters to be checked. Might be a string, a list/tuple of
            strings, or None.

    parameter_description : str
            A short description of the parameter. This will be used to raise an
            exception in case the parameter is not valid.

    valid_parameters : list of :obj:`str`
            A list/tuple of valid values for the parameters.

    default_parameters : list of :obj:`str` or None
            The list/tuple of default parameters that will be returned in case
            parameters is None or empty. If omitted, all valid_parameters are used.

    Returns
    -------
    list
            A list containing the valid parameters.

    Raises
    ------
    ValueError
            If some of the parameters are not valid.

    """

    if not parameters:
        # parameters are not specified, i.e., 'None' or empty lists
        parameters = (
            default_parameters
            if default_parameters is not None
            else valid_parameters
        )

    if not isinstance(parameters, (list, tuple, set)):
        # parameter is just a single element, not a tuple or list -> transform it
        # into a tuple
        parameters = (parameters,)

    # perform the checks
    for parameter in parameters:
        if parameter not in valid_parameters:
            raise ValueError(
                "Invalid %s '%s'. Valid values are %s, or lists/tuples of those"
                % (parameter_description, parameter, valid_parameters)
            )

    # check passed, now return the list of parameters
    return list(parameters)


def check_parameter_for_validity(
    parameter, parameter_description, valid_parameters, default_parameter=None
):
    """Checks the given parameter for validity

    Ensures a given parameter is in the set of valid parameters. If the
    parameter is ``None`` or empty, the value in ``default_parameter`` will
    be returned, in case it is specified, otherwise a :py:exc:`ValueError`
    will be raised.

    This function will return the parameter after the check tuple or list
    of parameters, or raise a :py:exc:`ValueError`.

    Parameters
    ----------
    parameter : :obj:`str` or :obj:`None`
            The single parameter to be checked. Might be a string or None.

    parameter_description : str
            A short description of the parameter. This will be used to raise an
            exception in case the parameter is not valid.

    valid_parameters : list of :obj:`str`
            A list/tuple of valid values for the parameters.

    default_parameter : list of :obj:`str`, optional
            The default parameter that will be returned in case parameter is None or
            empty. If omitted and parameter is empty, a ValueError is raised.

    Returns
    -------
    str
            The validated parameter.

    Raises
    ------
    ValueError
            If the specified parameter is invalid.

    """

    if parameter is None:
        # parameter not specified ...
        if default_parameter is not None:
            # ... -> use default parameter
            parameter = default_parameter
        else:
            # ... -> raise an exception
            raise ValueError(
                "The %s has to be one of %s, it might not be 'None'."
                % (parameter_description, valid_parameters)
            )

    if isinstance(parameter, (list, tuple, set)):
        # the parameter is in a list/tuple ...
        if len(parameter) > 1:
            raise ValueError(
                "The %s has to be one of %s, it might not be more than one "
                "(%s was given)."
                % (parameter_description, valid_parameters, parameter)
            )
        # ... -> we take the first one
        parameter = parameter[0]

    # perform the check
    if parameter not in valid_parameters:
        raise ValueError(
            "The given %s '%s' is not allowed. Please choose one of %s."
            % (parameter_description, parameter, valid_parameters)
        )

    # tests passed -> return the parameter
    return parameter
