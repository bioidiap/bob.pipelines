import random

import numpy as np
import pytest

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import bob.pipelines

from bob.pipelines import (
    CheckpointWrapper,
    DaskWrapper,
    Sample,
    SampleSet,
    SampleWrapper,
    check_parameter_for_validity,
    check_parameters_for_validity,
    flatten_samplesets,
    is_pipeline_wrapped,
    wrap,
)


def test_is_pipeline_wrapped():
    def do_something(X):
        return X

    my_pipe = make_pipeline(
        FunctionTransformer(do_something), FunctionTransformer(do_something)
    )

    np.testing.assert_array_equal(
        is_pipeline_wrapped(my_pipe, SampleWrapper), [False, False]
    )
    np.testing.assert_array_equal(
        is_pipeline_wrapped(my_pipe, CheckpointWrapper), [False, False]
    )
    np.testing.assert_array_equal(
        is_pipeline_wrapped(my_pipe, DaskWrapper), [False, False]
    )

    # Sample wrap
    my_pipe = wrap(["sample"], my_pipe)
    np.testing.assert_array_equal(
        is_pipeline_wrapped(my_pipe, SampleWrapper), [True, True]
    )
    np.testing.assert_array_equal(
        is_pipeline_wrapped(my_pipe, CheckpointWrapper), [False, False]
    )
    np.testing.assert_array_equal(
        is_pipeline_wrapped(my_pipe, DaskWrapper), [False, False]
    )

    # Checkpoint wrap
    my_pipe = wrap(["checkpoint"], my_pipe)
    np.testing.assert_array_equal(
        is_pipeline_wrapped(my_pipe, SampleWrapper), [True, True]
    )
    np.testing.assert_array_equal(
        is_pipeline_wrapped(my_pipe, CheckpointWrapper), [True, True]
    )
    np.testing.assert_array_equal(
        is_pipeline_wrapped(my_pipe, DaskWrapper), [False, False]
    )

    # Dask wrap
    my_pipe = wrap(["dask"], my_pipe)
    np.testing.assert_array_equal(
        is_pipeline_wrapped(my_pipe, SampleWrapper), [False, True, True]
    )
    np.testing.assert_array_equal(
        is_pipeline_wrapped(my_pipe, CheckpointWrapper), [False, True, True]
    )
    np.testing.assert_array_equal(
        is_pipeline_wrapped(my_pipe, DaskWrapper), [False, True, True]
    )


def test_is_instance_nested():
    class A:
        pass

    class B:
        def __init__(self, o):
            self.o = o

    class C:
        def __init__(self, o):
            self.o = o

    o = C(B(A()))
    assert bob.pipelines.is_instance_nested(o, "o", C)
    assert bob.pipelines.is_instance_nested(o, "o", B)
    assert bob.pipelines.is_instance_nested(o, "o", A)

    o = C(B(object))
    assert bob.pipelines.is_instance_nested(o, "o", C)
    assert bob.pipelines.is_instance_nested(o, "o", B)
    assert not bob.pipelines.is_instance_nested(o, "o", A)


def test_break_sample_set():
    samplesets = []
    n_samples = 10
    X = np.ones(shape=(n_samples, 2), dtype=int)
    random.seed(10)

    # Creating a face list of samplesets
    for i in range(n_samples):
        samplesets.append(
            SampleSet(
                [
                    Sample(
                        data,
                        key=str(i),
                        sample_random_attriute="".join(
                            [random.choice("abcde") for _ in range(5)]
                        ),
                    )
                    for i, data in enumerate(X)
                ],
                key=str(i),
                sampleset_random_attriute="".join(
                    [random.choice("abcde") for _ in range(5)]
                ),
            )
        )

    # Flatting the SSET
    new_samplesets = flatten_samplesets(samplesets)

    assert len(new_samplesets) == n_samples * n_samples
    assert np.allclose(
        [len(s) for s in new_samplesets], np.ones(n_samples * n_samples)
    )


def test_check_parameter_validity():
    valid_values = ["accept", "true"]
    desc_str = "desc_str"

    # Valid parameter
    param = "true"
    retval = check_parameter_for_validity(param, "desc_str", valid_values)
    assert retval == param

    # Default value
    param = None
    default = "accept"
    retval = check_parameter_for_validity(
        param, desc_str, valid_values, default
    )
    assert retval == default

    # Invalid parameter
    param = "false"
    with pytest.raises(ValueError) as except_info:
        retval = check_parameter_for_validity(param, desc_str, valid_values)
    assert (
        str(except_info.value)
        == f"The given {desc_str} '{param}' is not allowed. Please choose one of {valid_values}."
    )

    # Invalid default parameter
    param = None
    default = "false"
    with pytest.raises(ValueError) as except_info:
        retval = check_parameter_for_validity(
            param, desc_str, valid_values, default
        )
    assert (
        str(except_info.value)
        == f"The given {desc_str} '{default}' is not allowed. Please choose one of {valid_values}."
    )

    # Too many parameters
    param = ["accept", "accept"]
    with pytest.raises(ValueError) as except_info:
        retval = check_parameter_for_validity(param, desc_str, valid_values)
    assert (
        str(except_info.value)
        == f"The {desc_str} has to be one of {valid_values}, it might not be more than one ({param} was given)."
    )


def test_check_parameters_validity():
    valid_values = ["accept", "true"]
    desc_str = "desc_str"

    # Valid single parameter
    param = "true"
    retval = check_parameters_for_validity(param, "desc_str", valid_values)
    assert type(retval) is list
    assert retval == [param]

    # Valid multi parameter
    param = ["true", "accept"]
    retval = check_parameters_for_validity(param, "desc_str", valid_values)
    assert type(retval) is list
    assert retval == param

    # Tuple parameter
    param = ("true", "accept")
    retval = check_parameters_for_validity(param, "desc_str", valid_values)
    assert type(retval) is list
    assert retval == list(param)

    # Default value
    param = None
    default = "accept"
    retval = check_parameters_for_validity(
        param, desc_str, valid_values, default
    )
    assert type(retval) is list
    assert retval == [default]

    # Invalid parameter
    param = "false"
    with pytest.raises(ValueError) as except_info:
        retval = check_parameters_for_validity(param, desc_str, valid_values)
    assert (
        str(except_info.value)
        == f"Invalid {desc_str} '{param}'. Valid values are {valid_values}, or lists/tuples of those"
    )

    # Invalid multi parameter
    param = ["accept", "false"]
    with pytest.raises(ValueError) as except_info:
        retval = check_parameters_for_validity(param, desc_str, valid_values)
    assert (
        str(except_info.value)
        == f"Invalid {desc_str} '{param[1]}'. Valid values are {valid_values}, or lists/tuples of those"
    )

    # Empty default parameter
    param = None
    default = None
    retval = check_parameters_for_validity(
        param, desc_str, valid_values, default
    )
    assert type(retval) is list
    assert retval == valid_values

    # Invalid default parameter
    param = None
    default = "false"
    with pytest.raises(ValueError) as except_info:
        retval = check_parameters_for_validity(
            param, desc_str, valid_values, default
        )
    assert (
        str(except_info.value)
        == f"Invalid {desc_str} '{default}'. Valid values are {valid_values}, or lists/tuples of those"
    )
