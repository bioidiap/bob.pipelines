import random

import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import bob.pipelines as mario

from bob.pipelines import (
    CheckpointWrapper,
    DaskWrapper,
    Sample,
    SampleSet,
    SampleWrapper,
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
    assert mario.is_instance_nested(o, "o", C)
    assert mario.is_instance_nested(o, "o", B)
    assert mario.is_instance_nested(o, "o", A)

    o = C(B(object))
    assert mario.is_instance_nested(o, "o", C)
    assert mario.is_instance_nested(o, "o", B)
    assert not mario.is_instance_nested(o, "o", A)


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
