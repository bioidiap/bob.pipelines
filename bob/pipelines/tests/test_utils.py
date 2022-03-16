import random

import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import bob.pipelines as mario

from bob.pipelines import Sample, SampleSet
from bob.pipelines.utils import flatten_samplesets, is_pipeline_wrapped
from bob.pipelines.wrappers import (
    CheckpointWrapper,
    DaskWrapper,
    SampleWrapper,
    wrap,
)


def test_is_pipeline_wrapped():
    def do_something(X):
        return X

    my_pipe = make_pipeline(
        FunctionTransformer(do_something), FunctionTransformer(do_something)
    )

    assert not np.alltrue(is_pipeline_wrapped(my_pipe, SampleWrapper))
    assert not np.alltrue(is_pipeline_wrapped(my_pipe, CheckpointWrapper))
    assert not np.alltrue(is_pipeline_wrapped(my_pipe, DaskWrapper))

    # Sample wrap
    my_pipe = wrap(["sample"], my_pipe)
    assert np.alltrue(is_pipeline_wrapped(my_pipe, SampleWrapper))
    assert not np.alltrue(is_pipeline_wrapped(my_pipe, CheckpointWrapper))
    assert not np.alltrue(is_pipeline_wrapped(my_pipe, DaskWrapper))

    # Checkpoint wrap
    my_pipe = wrap(["checkpoint"], my_pipe)
    assert np.alltrue(is_pipeline_wrapped(my_pipe, SampleWrapper))
    assert np.alltrue(is_pipeline_wrapped(my_pipe, CheckpointWrapper))
    assert not np.alltrue(is_pipeline_wrapped(my_pipe, DaskWrapper))

    # Dask wrap
    my_pipe = wrap(["dask"], my_pipe)
    assert np.alltrue(is_pipeline_wrapped(my_pipe, SampleWrapper))
    assert np.alltrue(is_pipeline_wrapped(my_pipe, CheckpointWrapper))
    assert np.alltrue(is_pipeline_wrapped(my_pipe, DaskWrapper))


def test_isinstance_nested():
    class A:
        pass

    class B:
        def __init__(self, o):
            self.o = o

    class C:
        def __init__(self, o):
            self.o = o

    o = C(B(A()))
    assert mario.utils.isinstance_nested(o, "o", C)
    assert mario.utils.isinstance_nested(o, "o", B)
    assert mario.utils.isinstance_nested(o, "o", A)

    o = C(B(object))
    assert mario.utils.isinstance_nested(o, "o", C)
    assert mario.utils.isinstance_nested(o, "o", B)
    assert not mario.utils.isinstance_nested(o, "o", A)


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
