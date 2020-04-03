from bob.pipelines.sample import Sample, SampleSet, DelayedSample
import numpy

from nose.tools import assert_raises
import copy


def test_sampleset_collection():

    n_samples = 10
    X = numpy.ones(shape=(n_samples, 2), dtype=int)
    sampleset = SampleSet(
        [Sample(data, key=str(i)) for i, data in enumerate(X)], key="1"
    )
    assert len(sampleset) == n_samples

    # Testing insert
    sample = Sample(X, key=100)
    sampleset.insert(1, sample)
    assert len(sampleset) == n_samples + 1

    # Testing delete
    del sampleset[0]
    assert len(sampleset) == n_samples

    # Testing exception
    with assert_raises(ValueError):
        sampleset.insert(1, 10)

    # Testing set
    sampleset[0] = copy.deepcopy(sample)

    # Testing exception
    with assert_raises(ValueError):
        sampleset[0] = "xuxa"

    # Testing iterator
    for i in sampleset:
        assert isinstance(i, Sample)
