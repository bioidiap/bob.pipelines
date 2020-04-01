from bob.pipelines.sample import Sample, SampleSet, DelayedSample
import numpy

from nose.tools import assert_raises


def test_sampleset_collection():

    n_samples = 10
    X = numpy.ones(shape=(n_samples, 2), dtype=int)
    sampleset = SampleSet(
        [Sample(data, key=str(i)) for i, data in enumerate(X)], key="1"
    )
    assert len(sampleset) == n_samples

    # Testing __iter__
    for s in sampleset:
        assert hasattr(s, "key")

    # Testing __contains__
    for i in range(n_samples):
        assert i in sampleset
    assert 120 not in sampleset

    # Testing add
    sample = Sample(X, key=100)
    sampleset.add(sample)
    assert len(sampleset) == n_samples + 1

    # adding again, doesn't change
    sampleset.add(sample)
    assert len(sampleset) == n_samples + 1

    # Testing exception
    with assert_raises(ValueError):
        sampleset.add(10)

    # Testing discard
    sampleset.discard(100)
    assert len(sampleset) == n_samples
