import bob.pipelines as mario
import numpy

import copy


def test_sampleset_collection():

    n_samples = 10
    X = numpy.ones(shape=(n_samples, 2), dtype=int)
    sampleset = mario.SampleSet(
        [mario.Sample(data, key=str(i)) for i, data in enumerate(X)], key="1"
    )
    assert len(sampleset) == n_samples

    # Testing insert
    sample = mario.Sample(X, key=100)
    sampleset.insert(1, sample)
    assert len(sampleset) == n_samples + 1

    # Testing delete
    del sampleset[0]
    assert len(sampleset) == n_samples

    # Testing set
    sampleset[0] = copy.deepcopy(sample)

    # Testing iterator
    for i in sampleset:
        assert isinstance(i, mario.Sample)
