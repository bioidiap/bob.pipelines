import bob.pipelines as mario
import numpy

import copy
import pickle
import tempfile
import functools
import os

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


    def _load(path):
        return pickle.loads(open(path, "rb").read())

    # Testing delayed sample in the sampleset
    with tempfile.TemporaryDirectory() as dir_name:
        
        samples = [mario.Sample(data, key=str(i)) for i, data in enumerate(X)]
        filename = os.path.join(dir_name, "samples.pkl")
        with open(filename, "wb") as f:
            f.write(pickle.dumps(samples))

        sampleset = mario.SampleSet(mario.DelayedSample(functools.partial(_load, filename)), key=1)

        assert len(sampleset)==n_samples