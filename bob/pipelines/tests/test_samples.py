from bob.pipelines import (
    Sample,
    DelayedSample,
    SampleSet,
    sample_to_hdf5,
    hdf5_to_sample,
)
import bob.io.base
import numpy as np

import copy
import pickle
import tempfile
import functools
import os
import h5py


def test_sampleset_collection():

    n_samples = 10
    X = np.ones(shape=(n_samples, 2), dtype=int)
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

    # Testing set
    sampleset[0] = copy.deepcopy(sample)

    # Testing iterator
    for i in sampleset:
        assert isinstance(i, Sample)

    def _load(path):
        return pickle.loads(open(path, "rb").read())

    # Testing delayed sample in the sampleset
    with tempfile.TemporaryDirectory() as dir_name:

        samples = [Sample(data, key=str(i)) for i, data in enumerate(X)]
        filename = os.path.join(dir_name, "samples.pkl")
        with open(filename, "wb") as f:
            f.write(pickle.dumps(samples))

        sampleset = SampleSet(DelayedSample(functools.partial(_load, filename)), key=1)

        assert len(sampleset) == n_samples


def test_sample_hdf5():
    n_samples = 10
    X = np.ones(shape=(n_samples, 2), dtype=int)

    samples = [Sample(data, key=str(i), subject="Subject") for i, data in enumerate(X)]
    with tempfile.TemporaryDirectory() as dir_name:

        # Single sample
        filename = os.path.join(dir_name, "sample.hdf5")

        with h5py.File(filename, "w", driver="core") as hdf5:
            sample_to_hdf5(samples[0], hdf5)

        with h5py.File(filename, "r") as hdf5:
            sample = hdf5_to_sample(hdf5)

        assert sample == samples[0]

        # List of samples
        filename = os.path.join(dir_name, "samples.hdf5")
        with h5py.File(filename, "w", driver="core") as hdf5:
            sample_to_hdf5(samples, hdf5)

        with h5py.File(filename, "r") as hdf5:
            samples_deserialized = hdf5_to_sample(hdf5)

        compare = [a == b for a, b in zip(samples_deserialized, samples)]
        assert np.sum(compare) == 10
