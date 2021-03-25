import copy
import functools
import os
import pickle
import tempfile

import h5py
import numpy as np

from bob.pipelines import DelayedSample
from bob.pipelines import DelayedSampleSet
from bob.pipelines import DelayedSampleSetCached
from bob.pipelines import Sample
from bob.pipelines import SampleSet
from bob.pipelines import hdf5_to_sample
from bob.pipelines import sample_to_hdf5


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

    # Testing delayed sampleset
    with tempfile.TemporaryDirectory() as dir_name:

        samples = [Sample(data, key=str(i)) for i, data in enumerate(X)]
        filename = os.path.join(dir_name, "samples.pkl")
        with open(filename, "wb") as f:
            f.write(pickle.dumps(samples))

        sampleset = DelayedSampleSet(functools.partial(_load, filename), key=1)

        assert len(sampleset) == n_samples
        assert sampleset.samples == samples

    # Testing delayed sampleset cached
    with tempfile.TemporaryDirectory() as dir_name:

        samples = [Sample(data, key=str(i)) for i, data in enumerate(X)]
        filename = os.path.join(dir_name, "samples.pkl")
        with open(filename, "wb") as f:
            f.write(pickle.dumps(samples))

        sampleset = DelayedSampleSetCached(functools.partial(_load, filename), key=1)

        assert len(sampleset) == n_samples
        assert sampleset.samples == samples


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


def test_delayed_samples():
    def load_data():
        return 0

    def load_annot():
        return "annotation"

    delayed_sample = DelayedSample(load_data, delayed_attributes=dict(annot=load_annot))
    assert delayed_sample.data == 0, delayed_sample.data
    assert delayed_sample.annot == "annotation", delayed_sample.annot

    child_sample = Sample(1, parent=delayed_sample)
    assert child_sample.data == 1, child_sample.data
    assert child_sample.annot == "annotation", child_sample.annot
    assert child_sample.__dict__ == {
        "data": 1,
        "annot": "annotation",
    }, child_sample.__dict__

    delayed_sample.annot = "changed"
    assert delayed_sample.annot == "changed", delayed_sample.annot
