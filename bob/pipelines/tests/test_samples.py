import copy
import functools
import os
import pickle
import tempfile

import numpy as np

from bob.pipelines import DelayedSample
from bob.pipelines import DelayedSampleSet
from bob.pipelines import DelayedSampleSetCached
from bob.pipelines import Sample
from bob.pipelines import SampleSet


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


def test_delayed_samples():
    def load_data():
        return 0

    def load_annot():
        return "annotation"

    def load_annot_variant():
        return "annotation_variant"

    delayed_attr_read = False

    def load_check():
        nonlocal delayed_attr_read
        delayed_attr_read = True
        return "delayed_attr_data"

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

    # Overwriting and adding delayed_attributes to the child
    new_delayed_attr = {
        "annot": load_annot_variant,  # Override parent's annot
        "new_annot": load_annot,  # Add the new_annot attribute
        "read_check": load_check,
    }
    child_sample = DelayedSample(
        load_data, parent=delayed_sample, delayed_attributes=new_delayed_attr
    )

    assert delayed_sample._delayed_attributes == dict(annot=load_annot)
    assert child_sample.data == 0, child_sample.data
    assert child_sample.annot == "annotation_variant", child_sample.annot
    assert child_sample.new_annot == "annotation", child_sample.new_annot
    assert not delayed_attr_read, "delayed attribute has been read early"
    assert child_sample.read_check == "delayed_attr_data", child_sample.read_check
    assert delayed_attr_read, "delayed attribute should have been read by now"

    delayed_sample.annot = "changed"
    assert delayed_sample.annot == "changed", delayed_sample.annot
