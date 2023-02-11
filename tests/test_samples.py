import copy
import functools
import os
import pickle
import tempfile

import numpy as np
import pytest

from bob.pipelines import (
    DelayedSample,
    DelayedSampleSet,
    DelayedSampleSetCached,
    Sample,
    SampleSet,
)


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

        sampleset = DelayedSampleSetCached(
            functools.partial(_load, filename), key=1
        )

        assert len(sampleset) == n_samples
        assert sampleset.samples == samples


def test_delayed_samples():
    def load_data():
        return 0

    def load_annot():
        return "annotation"

    def load_annot_variant():
        return "annotation_variant"

    delayed_sample = DelayedSample(
        load_data, delayed_attributes=dict(annot=load_annot)
    )
    assert delayed_sample.data == 0
    assert delayed_sample.annot == "annotation"

    child_sample = Sample(1, parent=delayed_sample)
    assert child_sample.data == 1
    assert child_sample.annot == "annotation"
    assert child_sample.__dict__ == {
        "data": 1,
        "annot": "annotation",
    }

    # Overwriting and adding delayed_attributes to the child
    delayed_attr_read = False

    def load_check():
        nonlocal delayed_attr_read
        delayed_attr_read = True
        return "delayed_attr_data"

    new_delayed_attr = {
        "annot": load_annot_variant,  # Override parent's annot
        "new_annot": load_annot,  # Add the new_annot attribute
        "read_check": load_check,
    }
    child_sample = DelayedSample(
        load_data, parent=delayed_sample, delayed_attributes=new_delayed_attr
    )

    assert delayed_sample._delayed_attributes == dict(annot=load_annot)
    assert child_sample.data == 0
    assert child_sample.annot == "annotation_variant"
    assert child_sample.new_annot == "annotation"
    assert not delayed_attr_read, "delayed attribute has been read early"
    assert child_sample.read_check == "delayed_attr_data"
    assert delayed_attr_read, "delayed attribute should have been read by now"

    delayed_sample.annot = "changed"
    assert delayed_sample.annot == "changed"

    # Test DelayedSample.from_sample

    # Check if a non-delayed sample is correctly converted to a delayed sample
    non_delayed_sample = Sample(1)
    delayed_sample = DelayedSample.from_sample(non_delayed_sample, annot="test")
    assert delayed_sample.data == 1
    assert delayed_sample.annot == "test"

    # Check if converting a delayed sample will not load the data and delayed attributes
    raise_error = True

    def never_load():
        nonlocal raise_error
        if raise_error:
            raise ValueError("never_load should not be called")
        return 0

    delayed_sample = DelayedSample(
        never_load, delayed_attributes=dict(annot=never_load)
    )
    # Should not raise an error when creating the delayed sample
    child_sample = DelayedSample.from_sample(delayed_sample)

    raise_error = False
    assert child_sample.data == 0
    assert child_sample.annot == 0

    # If attribute is in kwargs and delayed_sample, it should raise an exception
    with pytest.raises(ValueError):
        DelayedSample(
            load_data,
            delayed_attributes={"annotations": load_annot},
            annotations="some_other_annotations",
        )

    # kwargs should take precedence over the parent's delayed_attributes
    parent = DelayedSample(
        load_data,
        delayed_attributes={"annotation": load_annot},
        non_delay_attribute=4,
    )
    # annotation=100 overwrites parent.annotation
    child = DelayedSample(lambda: 12, parent=parent, annotation=100)

    assert child.data == 12
    assert child._delayed_attributes is None
    assert child.annotation == 100
