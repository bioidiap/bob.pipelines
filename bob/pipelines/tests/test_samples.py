from bob.pipelines import Sample, SampleSet, DelayedSample
import numpy as np
from distributed.protocol.serialize import serialize,deserialize

import copy
import pickle
import msgpack
import tempfile
import functools
import os

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

        assert len(sampleset)==n_samples


def test_sample_serialization():
    sample = Sample(np.random.rand(1, 1, 2), key=1)
    header, frame = serialize(sample)
    deserialized_sample = deserialize(header, frame)
    assert isinstance(deserialized_sample, Sample)


    # Testing serialization Sampleset
    sample = Sample(np.random.rand(1, 1, 2), key=1)
    sampleset = SampleSet([sample], key=1)
    header, frame = serialize(sampleset)
    deserialized_sampleset = deserialize(header, frame)

    assert isinstance(deserialized_sampleset, SampleSet)
    deserialized_sampleset[0] = Sample(np.random.rand(3, 480, 400), key=1)

    # serialize again
    header, frame = serialize(deserialized_sampleset)
    deserialized_sampleset = deserialize(header, frame)
    assert isinstance(deserialized_sampleset, SampleSet)
    
    # Testing list serialization
    header, frame = serialize([deserialized_sampleset])
    deserialized_sampleset = deserialize(header, frame)
    assert isinstance(deserialized_sampleset, list)
    assert isinstance(deserialized_sampleset[0], SampleSet)


def test_sample_serialization_scale():

    def create_samplesets(n_sample_sets, n_samples):
        return [
            SampleSet(
                [Sample(data=np.random.rand(20, 1,)) for _ in range(n_samples)],
                key=i,
                references=list(range(1000))
            )
            for i in range(n_sample_sets)
        ]

    samplesets = create_samplesets(10, 10)    
    header, frame = serialize(samplesets)

    # header needs to be serializable with msgpack
    msgpack.dumps(header)

    deserialized_samplesets = deserialize(header, frame)
    assert isinstance(deserialized_samplesets, list)
    assert isinstance(deserialized_samplesets[0], SampleSet)


def test_sample_serialization_delayed():

    with tempfile.TemporaryDirectory() as dir_name:
        

        def create_samplesets(n_sample_sets, n_samples, as_list=False):

            samples = [Sample(data=np.random.rand(20, 1,)) for _ in range(n_samples)]
            filename = os.path.join(dir_name, "xuxa.pkl")
            open(filename, "wb").write(pickle.dumps(samples))

            def _load(path):
                return pickle.loads(open(path, "rb").read())

            if as_list:
                delayed_samples = [DelayedSample(functools.partial(_load, filename), key=1, references=list(range(1000))   )]
            else:
                delayed_samples = DelayedSample(functools.partial(_load, filename), key=1, references=np.array(list(range(1000)), dtype="float")    )

            return [
                SampleSet(
                    delayed_samples,
                    key=i,
                    references=np.array(list(range(1000)), dtype="float")
                )
                for i in range(n_sample_sets)
            ]
        
        samplesets = create_samplesets(1, 10, as_list=False)
        header, frame = serialize(samplesets)
        # header needs to be serializable with msgpack        
        msgpack.dumps(header)

        deserialized_samplesets = deserialize(header, frame)

        assert isinstance(deserialized_samplesets, list)
        assert isinstance(deserialized_samplesets[0], SampleSet)

        
        # Testing list of samplesets
        samplesets = create_samplesets(1, 10, as_list=True)
        header, frame = serialize(samplesets)

        # header needs to be serializable with msgpack
        msgpack.dumps(header)

        deserialized_samplesets = deserialize(header, frame)
        assert isinstance(deserialized_samplesets, list)
        assert isinstance(deserialized_samplesets[0], SampleSet)
        