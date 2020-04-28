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
import tempfile
import os
import numpy as np
with tempfile.TemporaryDirectory() as dir_name:
   checkpointing_transformer = mario.CheckpointWrapper(features_dir=dir_name)

   # now let's create some samples with ``key`` metadata
   # Creating X: 3 samples, 2 features
   X = np.zeros((3, 2))
   # 3 offsets: one for each sample
   offsets = np.arange(3).reshape((3, 1))
   # key values must be string because they will be used to create file names.
   samples = [mario.Sample(x, offset=o, key=str(i)) for i, (x, o) in enumerate(zip(X, offsets))]
   samples[0]

   # let's transform them
   transformed_samples = checkpointing_transformer.transform(samples)

   # Let's check the features directory
   os.listdir(dir_name)

