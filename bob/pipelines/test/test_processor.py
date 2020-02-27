#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import numpy

from bob.pipelines.processor import (
    ProcessorBlock,
    ProcessorPipeline
    )

from bob.pipelines.sample import (
    Sample, SampleSet, DelayedSample
    )


class Dummy(ProcessorBlock):

    def __init__(self, offset=1, **kwargs):
        self.is_fittable = False
        self.offset = offset
        super(Dummy, self).__init__()


    def transform(self, X):
        return X + self.offset


def test_processor_pipeline_only_transform():
    
    X_transform = numpy.zeros(shape=(10,2), dtype=int)

    sampleset_transform = [SampleSet([Sample(X_transform), Sample(X_transform+1)])]

    pip      = [("dummy1", Dummy(offset=1)), ("dummy2", Dummy(offset=2))]
    pipeline = ProcessorPipeline(pip)
        
    X_new_transform = pipeline.transform(sampleset_transform)

    assert numpy.allclose(X_new_transform[0].samples[0].data, numpy.zeros(shape=(10,2), dtype=int)+3)


def test_processor_pipeline_only_transform_delay():
    
    X_transform = numpy.zeros(shape=(10,2), dtype=int)

    sampleset_transform = [SampleSet([Sample(X_transform), Sample(X_transform+1)])]

    candidate = "./test_dummy"
    checkpoints = {"dummy2": candidate}

    pip      = [("dummy1", Dummy(offset=1)), ("dummy2", Dummy(offset=2))]
    pipeline = ProcessorPipeline(pip)
    
    X_new_transform = pipeline.transform(sampleset_transform, checkpoints)

    assert numpy.allclose(X_new_transform[0].samples[0].data, numpy.zeros(shape=(10,2), dtype=int)+3)

