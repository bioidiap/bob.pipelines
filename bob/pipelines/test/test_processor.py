#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import numpy
import os
from bob.pipelines.processor import ProcessorBlock, ProcessorPipeline

from bob.pipelines.sample import Sample, SampleSet, DelayedSample
import pkg_resources
import shutil
import functools


class Dummy(ProcessorBlock):
    def __init__(self, offset=1, **kwargs):
        self.is_fittable = False
        self.offset = offset
        super(Dummy, self).__init__()

    def transform(self, X):
        return X + self.offset


class DummyWithFit(ProcessorBlock):
    def __init__(self, **kwargs):
        super(DummyWithFit, self).__init__(is_fittable=True, **kwargs)

    def fit(self, X, y=None, **kwargs):
        super(DummyWithFit, self).fit(X, y, **kwargs)
        self.model = numpy.ones((2, 2))

    def transform(self, X):
        return X @ self.model


def test_processor_pipeline_only_transform():
    """
    Simple sequence of transformation.
    Here the processor are TRANSFORMABLE only
    """

    X_transform = numpy.zeros(shape=(10, 2), dtype=int)

    sampleset_transform = [SampleSet([Sample(X_transform), Sample(X_transform + 1)])]

    pip = [("dummy1", Dummy(offset=1)), ("dummy2", Dummy(offset=2))]
    pipeline = ProcessorPipeline(pip)

    X_new_transform = pipeline.transform(sampleset_transform)

    assert numpy.allclose(
        X_new_transform[0].samples[0].data, numpy.zeros(shape=(10, 2), dtype=int) + 3
    )


def test_processor_pipeline_only_transform_delay():
    """
    Simple sequence of transformation.
    Here the processor are TRANSFORMABLE only and they are checkpointable
    """

    candidate_path = pkg_resources.resource_filename("bob.pipelines", "dummy_test")

    try:
        X_transform = numpy.zeros(shape=(10, 2), dtype=int)

        sampleset_transform = [
            SampleSet(
                [
                    Sample(X_transform, path="sample1"),
                    Sample(X_transform + 1, path="sample2"),
                ]
            )
        ]

        checkpoints = {
            "dummy1": os.path.join(candidate_path, "dummy1"),
            "dummy2": os.path.join(candidate_path, "dummy2"),
        }

        pip = [
            ("dummy1", Dummy(offset=1)),
            ("dummy2", functools.partial(Dummy, offset=2)),
        ]
        pipeline = ProcessorPipeline(pip)

        X_new_transform = pipeline.transform(sampleset_transform, checkpoints)

        assert numpy.allclose(
            X_new_transform[0].samples[0].data,
            numpy.zeros(shape=(10, 2), dtype=int) + 3,
        )

        dummy = Dummy()
        data = dummy.read(os.path.join(candidate_path, "dummy1", "sample1"))
        assert numpy.allclose(data, numpy.ones((10, 2), dtype=int))
    finally:
        shutil.rmtree(candidate_path)


def test_processor_pipeline_fit_transform():
    """
    Simple sequence of transformation.
    Here the processor are TRANSFORMABLE only AND FITTABLE 
    FURTHERMORE they are checkpointable
    """

    X_fit = numpy.zeros(shape=(10, 2), dtype=int)
    sampleset_fit = [SampleSet([Sample(X_fit), Sample(X_fit + 1)])]

    X_transform = numpy.ones(shape=(10, 2), dtype=int)
    sampleset_transform = [SampleSet([Sample(X_transform)])]

    pip = [("dummy1", Dummy(offset=1)), ("dummy2", DummyWithFit())]
    pipeline = ProcessorPipeline(pip)
    pipeline.fit(sampleset_fit)

    X_new_transform = pipeline.transform(sampleset_transform)
    assert numpy.allclose(
        X_new_transform[0].samples[0].data, numpy.zeros(shape=(10, 2), dtype=int) + 4
    )


def test_processor_pipeline_fit_transform_with_model_checkpoint():
    """
    Simple sequence of transformation.
    Here the processor are TRANSFORMABLE only AND FITTABLE 
    FURTHERMORE only one is checkpointable
    """

    candidate_path = pkg_resources.resource_filename("bob.pipelines", "dummy_test")
    try:
        X_fit = numpy.zeros(shape=(10, 2), dtype=int)
        sampleset_fit = [SampleSet([Sample(X_fit), Sample(X_fit + 1)])]

        # Building a pipeline for fitting
        pip = [("dummy1", Dummy(offset=1)), ("dummy2", DummyWithFit())]
        checkpoints = {"dummy2": os.path.join(candidate_path, "dummy2")}
        pipeline = ProcessorPipeline(pip)
        pipeline.fit(sampleset_fit, checkpoints=checkpoints)

        # Building a pipeline for transforming.
        # This one will load the model directly in the worker
        X_transform = numpy.ones(shape=(10, 2), dtype=int)
        sampleset_transform = [SampleSet([Sample(X_transform, path="sample1")])]

        new_pipeline = ProcessorPipeline(pip)
        X_new_transform = new_pipeline.transform(
            sampleset_transform, checkpoints=checkpoints
        )
        assert numpy.allclose(
            X_new_transform[0].samples[0].data,
            numpy.zeros(shape=(10, 2), dtype=int) + 4,
        )
    finally:
        shutil.rmtree(candidate_path)


def test_processor_pipeline_fit_transform_with_model_checkpoint_2():

    candidate_path = pkg_resources.resource_filename("bob.pipelines", "dummy_test")
    try:
        X_fit = numpy.zeros(shape=(10, 2), dtype=int)
        sampleset_fit = [SampleSet([Sample(X_fit), Sample(X_fit + 1)])]

        # Building a pipeline for fitting
        pip = [
            ("dummy1", Dummy(offset=1)),
            ("dummy2", DummyWithFit()),
            ("dummy3", DummyWithFit()),
        ]
        checkpoints = {"dummy2": os.path.join(candidate_path, "dummy2")}
        pipeline = ProcessorPipeline(pip)
        pipeline.fit(sampleset_fit, checkpoints=checkpoints)

        # Building a pipeline for transforming.
        # This one will load the model directly in the worker
        X_transform = numpy.ones(shape=(10, 2), dtype=int)
        sampleset_transform = [SampleSet([Sample(X_transform, path="sample1")])]

        X_new_transform = pipeline.transform(
            sampleset_transform, checkpoints=checkpoints
        )
        assert numpy.allclose(
            X_new_transform[0].samples[0].data,
            numpy.zeros(shape=(10, 2), dtype=int) + 8,
        )
    finally:
        shutil.rmtree(candidate_path)
