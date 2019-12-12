#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""
Execute a particular pipeline
"""
from bob.extension.scripts.click_helper import (
            verbosity_option, log_parameters)
import click
from bob.extension.scripts.click_helper import verbosity_option, ResourceOption
from bob.extension.config import load

EPILOG = '''\b


'''

@click.command(context_settings={'ignore_unknown_options': True,
                                 'allow_extra_args': True})
@verbosity_option(cls=ResourceOption)
@click.option('--use-dask-delay', is_flag=True)
@click.pass_context
def run(ctx, use_dask_delay, **kwargs):
    """Run a pipeline

    FROM THE TIME BEING NOT PASSING ANY PARAMETER

    \b

    bob pipelines run 


    """


    # TODO: THIS WILL BE THE MAIN EXECUTOR
    # FOR TEST PURPOSES EVERYTHING IS HARD CODED IN THIS SCRIPT.
    # THE GOAL IS JUST TO GRASP ALL THE NECESSARY TASKS AND POSSIBLE WORK
    # AROUNDS THAT WE WILL NEED TO DO TO FINISH THIS TASK


    #1. DEFINING THE CLIENT FOR EXECUTION
    #   THIS COULD BE IN A CONFIG FILE
    from bob.pipelines.distributed.local import debug_client
    from bob.pipelines.distributed.sge import sge_iobig_client

    client = debug_client(1)
    #client = sge_iobig_client(10)
    #import ipdb; ipdb.set_trace()


    #2. DEFINING THE EXPERIMENT SETUP

    # 2.1 DATABASE
    import bob.db.atnt
    database = bob.db.atnt.Database()


    # 2.1 SIGNAL PROCESSING AND ML TOOLS
    import numpy

    # 2.1.1 preprocessor
    import bob.bio.face
    preprocessor = bob.bio.face.preprocessor.Base(color_channel="gray",
                                                  dtype = numpy.float64)

    # 2.1.2 extractor
    import bob.bio.base
    extractor = bob.bio.base.extractor.Linearize()

    # 2.1.3 Algorithm
    from bob.bio.base.algorithm import PCA
    algorithm = PCA(0.99)

    # 2.1.........



    # 3. FETCHING SAMPLES
    from bob.pipelines.samples.biometric_samples import create_training_samples, create_biometric_reference_samples, create_biometric_probe_samples

    training_samples = create_training_samples(database)
    biometric_reference_samples = create_biometric_reference_samples(database)
    probe_samples = create_biometric_probe_samples(database, biometric_reference_samples)
    
    # 4. RUNNING THE PIPELINE
    from bob.pipelines.bob_bio.simple_pipeline import pipeline, pipeline_DELAY


    if use_dask_delay:
        pipeline_DELAY(training_samples,
                biometric_reference_samples,
                probe_samples,
                preprocessor,
                extractor,
                algorithm,
                client
                )
    else:
        pipeline(training_samples,
                biometric_reference_samples,
                probe_samples,
                preprocessor,
                algorithm,
                extractor,
                client
            )
    #import ipdb; ipdb.set_trace()
    client.shutdown()

    pass
