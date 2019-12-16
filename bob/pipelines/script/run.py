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
@click.pass_context
def run(ctx, **kwargs):
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

    #2. DEFINING THE EXPERIMENT SETUP

    # 2.1 DATABASE
    protocol = "Default"
    
    import bob.db.atnt
    #import bob.db.mobio
    database = bob.db.atnt.Database()
    #database = bob.db.mobio.Database(original_directory="/idiap/resource/database/mobio/IMAGES_PNG", annotation_directory="/idiap/resource/database/mobio/IMAGE_ANNOTATIONS")


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

    # TODO: WE SHOULD WORK THIS OUT

    #training_samples = create_training_samples(database, protocol=protocol)

    ### 
    biometric_reference_samples = create_biometric_reference_samples(database, protocol=protocol)
    probe_samples = create_biometric_probe_samples(database, biometric_reference_samples, protocol=protocol)

    # 4. RUNNING THE PIPELINE
    from bob.pipelines.bob_bio.simple_pipeline import pipeline

    #"""
    delayeds = pipeline(
            database.objects(protocol="Default", groups="world"),
            [(k, database.objects(protocol="Default", groups="dev",
                purposes="enroll", model_ids=(k,))) for k in
                database.model_ids(groups="dev")],
            ## N.B.: Demangling probe_samples to KISS
            [(k.sample_id, k.data, [z.sample_id for z in
                k.biometric_references]) for k in probe_samples],
            preprocessor,
            extractor,
            algorithm,
            npartitions=len(client.cluster.workers),
            experiment_path = "my_experiment"
        )
    #"""

    """

    delayeds = pipeline(                
            database.objects(protocol=protocol, groups="world"),
            [(k, database.objects(protocol=protocol, groups="dev",
                purposes="enroll", model_ids=(k,))) for k in
                database.model_ids(groups="dev")],
            ## N.B.: Demangling probe_samples to KISS
            [(k.sample_id, k.data, [z.sample_id for z in
                k.biometric_references]) for k in probe_samples],
            preprocessor,
            extractor,
            algorithm,
            npartitions=len(client.cluster.workers),
            experiment_path = "my_mobio"
        )
    """

    scores = delayeds.compute(scheduler=client)
    client.shutdown()
