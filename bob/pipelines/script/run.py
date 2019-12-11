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

    #1. DEFINING THE SCHEDU
    from bob.pipelines.distributed.local import debug_client
    from bob.pipelines.distributed.sge import sge_iobig_client

    client = debug_client(1)


    #2. Defining the bob.db database
    import bob.db.atnt
    database = bob.db.atnt.Database()


    #3 defining all the necessary tools to run the pipeline
    import numpy
    import bob.bio.face
    preprocessor = bob.bio.face.preprocessor.Base(color_channel="gray",
                                                  dtype = numpy.float64)

    #4 getting samples
    from bob.pipelines.samples.biometric_samples import create_training_samples, create_biometric_reference_samples, create_biometric_probe_samples

    training_samples = create_training_samples(database)
    biometric_reference_samples = create_biometric_reference_samples(database)
    probe_samples = create_biometric_probe_samples(database, biometric_reference_samples)
    
    # 5 fetching the pipeline
    from bob.pipelines.bob_bio.simple_pipeline import pipeline

    pipeline(training_samples,
            biometric_reference_samples,
            probe_samples,
            preprocessor,
            client
            )

    pass
