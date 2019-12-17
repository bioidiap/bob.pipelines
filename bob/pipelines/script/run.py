#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""Executes a particular pipeline"""

import os
import functools

import click

from bob.extension.scripts.click_helper import verbosity_option, ResourceOption


EPILOG = """\b


"""


@click.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.option(
    "-o",
    "--output",
    show_default=True,
    default="results",
    help="Name of output directory",
)
@verbosity_option(cls=ResourceOption)
@click.pass_context
def run(ctx, output, **kwargs):
    """Runs a pipeline

    FOR THE TIME BEING, NOT PASSING ANY PARAMETER

    \b

    bob pipelines run


    """

    # Configures the algorithm (concrete) implementations for the pipeline
    import bob.db.atnt
    from ..bob_bio.blocks import DatabaseConnector
    database = DatabaseConnector(bob.db.atnt.Database(), protocol="Default")

    from ..bob_bio.blocks import SampleLoader
    import bob.bio.base
    import bob.bio.face

    loader = SampleLoader(
        functools.partial(
            bob.bio.face.preprocessor.Base,
            color_channel="gray",
            dtype="float64",
        ),
        bob.bio.base.extractor.Linearize,
    )

    from ..bob_bio.blocks import AlgorithmAdaptor
    from bob.bio.base.algorithm import PCA
    algorithm = AlgorithmAdaptor(
        functools.partial(PCA, 0.99), os.path.join(output, "background",
            "model.hdf5"),
    )

    # Configures the execution context
    from bob.pipelines.distributed.local import debug_client
    client = debug_client(1)
    # from bob.pipelines.distributed.sge import sge_iobig_client
    # client = sge_iobig_client(10)

    # Chooses the pipeline to run
    from bob.pipelines.bob_bio.pipelines import first as pipeline

    if not os.path.exists(output):
        os.makedirs(output)

    result = pipeline(
        database.background_model_samples(),
        database.references(group="dev"),
        database.probes(group="dev"),
        loader,
        algorithm,
        npartitions=len(client.cluster.workers),
        cache=output,
    )

    # save graph on results directory
    result.visualize(os.path.join(output, 'graph'), format='pdf', rankdir="LR")

    result = result.compute(scheduler=client)
    for k in result:
        print(k.reference.subject, k.probe.subject, k.probe.path, k.data)

    client.shutdown()
