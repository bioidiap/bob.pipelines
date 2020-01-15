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
    """Run a pipeline

    FROM THE TIME BEING NOT PASSING ANY PARAMETER

    \b

    bob pipelines run


    """

    # Configures the algorithm (concrete) implementations for the pipeline
    # from ..bob_bio.annotated_blocks import DatabaseConnectorAnnotated as DatabaseConnector

    import bob.bio.face
    bob_db = bob.bio.face.database.MobioBioDatabase(
           original_directory="/idiap/resource/database/mobio/IMAGES_PNG",
           original_extension=".png",
           annotation_directory="/idiap/resource/database/mobio/IMAGE_ANNOTATIONS")



    from ..bob_bio.annotated_blocks import DatabaseConnectorAnnotated as DatabaseConnector
    database = DatabaseConnector(bob_db, protocol="mobile0-male")

    from ..bob_bio.blocks import SampleLoader

    import bob.bio.base
    import bob.bio.face

    # Using face crop
    CROPPED_IMAGE_HEIGHT = 80
    CROPPED_IMAGE_WIDTH = CROPPED_IMAGE_HEIGHT * 4 // 5

    ## eye positions for frontal images
    RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 - 1)
    LEFT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 * 3)

    # Defines the processing pipeline for loading samples
    # Can add any number of steps!
    pipeline = [
        (
            "preprocessor",
            functools.partial(
                bob.bio.face.preprocessor.FaceCrop,
                cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
                cropped_positions={'leye': LEFT_EYE_POS, 'reye': RIGHT_EYE_POS},

            ),
        ),
        ("extractor", bob.bio.base.extractor.Linearize),
    ]


    loader = SampleLoader(pipeline)



    from ..bob_bio.blocks import AlgorithmAdaptor
    from bob.bio.base.algorithm import PCA

    algorithm = AlgorithmAdaptor(functools.partial(PCA, 0.99))

    # Configures the execution context
    #from bob.pipelines.distributed.local import debug_client
    #client = debug_client(1)

    from bob.pipelines.distributed.sge import sge_iobig_client
    client = sge_iobig_client(15)

    # Chooses the pipeline to run
    from bob.pipelines.bob_bio.pipelines import biometric_pipeline

    if not os.path.exists(output):
        os.makedirs(output)

    checkpoints = {
        "background": {
            "preprocessor": os.path.join(output, "background", "preprocessed"),
            "extractor": os.path.join(output, "background", "extracted"),
            # at least, the next stage must be provided!
            "model": os.path.join(output, "background", "model"),
        },
        "references": {
            "preprocessor": os.path.join(output, "references", "preprocessed"),
            "extractor": os.path.join(output, "references", "extracted"),
            "enrolled": os.path.join(output, "references", "enrolled"),
        },
        "probes": {
            "preprocessor": os.path.join(output, "probes", "preprocessed"),
            "extractor": os.path.join(output, "probes", "extracted"),
        },
    }
    result = biometric_pipeline(
        database.background_model_samples(),
        database.references(group="dev"),
        database.probes(group="dev"),
        loader,
        algorithm,
        npartitions=len(client.cluster.workers),
        checkpoints=checkpoints,
    )

    # result.visualize(os.path.join(output, "graph.pdf"), rankdir="LR")

    result = result.compute(scheduler=client)
    for probe in result:
        for reference in probe.samples:
            print(reference.subject, probe.subject, probe.path, reference.data)

    client.shutdown()
