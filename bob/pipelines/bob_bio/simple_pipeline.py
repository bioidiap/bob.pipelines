#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
Biometric "blocks"

This file contains simple processing blocks meant to be used
for bob.bio experiments
"""

import os
from .simple_blocks import process_bobbio_samples
from bob.pipelines.samples.biometric_samples import cache_bobbio_samples


def split_data(input_data, n_groups):
    """
    Given a list of elements, split them in a particular number of groups
    This is useful for paralellization
    """

    offset = 0
    # Number of elements per groups
    step = len(input_data) // n_groups

    output_data = []
    for i in range(n_groups):
        output_data.append(input_data[offset : offset + step])
        offset += step

    return output_data


def pipeline(
    training_samples,
    biometric_reference_samples,
    probe_samples,
    preprocessor,
    extractor,
    client,
    experiment_path="my_experiment",
):
    """
    Create a simple pipeline for BIOMETRIC experiments.

    This contains the following steps:

      1. Preprocessing
      2. Extraction
      3. Train background model
      4. Project
      5. Create biometric references
      6. Scoring

    For the time being it's assumed that a training and development set.
    Follow a schematic of the pipeline


    training_data --> 1. preproc - 2. extract 3. train background ---
                                                                    |
    enroll data -->   1. preproc - 2. extract                       |-- 4. project - 5. create biometric references                                           |                | 
                                                                    i|                |
    probe data --> 1. preproc - 2. extract                          |-- 4. project - 6. compute scores

    
    """

    ## SPLITING THE TASK IN N NODES
    n_nodes = len(client.cluster.workers)
    training_split = split_data(training_samples, n_nodes)
    biometric_reference_split = split_data(biometric_reference_samples, n_nodes)
    probe_split = split_data(probe_samples, n_nodes)

    # 1. PREPROCESSING

    preproc_training_futures = []
    preproc_enroll_futures = []
    preproc_probe_futures = []

    # DECORATING FOR CACHING
    output_path = os.path.join(experiment_path, "./preprocessed")
    # decorated_preprocess = cache_bobbio_samples(output_path, ".hdf5")(process_bobbio_samples)
    decorated_preprocess = process_bobbio_samples  # I'VE CHOSEN NOT TO CACHE THIS ONE

    for t_o, e_o, p_o in zip(training_split, biometric_reference_split, probe_split):
        preproc_training_futures.append(
            client.submit(
                decorated_preprocess, biometric_samples=t_o, processor=preprocessor
            )
        )

        preproc_enroll_futures.append(
            client.submit(
                decorated_preprocess, biometric_samples=e_o, processor=preprocessor
            )
        )

        preproc_probe_futures.append(
            client.submit(
                decorated_preprocess, biometric_samples=p_o, processor=preprocessor
            )
        )

    # 2. EXTRACTION

    extractor_training_futures = []
    extractor_enroll_futures = []
    extractor_probe_futures = []

    # DECORATING FOR CACHING

    output_path = os.path.join(experiment_path, "./extracted")
    decorated_extractor = cache_bobbio_samples(output_path, ".hdf5")(
        process_bobbio_samples
    )

    for t_o, e_o, p_o in zip(
        preproc_training_futures, preproc_enroll_futures, preproc_probe_futures
    ):
        extractor_training_futures.append(
            client.submit(
                decorated_extractor, biometric_samples=t_o, processor=extractor
            )
        )

        extractor_enroll_futures.append(
            client.submit(
                decorated_extractor, biometric_samples=e_o, processor=extractor
            )
        )

        extractor_probe_futures.append(
            client.submit(
                decorated_extractor, biometric_samples=p_o, processor=extractor
            )
        )

    # Dumping futures
    for t_o, e_o, p_o in zip(
        extractor_training_futures, extractor_enroll_futures, extractor_probe_futures
    ):
        t_o.result()
        e_o.result()
        p_o.result()


def pipeline_DELAY(
    training_samples,
    biometric_reference_samples,
    probe_samples,
    preprocessor,
    extractor,
    client,
    experiment_path="my_experiment",
):
    """
    Create a simple pipeline for BIOMETRIC experiments.

    This contains the following steps:

      1. Preprocessing
      2. Extraction
      3. Train background model
      4. Project
      5. Create biometric references
      6. Scoring

    For the time being it's assumed that a training and development set.
    Follow a schematic of the pipeline


    training_data --> 1. preproc - 2. extract 3. train background ---
                                                                    |
    enroll data -->   1. preproc - 2. extract                       |-- 4. project - 5. create biometric references                                           |                | 
                                                                    i|                |
    probe data --> 1. preproc - 2. extract                          |-- 4. project - 6. compute scores

    
    """

    import dask.delayed
    import dask.bag

    ## SPLITING THE TASK IN N NODES
    n_nodes = len(client.cluster.workers)

    # training_split = split_data(training_samples, n_nodes)
    # biometric_reference_split = split_data(biometric_reference_samples, n_nodes)
    # probe_split = isplit_data(probe_samples, n_nodes)

    training_split = dask.bag.from_sequence(
        training_samples, npartitions=n_nodes
    ).to_delayed()

    biometric_reference_split = dask.bag.from_sequence(
        biometric_reference_samples, npartitions=n_nodes
    ).to_delayed()

    probe_split = dask.bag.from_sequence(
        probe_samples, npartitions=n_nodes
    ).to_delayed()

    # 1. PREPROCESSING

    preproc_training_futures = []
    preproc_enroll_futures = []
    preproc_probe_futures = []

    # DECORATING FOR CACHING
    output_path = os.path.join(experiment_path, "./preprocessed")
    # decorated_preprocess = cache_bobbio_samples(output_path, ".hdf5")(process_bobbio_samples)

    decorated_preprocess = process_bobbio_samples  # I'VE CHOSEN NOT TO CACHE THIS ONE

    for t_o, e_o, p_o in zip(training_split, biometric_reference_split, probe_split):
        preproc_training_futures.append(
            dask.delayed(decorated_preprocess)(
                biometric_samples=t_o, processor=preprocessor
            )
        )

        preproc_enroll_futures.append(
            dask.delayed(decorated_preprocess)(
                biometric_samples=e_o, processor=preprocessor
            )
        )

        preproc_probe_futures.append(
            dask.delayed(decorated_preprocess)(
                biometric_samples=p_o, processor=preprocessor
            )
        )

    # 2. EXTRACTION

    extractor_training_futures = []
    extractor_enroll_futures = []
    extractor_probe_futures = []

    # DECORATING FOR CACHING

    output_path = os.path.join(experiment_path, "./extracted")
    decorated_extractor = cache_bobbio_samples(output_path, ".hdf5")(
        process_bobbio_samples
    )

    for t_o, e_o, p_o in zip(
        preproc_training_futures, preproc_enroll_futures, preproc_probe_futures
    ):
        extractor_training_futures.append(
            dask.delayed(decorated_extractor)(
                biometric_samples=t_o, processor=extractor
            )
        )

        extractor_enroll_futures.append(
            dask.delayed(decorated_extractor)(
                biometric_samples=e_o, processor=extractor
            )
        )

        extractor_probe_futures.append(
            dask.delayed(decorated_extractor)(
                biometric_samples=p_o, processor=extractor
            )
        )

    # Dumping futures
    for t_o, e_o, p_o in zip(
        extractor_training_futures, extractor_enroll_futures, extractor_probe_futures
    ):
        t_o.compute(scheduler=client)
        e_o.compute(scheduler=client)
        p_o.compute(scheduler=client)
