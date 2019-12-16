#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
Biometric "blocks"

This file contains simple processing blocks meant to be used
for bob.bio experiments
"""

import os
from .simple_blocks import process_bobbio_samples, train_bobbio_algorithm
from bob.pipelines.samples.biometric_samples import cache_bobbio_samples, cache_bobbio_algorithms


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
    algorithm,
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
    algorithm,
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

    extracted_training_futures = []
    extracted_enroll_futures = []
    extracted_probe_futures = []

    # DECORATING FOR CACHING

    output_path = os.path.join(experiment_path, "./extracted")
    decorated_extractor = cache_bobbio_samples(output_path, ".hdf5")(
        process_bobbio_samples
    )

    for t_o, e_o, p_o in zip(
        preproc_training_futures, preproc_enroll_futures, preproc_probe_futures
    ):
        extracted_training_futures.append(
            dask.delayed(decorated_extractor)(
                biometric_samples=t_o, processor=extractor
            )
        )

        extracted_enroll_futures.append(
            dask.delayed(decorated_extractor)(
                biometric_samples=e_o, processor=extractor
            )
        )

        extracted_probe_futures.append(
            dask.delayed(decorated_extractor)(
                biometric_samples=p_o, processor=extractor
            )
        )


    # TRAINING

    output_model = os.path.join(experiment_path, "Project.hdf5")


    #concatenating delayed for training
    extracted_training_concat = dask.bag.concat(extracted_training_futures)

    #background_model_future = cache_bobbio_algorithms(output_model)

    background_model_future = dask.delayed(train_bobbio_algorithm)(
                                          extracted_training_concat, algorithm, output_model
                                          )

    background_model_future.compute(scheduler=client)



    # PROJECT



    #background_model.result()


    # Dumping futures
    #for t_o, e_o, p_o in zip(
    #    extractor_training_futures, extractor_enroll_futures, extractor_probe_futures
    #):
    #    t_o.compute(scheduler=client)
    #    e_o.compute(scheduler=client)
    #    p_o.compute(scheduler=client)



def pipeline_ANDRE(
    training_samples,
    reference_samples,
    probing_samples,
    preprocessor,
    extractor,
    algorithm,
    client,
    experiment_path="my_experiment",
    ):
    """Creates a simple pipeline for **biometric** experiments.

    This contains the following steps:

      1. Train background model (without labels)
      2. Create biometric references (requires identity)
      3. Scoring (requires probe/reference matching and probe identity)

    What is assumed:

      1. Inputs are bob.bio.base.File objects, that know how to load and save
         themselves to disk
      2. The ``experiment_path`` is a path lying on a shared file system
      3. The ``algorithm`` follows the bob.bio.base.Algorithm API


    Parameters
    ----------

    training_samples : list
        List of biometric samples to be used for training the background model

    reference_samples : :py:class:`list` of :py:class:`tuple`
        List of biometric references to be created.  Each element should be
        double in which the first entry is the **unique** identifier of the
        reference, and the second, the list of samples related to this
        biometric reference.

    probing_samples : :py:class:`list` of :py:class:`tuple`
        List of biometric probing samples to probe.  Each element should be
        triple in which the first entry is the **unique** identifier of the
        probe, and the second, a list of samples related to this
        biometric probe and, the third, a list of unique identifiers that
        relate to the identity of each reference sample produced

    preprocessor : callable
        A callable to preprocess each biometric sample.  We assume it inputs a
        :py:class:`numpy.ndarray` and outputs another one.

    extractor : callable
        A callable to process each biometric sample, after it has been
        preprocessed.  We assume it inputs a :py:class:`numpy.ndarray` and
        outputs another one.

    algorithm : object
        An object that has a method called ``train_projector``, that inputs the
        stacked "extracted" samples and saves a machine on disk

    client : object
        A dask client that can be used to compute things

    experiment_path : :py:class:`str`, optional
        A path that points to location that is shared by all workers in the
        client
    """

    import dask.bag
    import dask.array
    import dask.delayed

    ## SPLITING THE TASK IN N NODES
    n_nodes = len(client.cluster.workers)

    ## 1. TRAINING BACKGROUND MODEL:
    ##    a. Preprocess and Extract all samples
    ##    b. Feed all samples to the trainer
    splits = dask.bag.from_sequence(training_samples, npartitions=n_nodes)
    splits = splits.map(lambda x: x.load())
    splits = splits.map(lambda x: preprocessor(x))
    splits = splits.map(lambda x: extractor(x))

    ## TODO: temporarily serializing extracted features?  Tip: Check how
    ## dask.bag.to_textfiles or dask.bag.to_avro is implemented, copy for HDF5
    ## caching?  If you do so, it would be better to prefix the initial map()
    ## above with a file checking to avoid recomputing the features and
    ## re-storing them.

    ## TODO: this is related to bob/bob.bio.base#106.  The PCA projector will
    ## be trained from the training data above, it will get stored to disk.
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    background_model_path = os.path.join(experiment_path, "Project.hdf5")

    ## TODO: model is not really returned due to the way train_projector is
    ## encoded
    ## TODO: does passing the splits like this implies in delayed execution
    ## of the splits as well?
    background_model = dask.delayed(algorithm.train_projector)(
            splits, background_model_path)

    ## 2. CREATE BIOMETRIC REFERENCES:
    ##    a. Preprocess and Extract all samples related to all references
    ##    b. Feed all samples to the trained projector and get models
    splits = dask.bag.from_sequence(reference_samples, npartitions=n_nodes)
    splits = splits.map(lambda x: (x[0], [k.load() for k in x[1]]))
    splits = splits.map(lambda x: (x[0], [preprocessor(k) for k in x[1]]))
    splits = splits.map(lambda x: (x[0], [extractor(k) for k in x[1]]))

    def _enroll(data, background, background_path):
        algorithm.load_projector(background_path)
        ## TODO: shall we split the partial projection as well?
        feats = [algorithm.project(k) for k in data[1]]
        return (data[0], algorithm.enroll(feats))

    models = splits.map(_enroll, background_model, background_model_path)

    models_path = os.path.join(experiment_path, "models")
    def _store(data, path):
        from bob.io.base import save
        save(data[1], os.path.join(path, '%s.hdf5' % data[0]),
                create_directories=True)
        return data

    # N.B.: This step will return the precomputed models
    models = models.map(_store, models_path)

    ## 3. SCORING:
    ##    a. Preprocess and Extract all samples related to all probes
    ##    b. Feed all samples to the trained projector and get projections
    ##    c. Compare each projection to the required models

    ## Runs the graph using the client
    models.compute(scheduler=client)
