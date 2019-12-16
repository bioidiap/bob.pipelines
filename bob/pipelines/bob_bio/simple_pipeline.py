#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
Biometric "blocks"

This file contains simple processing blocks meant to be used
for bob.bio experiments
"""

import os

def pipeline(
    training_samples,
    reference_samples,
    probing_samples,
    preprocessor,
    extractor,
    algorithm,
    npartitions=1,
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
        probe, the second, a list of samples related to this biometric probe
        and, the third, a list of unique identifiers that relate to the
        identity of each reference sample to each this probe should be matched

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

    npartitions : :py:class:`int`, optional
        Number of partitions to use when processing this pipeline

    experiment_path : :py:class:`str`, optional
        A path that points to location that is shared by all workers in the
        client


    Returns
    -------

      delayeds: list
        List of :py:class:`dask.delayed` composing this pipeline.

    """

    import dask.bag
    import dask.delayed

    ## 1. TRAINING BACKGROUND MODEL:
    ##    a. Preprocess and Extract all samples
    ##    b. Feed all samples to the trainer
    npartititions = 1
    db = dask.bag.from_sequence(training_samples, npartitions=npartitions)
    db = db.map_partitions(lambda x: [k.load() for k in x])
    db = db.map_partitions(lambda x: [preprocessor(k) for k in x])
    db = db.map_partitions(lambda x: [extractor(k) for k in x])

    ## TODO: temporarily serializing extracted features?  Tip: Check how
    ## dask.bag.to_textfiles or dask.bag.to_avro is implemented, copy for HDF5
    ## caching?  If you do so, it would be better to prefix the initial map()
    ## above with a file checking to avoid recomputing the features and
    ## re-storing them.

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    background_model_path = os.path.join(experiment_path, "Project.hdf5")

    ## TODO: model is not really returned due to the way train_projector is
    ## encoded. This is related to bob/bob.bio.base#106.  The PCA projector
    ## will be trained from the training data above, it will get stored to
    ## disk.
    ##
    ## CHECKED: does passing the db like this implies in delayed execution
    ## of the db as well? YES, it does work as expected acording to Tiago
    if algorithm.requires_projector_training:
        background_model = dask.delayed(algorithm.train_projector)(db,
                background_model_path)
    else:
        background_model = \
                dask.delayed(algorithm.load_projector)(background_model)

    ## 2. CREATE BIOMETRIC REFERENCES:
    ##    a. Preprocess and Extract all samples related to all references
    ##    b. Feed all samples to the trained projector and get models
    db = dask.bag.from_sequence(reference_samples, npartitions=npartitions)
    db = db.map_partitions(lambda y: [(x[0], [k.load() for k in x[1]]) for x in y])
    db = db.map_partitions(lambda y: [(x[0], [preprocessor(k) for k in x[1]]) for x in y])
    db = db.map_partitions(lambda y: [(x[0], [extractor(k) for k in x[1]]) for x in y])

    def _enroll(data, background, background_path):
        algorithm.load_projector(background_path)
        ## TODO: shall we split the partial projection as well?
        feats = [algorithm.project(k) for k in data[1]]
        return (data[0], algorithm.enroll(feats))

    models = db.map(_enroll, background_model, background_model_path)

    # TODO: This phase is optional, it caches the models
    # N.B.: This step will return the precomputed models
    models_path = os.path.join(experiment_path, "models")
    def _store(data, path):
        from bob.io.base import save
        save(data[1], os.path.join(path, '%s.hdf5' % data[0]),
                create_directories=True)
        return data
    models = models.map(_store, models_path)

    ## 3. SCORING:
    ##    a. Preprocess and Extract all samples related to all probes
    ##    b. Feed all samples to the trained projector and get projections
    ##    c. Compare each projection to the required models
    db = dask.bag.from_sequence(probing_samples, npartitions=npartitions)
    db = db.map_partitions(lambda y: [(x[0], [k.load() for k in x[1]], x[2]) for x in y])
    db = db.map_partitions(lambda y: [(x[0], [preprocessor(k) for k in x[1]], x[2]) for x in y])
    db = db.map_partitions(lambda y: [(x[0], [extractor(k) for k in x[1]], x[2]) for x in y])

    def _score(data, background, background_path, models):
        algorithm.load_projector(background_path)
        ## TODO: shall we split the partial projection as well?
        feats = [algorithm.project(k) for k in data[1]]
        ## scores the probe against each model, for as long as
        ## we should produce a score for each
        retval = []
        for model in [m for m in models if m[0] in data[2]]:
            for n, sample in enumerate(feats):
                retval.append((model[0], n, algorithm.score(model[1], sample)))
        return (data[0], retval)

    ## TODO: Here, we are sending all computed models to all probes.  It would
    ## be more efficient if only the models related to each probe are sent to
    ## the probing split.  An option would be to use caching and allow the
    ## ``_score`` function above to load the required data from the disk,
    ## directly.  A second option would be to generate named delays for each
    ## model and then associate them here.
    ## TODO: Did not find a way to pass a "bag" as a parameter to the
    ## ``.map()`` function of **another** bag
    #import ipdb; ipdb.set_trace()
    all_models = dask.delayed(list)(models)
    scores = db.map(_score, background_model, background_model_path,
            all_models)


    return scores
