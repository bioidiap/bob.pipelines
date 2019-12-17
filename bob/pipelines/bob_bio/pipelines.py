#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
Biometric "blocks"

This file contains simple processing blocks meant to be used
for bob.bio experiments
"""

import os


def first(
    background_model_samples,
    references,
    probes,
    loader,
    algorithm,
    npartitions,
    cache,
    ):
    """Creates a simple pipeline for **biometric** experiments.

    This contains the following steps:

      1. Train background model (without labels)
      2. Create biometric references (requires identity)
      3. Scoring (requires probe/reference matching and probe identity)


    Parameters
    ----------

    background_model_samples : list
        List of samples to be used for training an background model.  Elements
        provided must conform to the :py:class:`.samples.Sample` API, or be a
        delayed version of such.

    references : list
        List of references to be created in this biometric pipeline.  Elements
        provided must conform to the :py:class:`.samples.Reference` API, or be
        a delayed version of such.

    probes : list
        List of probes to be scored in this biometric pipeline.  Elements
        provided must conform to the :py:class:`.samples.Probe` API, or be
        a delayed version of such.

    loader : object
        An object that conforms to the :py:class:`.blocks.SampleLoader` API

    algorithm : object
        An object that conforms to the :py:class:`.blocks.AlgorithmAdaptor` API

    npartitions : :py:class:`int`, optional
        Number of partitions to use when processing this pipeline.  Notice that
        the number of partitions dictate how many preprocessor/feature
        extraction/algorithms objects will be effectively initialized (that is,
        will have their constructor called).  Internally, we use
        :py:class:`dask.bag`'s and :py:meth:`dask.bag.map_partitions` to
        process one full partition in a single pass.

    cache : :py:class:`str`, optional
        A path that points to location that is shared by all workers in the
        client, and is used to potentially cache partial outputs of this
        pipeline


    Returns
    -------

      scores: list
        A delayed list of scores, that can be obtained by computing the graph

    """

    import dask.bag
    import dask.delayed

    ## Training background model (fit will return even if samples is ``None``,
    ## in which case we suppose the algorithm is not trainable in any way)
    db = dask.bag.from_sequence(background_model_samples, npartitions=npartitions)
    db = db.map_partitions(loader)
    background_model = dask.delayed(algorithm.fit)(db)

    ## Enroll biometric references
    db = dask.bag.from_sequence(references, npartitions=npartitions)
    db = db.map_partitions(loader)
    references = db.map_partitions(algorithm.enroll, background_model)

    # TODO: This phase is optional, it caches the models
    # N.B.: This step will return the precomputed models
    #models_path = os.path.join(experiment_path, "models")
    #def _store(data, path):
    #    from bob.io.base import save
    #    save(data[1], os.path.join(path, '%s.hdf5' % data[0]),
    #            create_directories=True)
    #    return data
    #models = models.map(_store, models_path)

    ## Scores all probes
    db = dask.bag.from_sequence(probes, npartitions=npartitions)
    db = db.map_partitions(loader)

    ## TODO: Here, we are sending all computed biometric references to all
    ## probes.  It would be more efficient if only the models related to each
    ## probe are sent to the probing split.  An option would be to use caching
    ## and allow the ``score`` function above to load the required data from
    ## the disk, directly.  A second option would be to generate named delays
    ## for each model and then associate them here.
    all_references = dask.delayed(list)(references)
    return db.map_partitions(algorithm.score, all_references, background_model)
