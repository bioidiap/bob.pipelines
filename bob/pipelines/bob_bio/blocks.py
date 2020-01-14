#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


"""Re-usable blocks for legacy bob.bio.base algorithms"""


import os
import copy
import functools

import bob.io.base


def _copy_attributes(s, d):
    """Copies attributes from a dictionary to self
    """
    s.__dict__.update(
        dict(
            [k, v] for k, v in d.items() if k not in ("data", "load", "samples")
        )
    )


class DelayedSample:
    """Representation of sample that can be loaded via a callable

    The optional ``**kwargs`` argument allows you to attach more attributes to
    this sample instance.


    Parameters
    ----------

        load : function
            A python function that can be called parameterlessly, to load the
            sample in question from whatever medium

        parent : :py:class:`DelayedSample`, :py:class:`Sample`, None
            If passed, consider this as a parent of this sample, to copy
            information

        kwargs : dict
            Further attributes of this sample, to be stored and eventually
            transmitted to transformed versions of the sample

    """

    def __init__(self, load, parent=None, **kwargs):
        self.load = load
        if parent is not None:
            _copy_attributes(self, parent.__dict__)
        _copy_attributes(self, kwargs)

    @property
    def data(self):
        """Loads the data from the disk file"""
        return self.load()


class Sample:
    """Representation of sample that is sufficient for the blocks in this module

    Each sample must have the following attributes:

        * attribute ``data``: Contains the data for this sample


    Parameters
    ----------

        data : object
            Object representing the data to initialize this sample with.

        parent : object
            A parent object from which to inherit all other attributes (except
            ``data``)

    """

    def __init__(self, data, parent=None, **kwargs):
        self.data = data
        if parent is not None:
            _copy_attributes(self, parent.__dict__)
        _copy_attributes(self, kwargs)


class SampleSet:
    """A set of samples with extra attributes"""

    def __init__(self, samples, parent=None, **kwargs):
        self.samples = samples
        if parent is not None:
            _copy_attributes(self, parent.__dict__)
        _copy_attributes(self, kwargs)


class DatabaseConnector:
    """Wraps a bob.bio.base database and generates conforming samples

    This connector allows wrapping generic bob.bio.base datasets and generate
    samples that conform to the specifications of biometric pipelines defined
    in this package.


    Parameters
    ----------

    database : object
        An instantiated version of a bob.bio.base.Database object

    protocol : str
        The name of the protocol to generate samples from.
        To be plugged at :py:method:`bob.db.base.Database.objects`.

    """

    def __init__(self, database, protocol):
        self.database = database
        self.protocol = protocol
        self.directory = database.original_directory
        self.extension = database.original_extension

    def background_model_samples(self):
        """Returns :py:class:`Sample`'s to train a background model (group
        ``world``).


        Returns
        -------

            samples : list
                List of samples conforming the pipeline API for background
                model training.  See, e.g., :py:func:`.pipelines.first`.

        """

        # TODO: This should be organized by client
        retval = []

        objects = self.database.objects(protocol=self.protocol, groups="world")

        return [
            SampleSet(
                [
                    DelayedSample(
                        load=functools.partial(
                            k.load,
                            self.database.original_directory,
                            self.database.original_extension,
                        ),
                        id=k.id,
                        path=k.path,
                    )
                ]
            )
            for k in objects
        ]

    def references(self, group="dev"):
        """Returns :py:class:`Reference`'s to enroll biometric references


        Parameters
        ----------

            group : :py:class:`str`, optional
                A ``group`` to be plugged at
                :py:meth:`bob.db.base.Database.objects`


        Returns
        -------

            references : list
                List of samples conforming the pipeline API for the creation of
                biometric references.  See, e.g., :py:func:`.pipelines.first`.

        """

        retval = []

        for m in self.database.model_ids(protocol=self.protocol, groups=group):

            objects = self.database.objects(
                protocol=self.protocol,
                groups=group,
                model_ids=(m,),
                purposes="enroll",
            )

            retval.append(
                SampleSet(
                    [
                        DelayedSample(
                            load=functools.partial(
                                k.load,
                                self.database.original_directory,
                                self.database.original_extension,
                            ),
                            id=k.id,
                            path=k.path,
                        )
                        for k in objects
                    ],
                    id=m,
                    path=str(m),
                    subject=objects[0].client_id,
                )
            )

        return retval

    def probes(self, group):
        """Returns :py:class:`Probe`'s to score biometric references


        Parameters
        ----------

            group : str
                A ``group`` to be plugged at
                :py:meth:`bob.db.base.Database.objects`


        Returns
        -------

            probes : list
                List of samples conforming the pipeline API for the creation of
                biometric probes.  See, e.g., :py:func:`.pipelines.first`.

        """

        probes = dict()

        for m in self.database.model_ids(protocol=self.protocol, groups=group):

            # Getting all the probe objects from a particular biometric
            # reference
            objects = self.database.objects(
                protocol=self.protocol,
                groups=group,
                model_ids=(m,),
                purposes="probe",
            )

            # Creating probe samples
            for o in objects:
                if o.id not in probes:
                    probes[o.id] = SampleSet(
                        [
                            DelayedSample(
                                load=functools.partial(
                                    o.load,
                                    self.database.original_directory,
                                    self.database.original_extension,
                                ),
                                id=o.id,
                                path=o.path,
                            )
                        ],
                        id=o.id,
                        path=o.path,
                        subject=o.client_id,
                        references=[m],
                    )
                else:
                    probes[o.id].references.append(m)

        return list(probes.values())


class SampleLoader:
    """Adaptor for loading, preprocessing and feature extracting samples

    This adaptor class wraps around sample:

    .. code-block:: text

       [loading [-> preprocessing [-> extraction]]]

    The input sample object must obbey the following (minimal) API:

        * attribute ``id``: Contains an unique (string-fiable) identifier for
          processed samples
        * attribute ``data``: Contains the data for this sample

    Optional checkpointing is also implemented for each of the states,
    independently.  You may check-point just the preprocessing, feature
    extraction or both.


    Parameters
    ----------

    pipeline : :py:class:`list` of (:py:class:`str`, callable)
        A list of doubles in which the first entry are names of each processing
        step in the pipeline and second entry must be default-constructible
        :py:class:`bob.bio.base.preprocessor.Preprocessor` or
        :py:class:`bob.bio.base.preprocessor.Extractor` in any order.  Each
        of these objects must be a python type, that can be instantiated and
        used through its ``__call__()`` interface to process a single entry of
        a sample.  For python types that you may want to plug-in, but do not
        offer a default constructor that you like, pass the result of
        :py:func:`functools.partial` instead.

    """

    def __init__(self, pipeline):
        self.pipeline = copy.deepcopy(pipeline)

    def _handle_step(self, sset, func, checkpoint):
        """Handles a single step in the pipeline, with optional checkpointing

        Parameters
        ----------

        sset : SampleSet
            The original sample set to be processed (delayed or pre-loaded)

        func : callable
            The processing function to call for processing **each** sample in
            the set, if needs be

        checkpoint : str, None
            An optional string that may point to a directory that will be used
            for checkpointing the processing phase in question


        Returns
        -------

        r : SampleSet
            The prototype processed sample.  If no checkpointing required, this
            will be of type :py:class:`Sample`.  Otherwise, it will be a
            :py:class:`DelayedSample`

        """

        if checkpoint is not None:
            samples = []  # processed samples
            for s in sset.samples:
                # there can be a checkpoint for the data to be processed
                candidate = os.path.join(checkpoint, s.path + ".hdf5")
                if not os.path.exists(candidate):
                    # preprocessing is required, and checkpointing, do it now

                    # TODO: Do a decent check of the `annotations` keyword argument
                    # that is used in the preprocessor (with annotations) and extraction (no annotations)
                    try:
                        data = func(s.data, annotations=s.annotations)
                    except:
                        data = func(s.data)

                    # notice this can be called in parallel w/o failing
                    bob.io.base.create_directories_safe(
                        os.path.dirname(candidate)
                    )
                    # bob.bio.base standard interface for preprocessor
                    # has a read/write_data methods
                    writer = (
                        getattr(func, "write_data")
                        if hasattr(func, "write_data")
                        else getattr(func, "write_feature")
                    )
                    writer(data, candidate)

                # because we are checkpointing, we return a DelayedSample
                # instead of normal (preloaded) sample. This allows the next
                # phase to avoid loading it would it be unnecessary (e.g. next
                # phase is already check-pointed)
                reader = (
                    getattr(func, "read_data")
                    if hasattr(func, "read_data")
                    else getattr(func, "read_feature")
                )
                samples.append(
                    DelayedSample(
                        functools.partial(reader, candidate), parent=s
                    )
                )
        else:
            # if checkpointing is not required, load the data and preprocess it
            # as we would normally do
            samples = [Sample(func(s.data), parent=s) for s in sset.samples]

        r = SampleSet(samples, parent=sset)
        return r

    def _handle_sample(self, sset, pipeline):
        """Handles a single sampleset through a pipelien

        Parameters
        ----------

        sset : SampleSet
            The original sample set to be processed (delayed or pre-loaded)

        pipeline : :py:class:`list` of :py:class:`tuple`
            A list of tuples, each comprising of one processing function and
            one checkpoint directory (:py:class:`str` or ``None``, to avoid
            checkpointing that phase), respectively


        Returns
        -------

        r : Sample
            The processed sample

        """

        r = sset
        for func, checkpoint in pipeline:
            r = r if func is None else self._handle_step(r, func, checkpoint)
        return r

    def __call__(self, samples, checkpoints):
        """Applies the pipeline chaining with optional checkpointing

        Our implementation is optimized to minimize disk I/O to the most.  It
        yields :py:class:`DelayedSample`'s instead of :py:class:`Sample` if
        checkpointing is enabled.


        Parameters
        ----------

        samples : list
            List of :py:class:`SampleSet` to be treated by this pipeline

        checkpoints : dict
            A dictionary (with any number of entries) that may contain as many
            keys as those defined when you constructed this class with the
            pipeline tuple list.  Upon execution, the existance of an entry
            that defines checkpointing, this phase of the pipeline will be
            checkpointed.  Notice that you are in the control of checkpointing.
            If you miss an intermediary step, it will trigger this loader to
            load the relevant sample, even if the next phase is supposed to be
            checkpointed.  This strategy keeps the implementation as simple as
            possible.


        Returns
        -------

        samplesets : list
            Loaded samplesets, after optional preprocessing and extraction

        """

        pipe = [(v(), checkpoints.get(k)) for k, v in self.pipeline]
        return [self._handle_sample(k, pipe) for k in samples]


class AlgorithmAdaptor:
    """Describes a biometric model based on :py:class:`bob.bio.base.algorithm.Algorithm`'s

    The model can be fitted (optionally).  Otherwise, it can only execute
    biometric model enrollement, via ``enroll()`` and scoring, with
    ``score()``.

    Parameters
    ----------

        algorithm : object
            An object that can be initialized by default and posseses the
            following attributes and methods:

            * attribute ``requires_projector_training``: indicating if the
              model is fittable or not
            * method ``train_projector(samples, path)``: receives a list of
              objects produced by the equivalent ``Sample.data`` object, fed
              **after** sample loading by the equivalent pipeline, and records
              the model to an on-disk file
            * method ``load_projector(path)``: loads the model state from a file
            * method ``project(sample)``: projects the data to an embedding
              from a single sample
            * method ``enroll(samples)``: creates a scorable biometric
              reference from a set of input samples
            * method ``score(model, probe)``: scores a single probe, given the
              input model, which can be obtained by a simple
              ``project(sample)``

            If the algorithm cannot be initialized by default, pass the result
            of :py:func:`functools.partial` instead.

        path : string
            A path leading to a place where to save the fitted model or, in
            case this model is not fittable (``not is_fitable == False``), then
            name of the model to load for running prediction and scoring.

    """

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.extension = ".hdf5"

    def fit(self, samplesets, checkpoint):
        """Fits this model, if it is fittable

        Parameters
        ----------

            samplesets : list
                A list of :py:class:`SampleSet`s to be used for fitting this
                model

            checkpoint : str
                If provided, must the path leading to a location where this
                model should be saved at (complete path without extension) -
                currently, it needs to be provided because of existing
                serialization requirements (see bob/bob.io.base#106), but
                checkpointing will still work as expected.


        Returns
        -------

            model : str
                A path leading to the fitted model

        """

        self.path = checkpoint + self.extension
        if not os.path.exists(self.path):  # needs training
            model = self.algorithm()
            bob.io.base.create_directories_safe(os.path.dirname(self.path))
            if model.requires_projector_training:
                alldata = [
                    sample.data
                    for sampleset in samplesets
                    for sample in sampleset.samples
                ]
                model.train_projector(alldata, self.path)

        return self.path

    def enroll(self, references, path, checkpoint, *args, **kwargs):
        """Runs prediction on multiple input samples

        This method is optimized to deal with multiple reference biometric
        samples at once, organized in partitions


        Parameters
        ----------

            references : list
                A list of :py:class:`SampleSet` objects to be used for
                creating biometric references.  The sets must be identified
                with a unique id and a path, for eventual checkpointing.

            path : str
                Path pointing to stored model on disk

            checkpoint : str, None
                If passed and not ``None``, then it is considered to be the
                path of a directory containing possible cached values for each
                of the references in this experiment.  If that is the case, the
                values are loaded from there and not recomputed.

            *args, **kwargs :
                Extra parameters that can be used to hook-up processing graph
                dependencies, but are currently ignored

        Returns
        -------

            references : list
                A list of :py:class:`.samples.Reference` objects that can be
                used in scoring

        """

        class _CachedModel:
            def __init__(self, algorithm, path):
                self.model = algorithm()
                self.loaded = False
                self.path = path

            def load(self):
                if not self.loaded:
                    self.model.load_projector(self.path)
                    self.loaded = True

            def enroll(self, k):
                self.load()
                return self.model.enroll(
                    [self.model.project(s.data) for s in k.samples]
                )

            def write_enrolled(self, k, path):
                self.model.write_model(k, path)

        model = _CachedModel(self.algorithm, path)

        retval = []
        for k in references:
            if checkpoint is not None:
                candidate = os.path.join(
                    os.path.join(checkpoint, k.path + ".hdf5")
                )
                if not os.path.exists(candidate):
                    # create new checkpoint
                    bob.io.base.create_directories_safe(
                        os.path.dirname(candidate)
                    )
                    enrolled = model.enroll(k)
                    model.model.write_model(enrolled, candidate)
                retval.append(
                    DelayedSample(
                        functools.partial(model.model.read_model, candidate),
                        parent=k,
                    )
                )
            else:
                # compute on-the-fly
                retval.append(Sample(model.enroll(k), parent=k))
        return retval

    def score(self, probes, references, path, *args, **kwargs):
        """Scores a new sample against multiple (potential) references

        Parameters
        ----------

            probes : list
                A list of :py:class:`SampleSet` objects to be used for
                scoring the input references

            references : list
                A list of :py:class:`Sample` objects to be used for
                scoring the input probes, must have an ``id`` attribute that
                will be used to cross-reference which probes need to be scored.

            path : str
                Path pointing to stored model on disk

            *args, **kwargs :
                Extra parameters that can be used to hook-up processing graph
                dependencies, but are currently ignored


        Returns
        -------

            scores : list
                For each sample in a probe, returns as many scores as there are
                samples in the probe, together with the probe's and the
                relevant reference's subject identifiers.

        """

        model = self.algorithm()
        model.load_projector(path)

        retval = []
        for p in probes:
            data = [model.project(s.data) for s in p.samples]
            for subprobe_id, (s, parent) in enumerate(zip(data, p.samples)):
                # each sub-probe in the probe needs to be checked
                subprobe_scores = []
                for ref in [r for r in references if r.id in p.references]:
                    subprobe_scores.append(
                        Sample(model.score(ref.data, s), parent=ref)
                    )
                subprobe = SampleSet(subprobe_scores, parent=p)
                subprobe.subprobe_id = subprobe_id
                retval.append(subprobe)
        return retval