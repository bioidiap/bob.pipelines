#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


"""Re-usable blocks for legacy bob.bio.base algorithms"""


import os
import copy

from .samples import Sample, Reference, Probe, Score


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
            Sample(
                None,
                k.path,
                self.database.original_directory,
                self.database.original_extension,
                # these are optional
                subject=k.client_id,
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
                Reference(
                    None,
                    str(m),
                    [
                        Sample(
                            None,
                            k.path,
                            self.database.original_directory,
                            self.database.original_extension,
                        )
                        for k in objects
                    ],
                    objects[0].client_id,
                    m,
                )
            )

        return retval

    def probes(self, group="dev"):
        """Returns :py:class:`Probe`'s to score biometric references


        Parameters
        ----------

            group : :py:class:`str`, optional
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
                    probes[o.id] = Probe(
                        None,
                        o.path,
                        [
                            Sample(
                                None,
                                o.path,
                                self.database.original_directory,
                                self.database.original_extension,
                            )
                        ],
                        o.client_id,
                        o.id,
                        [m],
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

        * method ``load()``: Loads the data for this sample, which should be an
          iterable of :py:class:`numpy.ndarray`.
        * attribute ``data``: Contains the data for this sample.  This field
          may be set to ``None`` upon initialization.  It is used internally to
          store and transmit pre-loaded and transformed data between different
          processing stages.  It is an iterable which may contain elements of
          different nature than those returned by ``load()``, but respect the
          same ordering.  E.g., the first entry of ``data`` corresponds to a
          transformed version of the first array returned by ``load()``

    The sample is loaded if its ``data`` attribute is ``None``, by using its
    ``load()`` method.  After that, it is preprocessed, if the
    ``preprocessor_type`` is not ``None``.  Then, feature extraction follows if
    the ``extractor_type`` is not ``None``.


    Parameters
    ----------

    preprocessor_type : type
        A python type, that can be instantiated and used through its
        ``__call__()`` interface to preprocess a single entry of a sample.  If
        not set, then does not apply any preprocessing to the sample after
        loading it. If not set (or set to ``None``), then does not apply any
        feature extraction to the sample after preprocessing it.  For python
        types that you may want to plug-in, but do not offer a default
        constructor that you like, pass the result of
        :py:func:`functools.partial` instead.

    extractor_type : type
        A python type, that can be instantiated and used through its
        ``__call__()`` interface to extract features from a single entry of a
        sample.  If not set (or set to ``None``), then does not apply any
        feature extraction to the sample after preprocessing it.  For python
        types that you may want to plug-in, but do not offer a default
        constructor that you like, pass the result of
        :py:func:`functools.partial` instead.

    """

    def __init__(self, preprocessor_type=None, extractor_type=None):
        self.preprocessor_type = preprocessor_type
        self.extractor_type = extractor_type

    def __call__(self, samples):
        """Applies the chain load() -> preproc() -> extract() to a list of samples


        Parameters
        ----------

        samples : list
            A list of samples that should be treated


        Returns
        -------

        samples : list
            Prepared samples

        """

        # the preprocessor and extractor are initialized once
        preprocessor = (
            self.preprocessor_type()
            if self.preprocessor_type is not None
            else None
        )
        extractor = (
            self.extractor_type() if self.extractor_type is not None else None
        )

        loaded = []
        for s in samples:
            r = copy.copy(s)
            r.data = []
            for d in s.load():
                if preprocessor is not None:
                    d = preprocessor(d)
                if extractor is not None:
                    d = extractor(d)
                r.data.append(d)
            loaded.append(r)
        return loaded


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

    def __init__(self, algorithm, path):
        self.algorithm = algorithm
        self.path = path

    def fit(self, samples):
        """Fits this model, if it is fittable

        Parameters
        ----------

            samples : list
                A list of :py:class:`.samples.Sample` objects to be used for
                fitting this model


        Returns
        -------

            model : str
                A path leading to the fitted model

        """

        ## TODO: model is not really returned due to the way train_projector is
        ## encoded. This is related to bob/bob.bio.base#106.  The PCA projector
        ## will be trained from the training data above, it will get stored to
        ## disk.

        def _flatten(l):
            return [item for sublist in l for item in sublist]

        model = self.algorithm()
        dirname = os.path.dirname(self.path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        if model.requires_projector_training:
            model.train_projector(
                _flatten([k.load() for k in samples]), self.path
            )
        return self.path

    def enroll(self, references, *args, **kwargs):
        """Runs prediction on multiple input samples

        This method is optimized to deal with multiple reference biometric
        samples at once, organized in partitions


        Parameters
        ----------

            references : list
                A list of :py:class:`.samples.Reference` objects to be used for
                creating biometric references

            *args, **kwargs :
                Extra parameters that can be used to hook-up processing graph
                dependencies, but are currently ignored

        Returns
        -------

            references : list
                A list of :py:class:`.samples.Reference` objects that can be
                used in scoring

        """

        model = self.algorithm()
        model.load_projector(self.path)
        retval = []
        for k in references:
            r = copy.copy(k)
            r.data = model.enroll([model.project(l) for l in k.load()])
            retval.append(r)
        return retval

    def score(self, probes, references, *args, **kwargs):
        """Scores a new sample against multiple (potential) references

        Parameters
        ----------

            probes : list
                A list of :py:class:`.samples.Probe` objects to be used for
                scoring the input references

            references : list
                A list of :py:class:`.samples.Reference` objects to be used for
                scoring the input probes

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
        model.load_projector(self.path)

        retval = []
        for p in probes:
            data = [model.project(l) for l in p.load()]
            for ref in [r for r in references if r.id in p.references]:
                for s in data:
                    retval.append(
                        Score(
                            probe=copy.copy(p),
                            reference=copy.copy(ref),
                            data=model.score(ref.data, s),
                        )
                    )
                    # we are not interested on the original probe and reference
                    # data in this score, so we just supress it to avoid
                    # trafficking too much information
                    retval[-1].probe.data = None
                    retval[-1].probe.references = []
                    retval[-1].reference.data = None
        return retval
