#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


"""Re-usable blocks for legacy bob.bio.base algorithms"""


import os
import copy

from .samples import Sample, Reference, Probe, Score
from .blocks import DatabaseConnector
from .blocks import SampleLoader


class DatabaseConnectorAnnotated(DatabaseConnector):
    """Wraps a bob.bio.base database and generates conforming samples

    This connector allows wrapping generic bob.bio.base datasets and generate
    samples that conform to the specifications of biometric pipelines defined
    in this package.
    
    ..Note: This connector supports annotations from bob.bio.database


    Parameters
    ----------

    database : object
        An instantiated version of a bob.bio.base.Database object

    protocol : str
        The name of the protocol to generate samples from.
        To be plugged at :py:method:`bob.db.base.Database.objects`.

    """

    def __init__(self, database, protocol):
        super(DatabaseConnectorAnnotated, self).__init__(database, protocol)

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
                annotations=self.database.annotations(k),
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

        for m in self.database.model_ids_with_protocol(protocol=self.protocol, groups=group):

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
                            annotations=self.database.annotations(k)
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

        for m in self.database.model_ids_with_protocol(protocol=self.protocol, groups=group):

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
                                annotations=self.database.annotations(o)
                            )
                        ],
                        o.client_id,
                        o.id,
                        [m],
                    )
                else:
                    probes[o.id].references.append(m)

        return list(probes.values())


class SampleLoaderAnnotated(SampleLoader):
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

 
    ..Note: This is supposed to handle databases with annotations


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
        super(SampleLoaderAnnotated, self).__init__(preprocessor_type, extractor_type)

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
            
            # Dumping the annotationsi
            if isinstance(s, Sample):
                annotations = [s.annotations]
            else:            
                annotations = [o.annotations for o in s.samples]    
            for d,a in zip(s.load(), annotations):
                if preprocessor is not None:
                    d = preprocessor(d, annotations=a)
                if extractor is not None:
                    d = extractor(d)
                r.data.append(d)
            loaded.append(r)
        return loaded

