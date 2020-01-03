#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


"""Re-usable blocks for legacy bob.bio.base algorithms"""


import os
import copy
import functools

import bob.io.base


from .blocks import DatabaseConnector, SampleSet, DelayedSample, Sample, SampleLoader


class DatabaseConnectorAnnotated(DatabaseConnector):
    """Wraps a bob.bio.base database and generates conforming samples for datasets
    that has annotations

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
                        annotations=self.database.annotations(k)
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

        for m in self.database.model_ids_with_protocol(protocol=self.protocol, groups=group):

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
                            annotations=self.database.annotations(k)
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
                                annotations=self.database.annotations(o)
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

