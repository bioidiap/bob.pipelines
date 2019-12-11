#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


from .samples import Sample


class ProbeSample(Sample):
    """
    Representation of a Probe Sample.

    In ML tasks very often it's necessary to compare one sample with a set of samples (or its combination).
    This task is called probing and this class aims to represent this operation.


    Parameters:

      sample_id:
        Sample ID

      data:
         Data representation

      biometric_references:
        List of :py:obj:`Sample` to be probed with
    """

    def __init__(self, sample_id, data, biometric_references):

        self.biometric_references = biometric_references
        super(ProbeSample, self).__init__(sample_id, data)


def create_training_samples(database, group="dev", protocol="Default"):
    """
     Generate :py:obj:`Samples` to **train** background models specific for biometric Pipelines using Bob.

     This function create :py:obj:`Sample` from  :py:class:`bob.db.base.Database` objects.

     Parameters:

       database:
         An instance of :py:class:`bob.db.base.Database`

       group:
         A `group` to be set in :py:method:`bob.db.base.Database.objects`

       protocol:
         A `protocol to be set in :py:method:`bob.db.base.Database.objects`

     Return:
        This returs a list of :py:obj:`Sample`

   """

    # TODO: This should be organized by client
    biometric_samples = []
    objects = database.objects(protocol=protocol, groups="world")

    for o in objects:

        # TODO: Implement the following 3 properties in bob.db.base
        o.current_directory = database.original_directory
        o.current_extension = database.original_extension
        o.sample = None

        biometric_sample = Sample(o.client_id, [o])
        biometric_samples.append(biometric_sample)

    return biometric_samples


def create_biometric_reference_samples(database, group="dev", protocol="Default"):
    """
     Generate :py:obj:`Samples` to create **biometric references (a.k.a templates)** in pipelines using Bob 

      This function create :py:obj:`Sample` from  :py:class:`bob.db.base.Database` objects

     Parameters:

       database:
         An instance of :py:class:`bob.db.base.Database`

       group:
         A `group` to be set in :py:method:`bob.db.base.Database.objects`

       protocol:
         A `protocol to be set in :py:method:`bob.db.base.Database.objects`

     Return:
        This returs a list of :py:obj:`Sample`
    """

    biometric_references = []

    # TODO: MISSING PROTOCOL
    model_ids = database.model_ids(groups=group)

    for m in model_ids:
        objects = database.objects(
            protocol=protocol, groups=group, model_ids=(m,), purposes="enroll"
        )

        for o in objects:
            # TODO: Implement the following 3 properties in bob.db.base
            o.current_directory = database.original_directory
            o.current_extension = database.original_extension
            o.sample = None

        biometric_reference = Sample(m, objects)
        biometric_references.append(biometric_reference)

    return biometric_references


def create_biometric_probe_samples(
    database, biometric_references, group="dev", protocol="Default"
):
    """
     Generate :py:obj:`Samples` to **probe** biometric references (a.k.a templates) in pipelines using Bob 

      This function create :py:obj:`Sample` from  :py:class:`bob.db.base.Database` objects

     Parameters:

       database:
         An instance of :py:class:`bob.db.base.Database`

       biometric_references:
         A list containing :py:obj:`Samples` that represents biometric references

       group:
         A `group` to be set in :py:method:`bob.db.base.Database.objects`

       protocol:
         A `protocol to be set in :py:method:`bob.db.base.Database.objects`

     Return:
        This returs a list of :py:obj:`Sample`
    """

    probes = dict()

    # Fetching all the biometric_references
    for br in biometric_references:

        # Getting all the probe objects from a particular biometric
        # reference
        objects = database.objects(
            protocol=protocol,
            groups=group,
            model_ids=(br.sample_id,),
            purposes="probe",
        )

        # Creating probe samples
        for o in objects:
            if o.id not in probes:

                o.current_directory = database.original_directory
                o.current_extension = database.original_extension
                o.sample = None

                probes[o.id] = ProbeSample(o.client_id, [o], [])

            probes[o.id].biometric_references.append(br)

    # Returning a list of probes
    probes_list = [probes[k] for k in probes]

    return probes_list


def cache_bobbio_samples(output_path, output_extension):
    """
    Decorator meant to be used to cache biometric samples

    **Parameters**

      output_path:

      output_extension:
    """

    def decorator(func):
        def wrapper(*args, **kwargs):

            #Loading from cache
            



            biometric_samples = func(*args, **kwargs)

 
            # Caching
            for bs in biometric_samples:

                for o in bs.data:

                    # Setting the current location of the biofile
                    o.current_directory = output_path
                    o.current_extension = output_extension

                    # Saving
                    file_name = o.make_path(o.current_directory, o.current_extension)
                    
                    # If it is already cached, don't do anything
                    if os.path.exists(file_name):
                        continue

                    # Save
                    bob.io.base.create_directories_safe(os.path.dirname(file_name))
                    write_bobbiodata(o.sample, file_name)
                    #o.sample = None


            return biometric_samples
             
        return wrapper
    return decorator


def read_biofiles(objects, loader, split_by_client=False, allow_missing_files=False):
    """read_features(file_names, extractor, split_by_client = False) -> extracted

  Reads the extracted features from ``file_names`` using the given ``extractor``.
  If ``split_by_client`` is set to ``True``, it is assumed that the ``file_names`` are already sorted by client.

  **Parameters:**

  file_names : [str] or [[str]]
    A list of names of files to be read.
    If ``split_by_client = True``, file names are supposed to be split into groups.

  extractor : py:class:`bob.bio.base.extractor.Extractor` or derived
    The extractor, used for reading the extracted features.

  split_by_client : bool
    Indicates if the given ``file_names`` are split into groups.

  allow_missing_files : bool
    If set to ``True``, extracted files that are not found are silently ignored.

  **Returns:**

  extracted : [object] or [[object]]
    The list of extracted features, in the same order as in the ``file_names``.
  """
    # file_names = utils.filter_missing_files(file_names, split_by_client, allow_missing_files)

    if split_by_client:
        return [
            [extractor.read_feature(f) for f in client_files]
            for client_files in file_names
        ]
    else:
        return numpy.vstack(
            [
                loader(s.make_path(s.current_directory, s.current_extension)).astype(
                    "float64"
                )
                if s.sample is None
                else s.sample
                for o in objects
                for s in o.samples
            ]
        )
