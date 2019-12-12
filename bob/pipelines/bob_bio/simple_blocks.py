#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
Biometric "blocks"

This file contains simple processing blocks meant to be used
for bob.bio experiments
"""


def process_bobbio_samples(biometric_samples,
                           processor):
    """
    Process :py:obj:`Sample` that wraps :py:obj:`bob.db.base.File`
    
    """

    for b in biometric_samples:

        for f in b.data:
               
            # Preprocessing
            data = f.sample
            processed = processor(data)

            # Injecting the sample in bio_file
            f.sample = processed

    return biometric_samples


def train_bobbio_algorithm(biometric_samples, algorithm, output_model):

    import bob.io.base
    from bob.pipelines.samples.biometric_samples import read_biofiles

    # Concatenating list of objects
    training_data = read_biofiles(biometric_samples, bob.io.base.load).astype("float64")
   
    # TODO: bob.bio.base#106
    algorithm.train_projector(training_data, output_model)
  
    return output_model
    
    
def project_bobalgorithm(biometric_samples, algorithm, background_model):

    # Loading backgroundmodel
    algorithm.load_projector(background_model)

    for f in biometric_samples:

        for s in f.samples:

            if s.sample is None:
                input_file = s.make_path(s.current_directory, s.current_extension)
                s.sample   = read_bobbiodata(input_file) 

            data = s.sample
            projected = algorithm.project(data)
            s.sample = projected

    return biometric_samples


def create_bobbio_templates(biometric_samples, algorithm):

    for f in biometric_samples:
        template_data = read_biofiles([f], bob.io.base.load)
        f.samples = algorithm.enroll(template_data)

    return biometric_samples    
