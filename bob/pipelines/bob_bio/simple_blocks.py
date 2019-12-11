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

