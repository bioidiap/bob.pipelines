#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


class Sample(object):
    """
    Representation of sample.
    
    A sample object is composed by an identifier and a representation of data
    This representation of data can be any kind of serializable object.
        
    Parameters:

      sample_id:
        Sample ID

      data:
         Data representation
    """

    def __init__(self, sample_id, data):

        self.sample_id = sample_id
        self.data = data
