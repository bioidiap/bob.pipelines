.. -*- coding: utf-8 -*-

.. _bob.pipelines:

===============
 Bob Pipelines
===============

Easilly boost your `Scikit Learn Pipelines <https://scikit-learn.org/stable/index.html>`_ with powerfull features, such as:

 


.. figure:: img/dask.png
    :width: 40%
    :align: center

    Scale them with Dask

.. figure:: img/metadata.png
   :width: 40%
   :align: center

   Wrap datapoints with metadata and pass them to the `estimator.fit` and `estimator.transform` methods
    
.. figure:: img/checkpoint.png
   :width: 40%
   :align: center

   Checkpoint datapoints after each step of your pipeline


.. warning::
    Before any investigation of this package is capable of, check the scikit learn `user guide <https://scikit-learn.org/stable/modules/compose.html#pipeline>`_. Several `tutorials <https://scikit-learn.org/stable/tutorial/index.html>`_ are available online.

.. warning::
    If you want to implement your own scikit-learn estimator, please, check it out this `link <https://scikit-learn.org/stable/developers/develop.html>`_



User Guide
==========

.. toctree::
   :maxdepth: 2

   sample
   checkpoint
   dask
   py_api

