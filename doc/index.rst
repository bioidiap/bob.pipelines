.. -*- coding: utf-8 -*-

.. _bob.pipelines:

===============
 Bob Pipelines
===============

Easily boost your :any:`sklearn.pipeline.Pipeline` with powerful features, such as:

* Scaling experiments on dask_.
* Wrapping data-points with metadata and passing them to the `estimator.fit` and `estimator.transform` methods.
* Checkpointing data-points after each step of your pipeline.
* Expressing database protocol as csv files and using them easily.


.. warning::

    Before any investigation of this package is capable of, check the scikit
    learn :ref:`user guide <scikit-learn:pipeline>`. Several :ref:`tutorials
    <scikit-learn:tutorial_menu>` are available online.

.. warning::

    If you want to implement your own scikit-learn estimator, please, check out
    this :doc:`link <scikit-learn:developers/develop>`

User Guide
==========

.. toctree::
   :maxdepth: 2

   sample
   checkpoint
   dask
   datasets
   xarray
   py_api

.. include:: links.rst
