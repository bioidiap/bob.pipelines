.. _dask:

========================================
 Dask: Scale your scikit.learn pipelines
========================================


"`Dask is <https://dask.org/>`_ a flexible library for parallel computing in Python.".
The purpose of this guide is not to describe how dask works, for that, go to its documentation.
Moreover, there are plenty of tutorials online.
For instance, `this official one <https://github.com/dask/dask-tutorial>`_, there is also a nice overview presented in `AnacondaCon 2018 <https://www.youtube.com/watch?v=tQBovBvSDvA>`_ and there's even one crafted for `Idiap <https://github.com/tiagofrepereira2012/tam-dask>`_.

The purpose of this guide is to describe:
 
    1. The integration of dask with scikit learn pipelines and samples
    2. The specificities of `Dask` under the Idiap SGE


Dask + scikit learn pipelines
-----------------------------


An arbitrary scikit learn pipeline can be transformed in a `dask graph <https://docs.dask.org/en/latest/graphs.html>`_ to be further executed using the :py:class:`bob.pipelines.mixins.DaskEstimatorMixin` mixin.
This can be mixed with the :py:func:`bob.pipelines.mixins.estimator_dask_it` function.


This function does two things.

  1. Edit the in input pipeline adding a new first step, where input samples are wrapped in `Dask Bags <https://docs.dask.org/en/latest/bag.html>`_

  2. Create a Dask graph for each step in your pipeline


..todo ::
   Provide code sample


Dask + Idiap SGE
----------------


..todo ::
   Provide code sample

