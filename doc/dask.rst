.. _dask:

========================================
 Dask: Scale your scikit.learn pipelines
========================================


"`Dask is <dask:index>`_ a flexible library for parallel computing in Python.".
The purpose of this guide is not to describe how dask works.
For that, go to its documentation.
Moreover, there are plenty of tutorials online.
For instance, `this official one <https://github.com/dask/dask-tutorial>`_; a nice overview was presented in `AnacondaCon 2018 <https://www.youtube.com/watch?v=tQBovBvSDvA>`_ and there's even one crafted for `Idiap <https://github.com/tiagofrepereira2012/tam-dask>`_.

The purpose of this guide is to describe:
 
    1. The integration of dask with scikit learn pipelines and samples
    2. The specificities of `Dask` under the Idiap SGE


From Scikit Learn pipelines to Dask Task Graphs
-----------------------------------------------

The purpose of :doc:`scikit learn pipelines <modules/generated/sklearn.pipeline.Pipeline>` is to assemble several :doc:`scikit estimators <modules/generated/sklearn.base.BaseEstimator>` in one final one.
Then, it is possible to use the methods `fit` and `transform` to create models and transform your data respectivelly.

Any :doc:`pipeline <modules/generated/sklearn.pipeline.Pipeline>` can be transformed in a :doc:`Dask Graph <graphs>` to be further executed by any :doc:`Dask Client <client>`.
This is carried out via the :py:func:`bob.pipelines.mixins.estimator_dask_it` function.
Such fuction does two things:

   1. Edit the current :py:class:`sklearn.pipeline.Pipeline` by adding a new first step, where input samples are transformed in :doc:`Dask Bag <bag>`. This allows the usage of :py:func:`dask.bag.map` for further transformations.

   2. Mix all :doc:`estimators <modules/generated/sklearn.base.BaseEstimator>` in the pipeline with the :py:class:`bob.pipelines.mixins.DaskEstimatorMixin`. Such mixin is reponsible for the creation of the task graph for the methods `.fit` and `.transform`.


The code snippet below enables such feature for an arbitrary :doc:`pipeline <modules/generated/sklearn.pipeline.Pipeline>`.


.. code:: python

   >>> from bob.pipelines.mixins import estimator_dask_it
   >>> dask_pipeline = estimator_dask_it(make_pipeline(...)) # Create a dask graph
   >>> dask_pipeline.fit_transform(....).compute() # Run the task graph using the default client


The code below is the same as the one presented in :ref:`checkpoint example <checkpoint_statefull>`.
However, lines 59-63 convert such pipeline in a :doc:`Dask Graph <graphs>` and runs it in a local computer.


.. literalinclude:: ./python/pipeline_example_dask.py
   :linenos:
   :emphasize-lines: 59-63


Such code generates the following graph.


.. figure:: python/dask_graph.png

   This graph can be seem by running `http://localhost:8787` during its execution.


Dask + Idiap SGE
----------------

Dask, allows the deployment and parallelization of graphs either locally or in complex job queuing systems, such as PBS, SGE....
This is achieved via :doc:`Dask-Jobqueue <dask-jobqueue:index>`.
Below follow a nice video explaining what is the :doc:`Dask-Jobqueue <dask-jobqueue:index>`, some of its features and how to use it to run :doc:`dask graphs <graphs>`.

 .. raw:: html
     
     <iframe width="560" height="315" src="https://www.youtube.com/embed/FXsgmwpRExM" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
     

The snippet below shows how to deploy the exact same pipeline from the previous section in the Idiap SGE cluster


.. code:: python

   >>> from bob.pipelines.distributed.sge import SGEIdiapCluster
   >>> from dask.distributed import Client
   >>> cluster = SGEIdiapCluster() # Creates the SGE launcher that launches jobs in the q_all
   >>> cluster.scale(10) # Submite 10 jobs in q_all
   >>> client = Client(cluster) # Creates the scheduler and attaching it to the SGE job queue system
   >>> dask_pipeline.fit_transform(....).compute(scheduler=client) # Runs my graph in the Idiap SGE


That's it, you just run a scikit pipeline in the Idiap SGE grid :-)

Dask provides generic :doc:`deployment <dask-jobqueue:examples>` mechanism for SGE systems, but it contains the following limitations:

  1. It assumes that a :doc:`dask graph <dask:graphs>` runs in an homogeneous grid setup. For instance, if parts your graph needs a specific resource that it's avaible in other SGE queues (e.g q_gpu, q_long_gpu, IO_BIG), the scheduler is not able to request those resources on the fly.

  2. As a result of 1., the mechanism of :doc:`adaptive deployment <dask:setup/adaptive>` is not able to handle job submissions of two or more queues.

For this reason the generic SGE laucher was extended to this one :py:class:`bob.pipelines.distributed.sge.SGEIdiapCluster`. Next subsections presents some code samples using this launcher in the most common cases you will probably find in your daily job.   


Launching jobs in different SGE queues
======================================

SGE queue specs are defined in python dictionary as in the example below, where, the root keys are the labels of the SGE queue and the other inner keys represents:

   1. **queue**: The real name of the SGE queue
   2. **memory**: The amount of memory required for the job
   3. **io_big**: Submit jobs with IO_BIG=TRUE
   4. **resource_spec**: Whatever other key using in `qsub -l`
   5. **resources**: Reference label used to tag :doc:`dask delayed <dask:delayed>` so it will run in a specific queue. This is a very important feature the will be discussed in the next section.

.. code:: python

    >>> Q_1DAY_GPU_SPEC = {
    ...         "default": {
    ...             "queue": "q_1day",
    ...             "memory": "8GB",
    ...             "io_big": True,
    ...             "resource_spec": "",
    ...             "resources": "",
    ...         },
    ...         "gpu": {
    ...             "queue": "q_gpu",
    ...             "memory": "12GB",
    ...             "io_big": False,
    ...             "resource_spec": "",
    ...             "resources": {"GPU":1},
    ...         },
    ...     }


Now that the queue specifications are set, let's trigger some jobs.

.. code:: python
   
   >>> from bob.pipelines.distributed.sge import SGEIdiapCluster
   >>> from dask.distributed import Client
   >>> cluster = SGEIdiapCluster(sge_job_spec=Q_1DAY_GPU_SPEC)
   >>> cluster.scale(1, sge_job_spec_key="gpu") # Submitting 1 job in the q_gpu queue
   >>> cluster.scale(10) # Submitting 9 jobs in the q_1day queue
   >>> client = Client(cluster) # Creating the scheduler

.. note::

    To check if the jobs were actually submitted always do `qstat`::

    $ qstat



Running estimator operations in specific SGE queues 
===================================================

Sometimes it's necessary to run parts of a :doc:`pipeline <modules/generated/sklearn.pipeline.Pipeline>`  in specific SGE queues (e.g. q_1day IO_BIG or q_gpu).
The example below shows how this is approached (lines 78 to 88).
In this example, the `fit` method of `MyBoostedFitTransformer` runs on `q_gpu`


.. literalinclude:: ./python/pipeline_example_dask_sge.py
   :linenos:
   :emphasize-lines: 78-88



Adaptive SGE: Scale up/down SGE cluster according to the graph complexity
=========================================================================

One note about the code from the last section.
Every time` cluster.scale` is executed to increase the amount of available SGE jobs to run a :doc:`dask graph <graphs>`, such resources will be available until the end of its execution.
Note that in `MyBoostedFitTransformer.fit` a delay of `120s`was introduced to fake "processing" in the GPU queue.
During the execution of `MyBoostedFitTransformer.fit` in `q_gpu`, other resources are idle, which is a waste of resources (imagined a CNN training of 2 days instead of the 2 minutes from our example).

For this reason there's the method adapt in :py:class:`bob.pipelines.distributed.sge.SGEIdiapCluster` that will adjust the SGE jobs available according to the needs of a :doc:`dask graph <graphs>`.

Its usage is pretty simple.
The code below determines that to run a :doc:`dask graph <graphs>`, the :py:class`distributed.scheduler.Scheduler` can demand a maximum of 10 SGE jobs. A lower bound was also set, in this case, two SGE jobs.


.. code:: python

   >>> cluster.adapt(minimum=2, maximum=10)


The code below shows the same example, but with adaptive cluster.
Note line 83

.. literalinclude:: ./python/pipeline_example_dask_sge_adaptive.py
   :linenos:
   :emphasize-lines: 83

