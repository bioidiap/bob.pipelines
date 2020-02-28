#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import copy
import logging
import numpy
logger = logging.getLogger(__name__)
import os
import bob.io.base
from bob.pipelines.sample import (
    Sample, SampleSet, DelayedSample
    )
from bob.pipelines.utils import is_picklable
import functools


class ProcessorBlock(object):
    """Base unit for processing any kind of data.

    This base processor follows the same API from scikit learn (https://scikit-learn.org/stable/glossary.html#glossary).
    
    Basically an arbitrary processors can be optionally **fittable** and mandatorily be able to **transform** data.
    
    For example, think about a PCA model (https://en.wikipedia.org/wiki/Principal_component_analysis).
    In the method `fit`, the PCA matrix should be trained and in the method `transform` the dot production between the PCA matrix
    and an arbitrary input data should be computed

    Parameters
    ----------

    """
    def __init__(self, is_fittable=False, **kwargs):
        self.fitted = False
        self.is_fittable = is_fittable
        self.model_name = "model.pickle"
        super(ProcessorBlock, self).__init__(**kwargs)


    def fit(self, X, y=None, **kwargs):
        """
        Train (or fit) a processor

        Parameters
        ----------

          X: array-like
            Data used for training an arbitrary model

          y: array-like (optional)
            Possible labels for training an arbitrary model
        """

        self.fitted = True


    def transform(self, X, **kwargs):
        """
        Transform `X`
        
        Parameters
        ----------

          X: array-like
            Data used for training an arbitrary model

        Returns
        -------
          X_new: array-like
            X trainsformed

        """

        pass


    def fit_transform(self, X, y=None, **kwargs):
        """
        Runs fit and transform given an input data
        
        Parameters
        ----------

          X: array-like
            Data used for training an arbitrary model

          y: array-like (optional)
            Possible labels for training an arbitrary model

        Returns
        -------
          X_new: array-like
            X trainsformed

        """
        if self.is_fittable:
            self.fit(X, y, **kwargs)
        return self.transform(X, **kwargs)


    def write(self, X, filename, default_extension=".hdf5"):
        """
        Saves itself into disk

        Parameters
        ----------
          X: array-like
            Input data

          path: str
            Path to save the model
        """
        import h5py

        with h5py.File(filename+default_extension, "w") as f:
            f.create_dataset("array", data=X)

    
    def read(self, filename, default_extension=".hdf5"):
        """
        Read model from disk

        Parameters
        ----------

          path: str
            Path to save the model
        """
        import h5py

        with h5py.File(filename+default_extension, "r") as f:
            data = f["array"].value
        return data


    def write_model(self, path):
        """
        Writes itself in disk

        Parameters
        ----------

          path:
             Base path to save the model

        """

        import pickle
        filename = os.path.join(path, self.model_name)
        logger.info(f"Saving processor in in {filename}")
        open(filename, "wb").write(pickle.dumps(self))


    def load_model(self, filename):
        """
        Loads itself in disk

        Parameters
        ----------

          path:
             Base path to save the model

        """
        logger.info(f"Loading processor in in {filename}")
        import pickle        
        return pickle.loads(open(filename, "rb").read())


class ProcessorPipeline(object):
    """Implement a pipeline that load samples and process a stack of them **sequentially**

    This class wraps around sample:

    .. code-block:: text

       [loading [-> processor_1 [-> processor_n+1]]]

    The input sample object must obbey the following (minimal) API:

        * attribute ``id``: Contains an unique (string-fiable) identifier for
          processed samples
        * attribute ``data``: Contains the data for this sample

    Optional checkpointing is also implemented for each of the states,
    independently.  You may check-point just the preprocessing, feature
    extraction or both.


    Parameters
    ----------

    pipeline : :py:class:`list` of (:py:class:`str`, callable)
        A list of doubles in which the first entry are names of each processing
        step in the pipeline and second entry must be default-constructible
        :py:class:`ProcessorBlock` Each of these objects must be a python type, that can be instantiated and
        used through its ``__transform__()`` interface to process a single entry of
        a sample.  For python types that you may want to plug-in, but do not
        offer a default constructor that you like, pass the result of
        :py:func:`functools.partial` instead.

    """

    def __init__(self, pipeline):
        self.pipeline = copy.deepcopy(pipeline)


    def _stack_samples_2_ndarray(self, samplesets, stack_per_sampleset=False):
        """
        Stack a set of :py:class:`bob.pipelines.sample.sample.SampleSet`
        and convert them to :py:class:`numpy.ndarray`

        Parameters
        ----------

            samplesets: :py:class:`bob.pipelines.sample.sample.SampleSet`
                         Set of samples to be stackted

            stack_per_sampleset: bool
                If true will return a list of :py:class:`numpy.ndarray`, each one for a sample set

        """

        if stack_per_sampleset:
            # TODO: Make it more efficient
            all_data = []
            for sampleset in samplesets:
                all_data.append(
                    numpy.vstack([sample.data for sample in sampleset.samples])
                )
            return all_data
        else:
            return numpy.vstack(
                [
                    sample.data
                    for sampleset in samplesets
                    for sample in sampleset.samples
                ]
            )


    def _handle_step(self, sset, func, checkpoint):
        """Handles a single step in the pipeline, with optional checkpointing

        Parameters
        ----------

        sset : SampleSet
            The original sample set to be processed (delayed or pre-loaded)

        func : callable
            The processing function to call for processing **each** sample in
            the set, if needs be

        checkpoint : str, None
            An optional string that may point to a directory that will be used
            for checkpointing the processing phase in question


        Returns
        -------

        r : SampleSet
            The prototype processed sample.  If no checkpointing required, this
            will be of type :py:class:`Sample`.  Otherwise, it will be a
            :py:class:`DelayedSample`

        """

        if checkpoint is not None:            
            samples = []  # processed samples
            for s in sset.samples:
                # there can be a checkpoint for the data to be processed
                candidate = os.path.join(checkpoint, s.path)
                if not os.path.exists(candidate):
                    # preprocessing is required, and checkpointing, do it now
                    data = func.transform(s.data)

                    # notice this can be called in parallel w/o failing
                    bob.io.base.create_directories_safe(os.path.dirname(candidate))
                    # bob.bio.base standard interface for preprocessor
                    # has a read/write_data methods
                    writer = getattr(func, "write")
                    writer(data, candidate)

                # because we are checkpointing, we return a DelayedSample
                # instead of normal (preloaded) sample. This allows the next
                # phase to avoid loading it would it be unnecessary (e.g. next
                # phase is already check-pointed)
                reader = getattr(func, "read")

                if is_picklable(reader):
                    samples.append(
                        DelayedSample(
                            functools.partial(reader, candidate), parent=s
                        )
                    )
                else:                    
                    logger.warning(f"The method {func} is not picklable. Shiping its unbounded method to `DelayedSample`.")
                    reader = reader.__func__ # The reader object might not be picklable

                    samples.append(
                        DelayedSample(
                            functools.partial(reader, None, candidate), parent=s
                        )
                    )
        else:
            # if checkpointing is not required, load the data and preprocess it
            # as we would normally do
            samples = [Sample(func.transform(s.data), parent=s) for s in sset.samples]

        r = SampleSet(samples, parent=sset)
        return r


    def _handle_sample(self, sset, pipeline):
        """Handles a single sampleset through a pipelien

        Parameters
        ----------

        sset : SampleSet
            The original sample set to be processed (delayed or pre-loaded)

        pipeline : :py:class:`list` of :py:class:`tuple`
            A list of tuples, each comprising of one processing function and
            one checkpoint directory (:py:class:`str` or ``None``, to avoid
            checkpointing that phase), respectively


        Returns
        -------

        r : Sample
            The processed sample

        """

        r = sset
        for func, checkpoint in pipeline:
            r = r if func is None else self._handle_step(r, func, checkpoint)
        return r

    def fit(self, samples, checkpoints=None):
        """Trains Applies the pipeline chaining with optional checkpointing

        Our implementation is optimized to minimize disk I/O to the most.  It
        yields :py:class:`DelayedSample`'s instead of :py:class:`Sample` if
        checkpointing is enabled.


        Parameters
        ----------

        samples : list
            List of :py:class:`SampleSet` to be treated by this pipeline

        checkpoints : dict
            A dictionary (with any number of entries) that may contain as many
            keys as those defined when you constructed this class with the
            pipeline tuple list.  Upon execution, the existance of an entry
            that defines checkpointing, this phase of the pipeline will be
            checkpointed.  Notice that you are in the control of checkpointing.
            If you miss an intermediary step, it will trigger this loader to
            load the relevant sample, even if the next phase is supposed to be
            checkpointed.  This strategy keeps the implementation as simple as
            possible.

        Returns
        -------

        samplesets : list
            List containing the fitted pipelines

        """

        logger.info("Fitting pipeline")

        X = self._stack_samples_2_ndarray(samples)

        _ = [v.fit_transform(X) for k, v in self.pipeline[:-1]] # Fitting+transform everyone but the last
        self.pipeline[-1][1].fit(X) # Fit the last

        # Checkpointing the pipeline        
        if checkpoints is not None:
            for k, v in self.pipeline:
                logger.info(f"  >> Saving {k}")
                if v.is_fittable:
                    path = checkpoints.get(k)
                    if path is None:
                        logger.info(f"  >> No checkpoint provided for {k}")
                    else:
                        bob.io.base.create_directories_safe(path)
                        v.write_model(path)

        return self.pipeline


    def transform(self, samples, checkpoints={}):
        """Trains Applies the pipeline chaining with optional checkpointing

        Our implementation is optimized to minimize disk I/O to the most.  It
        yields :py:class:`DelayedSample`'s instead of :py:class:`Sample` if
        checkpointing is enabled.


        Parameters
        ----------

        samples : list
            List of :py:class:`SampleSet` to be treated by this pipeline

        checkpoints : dict
            A dictionary (with any number of entries) that may contain as many
            keys as those defined when you constructed this class with the
            pipeline tuple list.  Upon execution, the existance of an entry
            that defines checkpointing, this phase of the pipeline will be
            checkpointed.  Notice that you are in the control of checkpointing.
            If you miss an intermediary step, it will trigger this loader to
            load the relevant sample, even if the next phase is supposed to be
            checkpointed.  This strategy keeps the implementation as simple as
            possible.


        Returns
        -------

        samplesets : list
            Loaded samplesets, after optional preprocessing and extraction

        """
        import inspect
        
        pipe = []        
        logger.info("Transforming pipeline")
        for k,v in self.pipeline:
            logger.info(f"  >> Transforming {k}")
            transformer = v() if inspect.isclass(v) or isinstance(v, functools.partial) else v
            if transformer.is_fittable and not transformer.fitted:                
                transformer = transformer.load_model(os.path.join(checkpoints.get(k), v.model_name))
            pipe.append((transformer, checkpoints.get(k)))

        return [self._handle_sample(k, pipe) for k in samples]
