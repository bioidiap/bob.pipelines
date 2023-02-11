.. _bob.pipelines.csv_database:

File List Databases (CSV)
=========================

We saw in :ref:`bob.pipelines.sample` that how using samples can improve the
workflow of our machine learning experiments. However, we did not discuss how to
create the samples in the first place.

In all reproducible machine learning experiments, each database comes with one
or several protocols that define exactly which files should be used for
training, development, and testing. These protocols can be defined in ``.csv``
files where each row represents a sample. Using ``.csv`` files to define the
protocols of a database is advantageous because the files are easy to create and
read. And, they can be imported and used in many different libraries.

Here, we provide :any:`bob.pipelines.FileListDatabase` that can be used to read
``.csv`` files and generate :py:class:`bob.pipelines.Sample`. The format is extremely
simple. You must put all the protocol files in a folder with the following
structure::

    dataset_protocols_path/<protocol>/<group>.csv

where each subfolder points to a specific *protocol* and each file contains the
samples of a specific *group* or *set* (e.g. training set). The names of the
protocols are the names of folders and the name of each group is the name of the
file.

.. note::

    Instead of pointing to a folder, you can also point to a compressed tarball
    that contains the protocol files.

The ``.csv`` files must have the following structure::

    attribute_1,attribute_2,...,attribute_n
    sample_1_attribute_1,sample_1_attribute_2,...,sample_1_attribute_n
    sample_2_attribute_1,sample_2_attribute_2,...,sample_2_attribute_n
    ...
    sample_n_attribute_1,sample_n_attribute_2,...,sample_n_attribute_n

Each row will contain exactly **one** sample (e.g. one image) and
each column will represent one attribute of samples (e.g. path to data or other
metadata).

An Example
----------

Below is an example of creating the iris database. The ``.csv`` files are
distributed with this package have the following format::

    iris_database/
        default/
            train.csv
            test.csv

As you can see there is only one protocol called ``default`` and two groups
``train`` and ``test``. Moreover, ``.csv`` files have the following format::

    sepal_length,sepal_width,petal_length,petal_width,target
    5.1,3.5,1.4,0.2,Iris-setosa
    4.9,3,1.4,0.2,Iris-setosa
    ...

.. doctest:: csv_iris_database

    >>> import bob.pipelines
    >>> dataset_protocols_path = "tests/data/iris_database"
    >>> database = bob.pipelines.FileListDatabase(
    ...     name="iris",
    ...     protocol="default",
    ...     dataset_protocols_path=dataset_protocols_path,
    ... )
    >>> database.samples(groups="train")
    [Sample(data=None, sepal_length='5.1', sepal_width='3.5', petal_length='1.4', petal_width='0.2', target='Iris-setosa'), Sample(...)]
    >>> database.samples(groups="test")
    [Sample(data=None, sepal_length='5', sepal_width='3', petal_length='1.6', petal_width='0.2', target='Iris-setosa'), Sample(...)]

As you can see, all attributes are strings. Furthermore, we may want to
*transform* our samples further before using them.

Transforming Samples
--------------------

:any:`bob.pipelines.FileListDatabase` accepts a transformer that will be applied
to all samples:

.. doctest:: csv_iris_database

    >>> import numpy as np
    >>> from sklearn.preprocessing import FunctionTransformer

    >>> def prepare_data(sample):
    ...     return np.array(
    ...         [sample.sepal_length, sample.sepal_width,
    ...          sample.petal_length, sample.petal_width],
    ...         dtype=float
    ...     )

    >>> def prepare_iris_samples(samples):
    ...     return [bob.pipelines.Sample(prepare_data(sample), parent=sample) for sample in samples]

    >>> database = bob.pipelines.FileListDatabase(
    ...     name="iris",
    ...     protocol="default",
    ...     dataset_protocols_path=dataset_protocols_path,
    ...     transformer=FunctionTransformer(prepare_iris_samples),
    ... )
    >>> database.samples(groups="train")
    [Sample(data=array([5.1, 3.5, 1.4, 0.2]), sepal_length='5.1', sepal_width='3.5', petal_length='1.4', petal_width='0.2', target='Iris-setosa'), Sample(...)]

.. note::

    The ``transformer`` used in the ``FileListDatabase`` will not be fitted and
    you should not perform any computationally heavy processing on the samples
    in this transformer. You are expected to do the minimal processing of
    samples here to make them ready for experiments. Most of the time you just
    load the data from disk in this transformer and return delayed samples.

Now our samples are ready to be used and we can run a simple experiment with
them.

Running An Experiment
---------------------

Here, we want to train a Linear Discriminant Analysis (LDA) on the data. Before
that, we want to normalize the range of our data and convert the ``target``
labels to integers.

.. doctest:: csv_iris_database

    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    >>> from sklearn.preprocessing import StandardScaler, LabelEncoder
    >>> from sklearn.pipeline import Pipeline
    >>> scaler = StandardScaler()
    >>> encoder = LabelEncoder()
    >>> lda = LinearDiscriminantAnalysis()

    >>> scaler = bob.pipelines.wrap(["sample"], scaler)
    >>> encoder = bob.pipelines.wrap(["sample"], encoder, input_attribute="target", output_attribute="y")
    >>> lda = bob.pipelines.wrap(["sample"], lda, fit_extra_arguments=[("y", "y")])

    >>> pipeline = Pipeline([('scaler', scaler), ('encoder', encoder), ('lda', lda)])
    >>> pipeline.fit(database.samples(groups="train"))
    Pipeline(...)
    >>> encoder.estimator.classes_
    array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']...)
    >>> predictions = pipeline.predict(database.samples(groups="test"))
    >>> predictions[0].data, predictions[0].target, predictions[0].y
    (0, 'Iris-setosa', 0)
