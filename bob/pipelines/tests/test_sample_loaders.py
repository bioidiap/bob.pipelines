import os

import numpy as np
import pkg_resources

from sklearn.pipeline import make_pipeline

import bob.io.base
import bob.io.image

from bob.pipelines.sample_loaders import AnnotationsLoader
from bob.pipelines.sample_loaders import CSVToSampleLoader


def test_sample_loader():
    path = pkg_resources.resource_filename(__name__, os.path.join("data", "samples"))

    sample_loader = CSVToSampleLoader(
        data_loader=bob.io.base.load, dataset_original_directory=path, extension=".pgm"
    )

    f = open(os.path.join(path, "samples.csv"))

    samples = sample_loader.transform(f)
    assert len(samples) == 2
    assert np.alltrue([s.data.shape == (112, 92) for s in samples])


def test_annotations_loader():
    path = pkg_resources.resource_filename(__name__, os.path.join("data", "samples"))

    csv_sample_loader = CSVToSampleLoader(
        data_loader=bob.io.base.load, dataset_original_directory=path, extension=".pgm"
    )
    annotation_loader = AnnotationsLoader(
        annotation_directory=path,
        annotation_extension=".pos",
        annotation_type="eyecenter",
    )

    sample_loader = make_pipeline(csv_sample_loader, annotation_loader)

    f = open(os.path.join(path, "samples.csv"))

    samples = sample_loader.transform(f)
    assert len(samples) == 2
    assert np.alltrue([s.data.shape == (112, 92) for s in samples])
    assert np.alltrue([isinstance(s.annotations, dict) for s in samples])
