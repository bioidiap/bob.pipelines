from ..samples.biometric_samples import (
    create_training_samples,
    create_biometric_reference_samples,
    create_biometric_probe_samples,
)
from ..samples.samples import Sample

import bob.db.atnt
import bob.db.base


def test_training_samples():

    database = bob.db.atnt.Database()

    samples = create_training_samples(database)

    assert len(samples) > 0
    assert isinstance(samples[0].data[0], bob.db.base.File)


def test_biometric_reference_samples():

    database = bob.db.atnt.Database()

    samples = create_biometric_reference_samples(database)

    assert len(samples) > 0
    assert isinstance(samples[0].data[0], bob.db.base.File)


def test_biometric_probe_samples():

    database = bob.db.atnt.Database()

    biometric_references = create_biometric_reference_samples(database)
    probe_samples = create_biometric_probe_samples(database, biometric_references)

    assert len(probe_samples) > 0
    assert isinstance(probe_samples[0].data[0], bob.db.base.File)
    assert len(probe_samples[0].biometric_references) > 0
    assert isinstance(probe_samples[0].biometric_references[0], Sample)
