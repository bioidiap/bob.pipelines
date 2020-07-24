import os

from tempfile import NamedTemporaryFile

import nose
import numpy as np

import bob.pipelines as mario


def test_io_vstack():

    paths = [1, 2, 3, 4, 5]

    def asser_(actual, desired, dtype=None):
        np.testing.assert_allclose(actual, desired)
        if dtype is not None:
            assert actual.dtype == dtype, (actual.dtype, dtype)

    def oracle(reader, paths):
        return np.vstack([reader(p) for p in paths])

    def reader_same_size_C(path):
        return np.arange(10).reshape(5, 2) + path

    def reader_different_size_C(path):
        return np.arange(2 * path).reshape(path, 2) + path

    def reader_same_size_F(path):
        return np.asfortranarray(np.arange(10).reshape(5, 2)) + path

    def reader_different_size_F(path):
        return np.asfortranarray(np.arange(2 * path).reshape(path, 2)) + path

    def reader_same_size_C2(path):
        return np.arange(30).reshape(5, 2, 3) + path

    def reader_different_size_C2(path):
        return np.arange(6 * path).reshape(path, 2, 3) + path

    def reader_same_size_F2(path):
        return np.asfortranarray(np.arange(30).reshape(5, 2, 3)) + path

    def reader_different_size_F2(path):
        return np.asfortranarray(np.arange(6 * path).reshape(path, 2, 3)) + path

    def reader_wrong_size(path):
        return np.arange(2 * path).reshape(2, path) + path

    dtype = "float32"
    # when same_size is False
    for reader in [
        reader_different_size_C,
        reader_different_size_F,
        reader_same_size_C,
        reader_same_size_F,
        reader_different_size_C2,
        reader_different_size_F2,
        reader_same_size_C2,
        reader_same_size_F2,
    ]:
        asser_(mario.utils.vstack_features(reader, paths), oracle(reader, paths))
        asser_(
            mario.utils.vstack_features(reader, paths, dtype=dtype),
            oracle(reader, paths),
            dtype,
        )

    # when same_size is True
    for reader in [
        reader_same_size_C,
        reader_same_size_F,
        reader_same_size_C2,
        reader_same_size_F2,
    ]:
        asser_(mario.utils.vstack_features(reader, paths, True), oracle(reader, paths))
        asser_(
            mario.utils.vstack_features(reader, paths, True, dtype=dtype),
            oracle(reader, paths),
            dtype,
        )

    with nose.tools.assert_raises(AssertionError):
        mario.utils.vstack_features(reader_wrong_size, paths)

    # test actual files
    suffix = ".npy"
    with NamedTemporaryFile(suffix=suffix) as f1, NamedTemporaryFile(
        suffix=suffix
    ) as f2, NamedTemporaryFile(suffix=suffix) as f3:
        paths = [f1.name, f2.name, f3.name]
        # try different readers:
        for reader in [
            reader_different_size_C,
            reader_different_size_F,
            reader_same_size_C,
            reader_same_size_F,
            reader_different_size_C2,
            reader_different_size_F2,
            reader_same_size_C2,
            reader_same_size_F2,
        ]:
            # save some data in files
            for i, path in enumerate(paths):
                np.save(path, reader(i + 1), allow_pickle=False)
            # test when all data is present
            reference = oracle(np.load, paths)
            asser_(mario.utils.vstack_features(np.load, paths), reference)
            asser_(
                mario.utils.vstack_features(np.load, paths, dtype=dtype),
                reference,
                dtype,
            )
            try:
                os.remove(paths[0])
                # Check if RuntimeError is raised when one of the files is missing
                with nose.tools.assert_raises(FileNotFoundError):
                    mario.utils.vstack_features(np.load, paths)
            finally:
                # create the file back so NamedTemporaryFile does not complain
                np.save(paths[0], reader(i + 1))


def test_isinstance_nested():
    class A:
        pass

    class B:
        def __init__(self, o):
            self.o = o

    class C:
        def __init__(self, o):
            self.o = o

    o = C(B(A()))
    assert mario.utils.isinstance_nested(o, "o", C)
    assert mario.utils.isinstance_nested(o, "o", B)
    assert mario.utils.isinstance_nested(o, "o", A)

    o = C(B(object))
    assert mario.utils.isinstance_nested(o, "o", C)
    assert mario.utils.isinstance_nested(o, "o", B)
    assert not mario.utils.isinstance_nested(o, "o", A)
