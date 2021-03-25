import os

from sklearn.preprocessing import FunctionTransformer

from bob.io.base import load

from ..wrappers import wrap


def file_loader(files, original_directory, original_extension):
    data = []
    for path in files:
        d = load(os.path.join(original_directory, path + original_extension))
        data.append(d)
    return data


def FileLoader(original_directory, original_extension=None, **kwargs):
    original_directory = original_directory or ""
    original_extension = original_extension or ""
    return FunctionTransformer(
        file_loader,
        validate=False,
        kw_args=dict(
            original_directory=original_directory, original_extension=original_extension
        ),
    )


def key_based_file_loader(original_directory, original_extension):
    transformer = FileLoader(original_directory, original_extension)
    # transformer takes as input sample.key and its output is saved in sample.data
    transformer = wrap(["sample"], transformer, input_attribute="key")
    return transformer
