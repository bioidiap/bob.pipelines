#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import dist, setup

from bob.extension.utils import find_packages, load_requirements

dist.Distribution(dict(setup_requires=["bob.extension"]))


install_requires = load_requirements()


setup(
    # This is the basic information about the project.
    name="bob.pipelines",
    version=open("version.txt").read().rstrip(),
    description="Tools to build robust and extensible pipelines",
    url="https://gitlab.idiap.cih/bob/bob.pipelines",
    license="BSD",
    # there may be multiple authors (separate entries by comma)
    author="Tiago de Freitas Pereira",
    author_email="tiago.pereira@idiap.ch",
    # there may be a maintainer apart from the author - you decide
    # maintainer='?'
    # maintainer_email='email@example.com'
    # you may add more keywords separating those by commas (a, b, c, ...)
    keywords="bob",
    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description=open("README.rst").read(),
    # This line is required for any distutils based packaging.
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    # Packages that should be installed when you "install" this package.
    install_requires=install_requires,
    # entry_points defines which scripts will be inside the 'bin' directory
    entry_points={
        "dask.client": [
            "local-parallel  = bob.pipelines.config.distributed.local_parallel:dask_client",
            "local-p4        = bob.pipelines.config.distributed.local_p4:dask_client",
            "local-p8        = bob.pipelines.config.distributed.local_p8:dask_client",
            "local-p16       = bob.pipelines.config.distributed.local_p16:dask_client",
            "local-p32       = bob.pipelines.config.distributed.local_p32:dask_client",
            "sge             = bob.pipelines.config.distributed.sge_default:dask_client",
            "sge-demanding   = bob.pipelines.config.distributed.sge_demanding:dask_client",
            "sge-io-big      = bob.pipelines.config.distributed.sge_io_big:dask_client",
            "sge-io-big-non-adaptive = bob.pipelines.config.distributed.sge_io_big_non_adaptive:dask_client",
            "sge-gpu         = bob.pipelines.config.distributed.sge_gpu:dask_client",
        ],
    },
    # check classifiers (important for PyPI), add and remove as you see fit.
    # full list here: https://pypi.org/classifiers/
    # don't remove the Bob framework unless it's not a bob package
    classifiers=[
        "Framework :: Bob",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
