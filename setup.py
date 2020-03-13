#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, dist

dist.Distribution(dict(setup_requires=["bob.extension"]))

from bob.extension.utils import load_requirements, find_packages

install_requires = load_requirements()


setup(
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
    long_description=open("README.rst").read(),
    # leave this here, it is pretty standard
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    entry_points={
        "bob.cli": ["pipelines = bob.pipelines.script.pipelines:pipelines",],
    },
    # check classifiers, add and remove as you see fit
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
