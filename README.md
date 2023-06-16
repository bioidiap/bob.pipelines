[![badge doc](https://img.shields.io/badge/docs-v4.0.0-orange.svg)](https://www.idiap.ch/software/bob/docs/bob/bob.pipelines/v4.0.0/sphinx/index.html)
[![badge pipeline](https://gitlab.idiap.ch/bob/bob.pipelines/badges/v4.0.0/pipeline.svg)](https://gitlab.idiap.ch/bob/bob.pipelines/commits/v4.0.0)
[![badge coverage](https://gitlab.idiap.ch/bob/bob.pipelines/badges/v4.0.0/coverage.svg)](https://www.idiap.ch/software/bob/docs/bob/bob.pipelines/v4.0.0/coverage)
[![badge gitlab](https://img.shields.io/badge/gitlab-project-0000c0.svg)](https://gitlab.idiap.ch/bob/bob.pipelines)

# Tools to build robust and extensible pipelines

This package is part of the signal-processing and machine learning toolbox
[Bob](https://www.idiap.ch/software/bob).

The goal is to provide more flexible pipeline mechanisms for
[bob.bio.base](http://gitlab.idiap.ch/bob/bob.bio.base) and
[bob.pad.base](http://gitlab.idiap.ch/bob/bob.pad.base).

It is based on the
[scikit-learn pipeline](https://scikit-learn.org/stable/modules/compose.html)
and adds a layer to route data and metadata through the pipeline (which is not
supported by scikit-learn yet).

## Installation

Complete bob's
[installation instructions](https://www.idiap.ch/software/bob/install). Then,
to install this package, run:

``` sh
conda install bob.pipelines
```

## Contact

For questions or reporting issues to this software package, contact our
development [mailing list](https://www.idiap.ch/software/bob/discuss).
