.. -*- coding: utf-8 -*-

.. image:: https://img.shields.io/badge/docs-stable-yellow.svg
   :target: https://www.idiap.ch/software/bob/docs/bob/bob.pipelines/stable/index.html
.. image:: https://img.shields.io/badge/docs-latest-orange.svg
   :target: https://beatubulatest.lab.idiap.ch/private/docs/bob/bob.pipelines/master/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.pipelines/badges/master/build.svg
   :target: https://gitlab.idiap.ch/bob/bob.pipelines/commits/master
.. image:: https://gitlab.idiap.ch/bob/bob.pipelines/badges/master/coverage.svg
   :target: https://gitlab.idiap.ch/bob/bob.pipelines/commits/master
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.pipelines
.. image:: https://img.shields.io/pypi/v/bob.pipelines.svg
   :target: https://pypi.python.org/pypi/bob.pipelines


===========================================================================
 Tool that helps you create pipelines for arbitrary scientific experiments
===========================================================================

This package is part of the signal-processing and machine learning toolbox Bob_.

This is **STILL EXPERIMENTAL** and the goal is to provide better option than `bob.bio.base <http://gitlab.idiap.ch/bob/bob.bio.base>`_ and `bob.pad.base <http://gitlab.idiap.ch/bob/bob.pad.base>`_  in terms of extensibility.

The goal is to have tool with the following skeleton::

  $ bob pipelines run <EXECUTION-CONFIG> <DATABASE-CONFIG> <EXPERIMENT-CONFIG> [<PIPELINES>] [OPTIONS] 


Installation
------------

Complete bob's `installation`_ instructions. Then, to install this
package, run::

  $ conda install bob.pipelines


Contact
-------

For questions or reporting issues to this software package, contact our
development `mailing list`_.


.. Place your references here:
.. _bob: https://www.idiap.ch/software/bob
.. _installation: https://www.idiap.ch/software/bob/install
.. _mailing list: https://www.idiap.ch/software/bob/discuss
