#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""
Execute a particular pipeline
"""
from bob.extension.scripts.click_helper import (
            verbosity_option, log_parameters)
import click
from bob.extension.scripts.click_helper import verbosity_option, ResourceOption
from bob.extension.config import load

EPILOG = '''\b


The execution 


'''

@click.command(context_settings={'ignore_unknown_options': True,
                                 'allow_extra_args': True})
@click.argument('execution', required=True)
@verbosity_option(cls=ResourceOption)
@click.pass_context
def run(ctx, execution, **kwargs):
    """Run a pipeline

    \b

    bob pipelines run ./bob/pipelines/configs/grid/local.py

    """

    import ipdb; ipdb.set_trace()
    pipeline_config = load([execution]) 

    pass
