# see https://docs.python.org/3/library/pkgutil.html
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

# DASK-click VALID_DASK_CLIENT_STRINGS
# TO BE USED IN:
# @click.option(
#    "--dask-client",
#    "-l",
#    entry_point_group="dask.client",
#    string_exceptions=VALID_DASK_CLIENT_STRINGS,
#    default="single-threaded",
#    help="Dask client for the execution of the pipeline.",
#    cls=ResourceOption,
# )

VALID_DASK_CLIENT_STRINGS = ("single-threaded", "sync", "threaded", "processes")
