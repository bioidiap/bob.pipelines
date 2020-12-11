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

try:
    import dask

    VALID_DASK_CLIENT_STRINGS = dask.base.named_schedulers
except (ModuleNotFoundError, ImportError):
    VALID_DASK_CLIENT_STRINGS = (
        "sync",
        "synchronous",
        "single-threaded",
        "threads",
        "threading",
        "processes",
        "multiprocessing",
    )


def dask_get_partition_size(cluster, n_objects, lower_bound=200):
    """
    Heuristics that gives you a number for dask.partition_size.
    The heuristics is pretty simple, given the max number of possible workers to be run
    in a queue (not the number of current workers running) and a total number objects to be processed do n_objects/n_max_workers:

    Check https://docs.dask.org/en/latest/best-practices.html#avoid-very-large-partitions
    for best practices

    Parameters
    ----------

        cluster:  :any:`bob.pipelines.distributed.sge.SGEMultipleQueuesCluster`
            Cluster of the type :any:`bob.pipelines.distributed.sge.SGEMultipleQueuesCluster`

        n_objects: int
            Number of objects to be processed

        lower_bound: int
            Minimum partition size.

    """
    from .sge import SGEMultipleQueuesCluster

    if not isinstance(cluster, SGEMultipleQueuesCluster):
        return None

    max_jobs = cluster.sge_job_spec["default"]["max_jobs"]

    # Trying to set a lower bound for the
    return (
        max(n_objects // max_jobs, lower_bound) if n_objects > max_jobs else n_objects
    )
