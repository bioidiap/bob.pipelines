from bob.pipelines.distributed.sge import SGEIdiapCluster
from dask.distributed import Client


Q_1DAY_IO_BIG_SPEC = {
    "default": {
        "queue": "q_1day",
        "memory": "4GB",
        "io_big": True,
        "resource_spec": "",
        "resources": "",
    }
}

n_jobs = 16
cluster = SGEIdiapCluster(sge_job_spec=Q_1DAY_IO_BIG_SPEC)
cluster.scale(n_jobs)

dask_client = Client(cluster)
