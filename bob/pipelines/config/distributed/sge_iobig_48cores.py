from bob.pipelines.distributed.sge import SGEIdiapCluster, Q_1DAY_IO_BIG_SPEC
from dask.distributed import Client

n_jobs = 48
cluster = SGEIdiapCluster(sge_job_spec=Q_1DAY_IO_BIG_SPEC)
cluster.scale(n_jobs)

dask_client = Client(cluster)
