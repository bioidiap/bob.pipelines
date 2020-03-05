from bob.pipelines.distributed.sge import SGEIdiapCluster
from dask.distributed import Client

n_jobs = 16
cluster = SGEIdiapCluster()
cluster.scale(n_jobs, queue="q_1day", io_big=True)

dask_client = Client(cluster)
