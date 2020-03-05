from bob.pipelines.distributed.sge import SGEIdiapCluster
from dask.distributed import Client

n_jobs = 16
n_gpu_jobs = 1
cluster = SGEIdiapCluster()
cluster.scale(n_jobs, queue="q_1day", io_big=True)
cluster.scale(n_gpu_jobs, queue="q_gpu", resources="GPU=1")

dask_client = Client(cluster)
