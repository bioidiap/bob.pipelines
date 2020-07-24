from multiprocessing import cpu_count

from dask.distributed import Client
from dask.distributed import LocalCluster

n_nodes = cpu_count()
threads_per_worker = 1

cluster = LocalCluster(
    nanny=False, processes=False, n_workers=1, threads_per_worker=threads_per_worker
)
cluster.scale_up(n_nodes)
dask_client = Client(cluster)  # start local workers as threads
