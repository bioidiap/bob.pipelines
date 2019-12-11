"""
Local executor with ONE process
"""

# chain loaded information
NODES = 1


nodes = NODES


from dask.distributed import Client, LocalCluster

cluster = LocalCluster(nanny=False, processes=False, n_workers=1, threads_per_worker=1)
cluster.scale_up(nodes)
client = Client(cluster)  # start local workers as threads
