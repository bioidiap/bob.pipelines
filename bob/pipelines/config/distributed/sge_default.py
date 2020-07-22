from dask.distributed import Client

from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster

cluster = SGEMultipleQueuesCluster(min_jobs=20)
dask_client = Client(cluster)
