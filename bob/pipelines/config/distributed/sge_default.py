from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster
from dask.distributed import Client

cluster = SGEMultipleQueuesCluster(min_jobs=20)
dask_client = Client(cluster)
