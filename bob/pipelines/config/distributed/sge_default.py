from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster
from dask.distributed import Client

cluster = SGEMultipleQueuesCluster()
dask_client = Client(cluster)
