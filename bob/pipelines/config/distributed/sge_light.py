from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster
from bob.pipelines.distributed.sge_queues import QUEUE_LIGHT
from dask.distributed import Client

cluster = SGEMultipleQueuesCluster(sge_job_spec=QUEUE_LIGHT)
dask_client = Client(cluster)