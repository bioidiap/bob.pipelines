from dask.distributed import Client

from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster
from bob.pipelines.distributed.sge_queues import QUEUE_GPU

cluster = SGEMultipleQueuesCluster(min_jobs=1, sge_job_spec=QUEUE_GPU)
dask_client = Client(cluster)
