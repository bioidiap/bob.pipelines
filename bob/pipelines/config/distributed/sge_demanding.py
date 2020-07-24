from dask.distributed import Client

from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster
from bob.pipelines.distributed.sge_queues import QUEUE_DEMANDING

cluster = SGEMultipleQueuesCluster(min_jobs=20, sge_job_spec=QUEUE_DEMANDING)
dask_client = Client(cluster)
