from dask.distributed import Client

from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster, get_max_jobs
from bob.pipelines.distributed.sge_queues import QUEUE_IOBIG

min_jobs = max_jobs = get_max_jobs(QUEUE_IOBIG)
cluster = SGEMultipleQueuesCluster(min_jobs=min_jobs, sge_job_spec=QUEUE_IOBIG)
cluster.scale(max_jobs)

dask_client = Client(cluster)
