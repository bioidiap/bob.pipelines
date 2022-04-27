from dask.distributed import Client

from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster, get_max_jobs
from bob.pipelines.distributed.sge_queues import QUEUE_IOBIG

min_jobs = max_jobs = get_max_jobs(QUEUE_IOBIG)
cluster = SGEMultipleQueuesCluster(min_jobs=min_jobs, sge_job_spec=QUEUE_IOBIG)
cluster.scale(max_jobs)
# Adapting to minimim 1 job to maximum 48 jobs
# interval: Milliseconds between checks from the scheduler
# wait_count: Number of consecutive times that a worker should be suggested for
#             removal before we remove it.
cluster.adapt(
    minimum=min_jobs,
    maximum=max_jobs,
    wait_count=5,
    interval=10,
    target_duration="10s",
)
dask_client = Client(cluster)
