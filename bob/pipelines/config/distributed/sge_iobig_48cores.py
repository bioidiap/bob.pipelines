from bob.pipelines.distributed.sge import SGEIdiapCluster, Q_1DAY_IO_BIG_SPEC
from dask.distributed import Client

n_jobs = 48
cluster = SGEIdiapCluster(sge_job_spec=Q_1DAY_IO_BIG_SPEC)
cluster.scale(10)

# Adapting to minimim 1 job to maximum 48 jobs
# interval: Milliseconds between checks from the scheduler
# wait_count: Number of consecutive times that a worker should be suggested for 
#             removal before we remove it.
#             Here the goal is to wait 2 minutes before scaling down since
#             it is very expensive to get jobs on the SGE grid
cluster.adapt(minimum=10, maximum=n_jobs, wait_count=120, interval=1000)


dask_client = Client(cluster)
