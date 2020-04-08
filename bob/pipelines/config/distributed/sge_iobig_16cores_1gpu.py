from bob.pipelines.distributed.sge import SGEIdiapCluster, Q_1DAY_GPU_SPEC
from dask.distributed import Client

n_jobs = 16
n_gpu_jobs = 1
cluster = SGEIdiapCluster(sge_job_spec=Q_1DAY_GPU_SPEC)
cluster.scale(n_jobs, sge_job_spec_key="default")
cluster.scale(n_gpu_jobs, sge_job_spec_key="gpu")

dask_client = Client(cluster)
