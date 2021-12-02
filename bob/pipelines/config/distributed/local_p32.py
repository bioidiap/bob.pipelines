from bob.pipelines.distributed import get_local_parallel_client

dask_client = get_local_parallel_client(32)
