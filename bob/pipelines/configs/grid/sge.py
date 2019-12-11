"""
SGE executor
"""

# POSSIBLE CHAIN LOADED INFO
NODES = 10
QUEUE = "q1d"
QUEUE_RESOURCE_SPEC = "q_1day=TRUE,io_big=TRUE"
MEMORY = "4GB"
SGE_LOG = "/remote/idiap.svm/user.active/tpereira/gitlab/bob/bob.pipelines/logs"
#gpu_resource_spec = "q_gpu=TRUE"



queue = QUEUE
queue_resource_spec = QUEUE_RESOURCE_SPEC
memory = MEMORY
sge_log = SGE_LOG
nodes = NODES



from dask_jobqueue import PBSCluster, SGECluster



cluster = SGECluster(queue=queue, memory=memory, cores=1,
        log_directory=sge_log,
        local_directory=sge_log,
        resource_spec=
        )

cluster.scale_up(nodes)
client = Client(cluster)  # start local workers as threads
