#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


def sge_iobig_client(n_nodes, queue="q_1day", queue_resource_spec="q_1day=TRUE,io_big=TRUE", memory="4GB", sge_log= "./logs"):

    from dask_jobqueue import SGECluster

    cluster = SGECluster(queue=queue, memory=memory, cores=1,
            log_directory=sge_log,
            local_directory=sge_log,
            resource_spec=queue_resource_spec
            )

    cluster.scale_up(n_nodes)
    client = Client(cluster)  # start local workers as threads

    return client

