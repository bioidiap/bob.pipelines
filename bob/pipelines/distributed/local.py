#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


def debug_client(n_nodes):
    """
    This executor will run everything in **ONE SINGLE PROCESS**

    This will return an instance of Dask Distributed Client 
    
    https://distributed.dask.org/en/latest/client.html

    **Parameters**

       n_nodes:
         Number of process
    """

    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(
        nanny=False, processes=False, n_workers=1, threads_per_worker=1
    )
    cluster.scale_up(n_nodes)
    client = Client(cluster)  # start local workers as threads

    return client
