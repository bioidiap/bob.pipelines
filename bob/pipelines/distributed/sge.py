#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import sys

import logging

logger = logging.getLogger(__name__)


from dask_jobqueue.core import JobQueueCluster, Job
from dask_jobqueue.sge import SGEJob
from distributed.scheduler import Scheduler
from distributed import SpecCluster
import dask
from distributed.scheduler import Scheduler
from distributed.deploy import Adaptive


class SGEIdiapJob(Job):
    """
    Launches a SGE Job in the IDIAP cluster.
    This class basically encodes the CLI command that bootstrap the worker 
    in a SGE job. Check here `https://distributed.dask.org/en/latest/resources.html#worker-resources` for more information
    
    ..note: This is class is temporary. It's basically a copy from SGEJob from dask_jobqueue.
            The difference is that here I'm also handling the dask job resources tag (which is not handled anywhere). This has to be patched in the Job class. Please follow here `https://github.com/dask/dask-jobqueue/issues/378` to get news about this patch
           

    """

    submit_command = "qsub"
    cancel_command = "qdel"

    def __init__(
        self,
        *args,
        queue=None,
        project=None,
        resource_spec=None,
        job_extra=None,
        config_name="sge",
        **kwargs,
    ):
        if queue is None:
            queue = dask.config.get("jobqueue.%s.queue" % config_name)
        if project is None:
            project = dask.config.get("jobqueue.%s.project" % config_name)
        if resource_spec is None:
            resource_spec = dask.config.get("jobqueue.%s.resource-spec" % config_name)
        if job_extra is None:
            job_extra = dask.config.get("jobqueue.%s.job-extra" % config_name)

        super().__init__(*args, config_name=config_name, death_timeout=10000, **kwargs)

        # Amending the --resources in the `distributed.cli.dask_worker` CLI command
        if "resources" in kwargs and kwargs["resources"]:
            resources = kwargs["resources"]

            # Preparing the string to be sent to `dask-worker` command
            def _resource_to_str(resources):
                resources_str = ""
                for k in resources:
                    resources_str += f"{k}={resources[k]}"
                return resources_str

            resources_str = _resource_to_str(resources)

            self._command_template += f" --resources {resources_str}"

        header_lines = []
        if self.job_name is not None:
            header_lines.append("#$ -N %(job-name)s")
        if queue is not None:
            header_lines.append("#$ -q %(queue)s")
        if project is not None:
            header_lines.append("#$ -P %(project)s")
        if resource_spec is not None:
            header_lines.append("#$ -l %(resource_spec)s")

        if self.log_directory is not None:
            header_lines.append("#$ -e %(log_directory)s/")
            header_lines.append("#$ -o %(log_directory)s/")
        header_lines.extend(["#$ -cwd", "#$ -j y"])
        header_lines.extend(["#$ %s" % arg for arg in job_extra])
        header_template = "\n".join(header_lines)

        config = {
            "job-name": self.job_name,
            "queue": queue,
            "project": project,
            "processes": self.worker_processes,
            "resource_spec": resource_spec,
            "log_directory": self.log_directory,
        }
        self.job_header = header_template % config
        logger.debug("Job script: \n %s" % self.job_script())


Q_ALL_SPEC = {
    "default": {
        "queue": "all.q",
        "memory": "4GB",
        "io_big": False,
        "resource_spec": "",
        "resources": "",
    }
}

Q_1DAY_IO_BIG_SPEC = {
    "default": {
        "queue": "q_1day",
        "memory": "8GB",
        "io_big": True,
        "resource_spec": "",
        "resources": "",
    }
}

Q_1DAY_GPU_SPEC = {
    "default": {
        "queue": "q_1day",
        "memory": "8GB",
        "io_big": True,
        "resource_spec": "",
        "resources": "",
    },
    "gpu": {
        "queue": "q_gpu",
        "memory": "12GB",
        "io_big": False,
        "resource_spec": "",
        "resources": {"gpu":1},
    },
}


class SGEIdiapCluster(JobQueueCluster):
    """ Launch Dask jobs in the IDIAP SGE cluster

    Parameters
    ----------
     log_directory: str
        Default directory for the SGE logs

      protocol: str
        Scheduler communication protocol

      dashboard_address: str
        Default port for the dask dashboard,
      
      env_extra: str,
        Extra environment variables to send to the workers

      sge_job_spec: dict
        Dictionary containing a minimum specification for the qsub command.
        It cosists of:

          queue: SGE queue
          memory: Memory requirement in GB (e.g. 4GB)
          io_bio: set the io_big flag
          resource_spec: Whatever extra argument to be sent to qsub (qsub -l)
          tag: Mark this worker with an specific tag so dask scheduler can place specific tasks to it (https://distributed.dask.org/en/latest/resources.html)


    Example
    -------

    Below follow a vanilla-example that will create a set of jobs on all.q:

    >>> from bob.pipelines.distributed.sge import SGEIdiapCluster  # doctest: +SKIP
    >>> from dask.distributed import Client # doctest: +SKIP
    >>> cluster = SGEIdiapCluster() # doctest: +SKIP
    >>> cluster.scale_up(10) # doctest: +SKIP
    >>> client = Client(cluster) # doctest: +SKIP

    It's possible to demand a resource specification yourself:
        
    >>> Q_1DAY_IO_BIG_SPEC = {
    ...        "default": {
    ...        "queue": "q_1day",
    ...        "memory": "8GB",
    ...        "io_big": True,
    ...        "resource_spec": "",
    ...        "resources": "",
    ...    }
    ... }
    >>> cluster = SGEIdiapCluster(sge_job_spec=Q_1DAY_IO_BIG_SPEC) # doctest: +SKIP
    >>> cluster.scale_up(10) # doctest: +SKIP
    >>> client = Client(cluster) # doctest: +SKIP



    More than one jon spec can be set:
    >>> Q_1DAY_GPU_SPEC = {
    ...         "default": {
    ...             "queue": "q_1day",
    ...             "memory": "8GB",
    ...             "io_big": True,
    ...             "resource_spec": "",
    ...             "resources": "",
    ...         },
    ...         "gpu": {
    ...             "queue": "q_gpu",
    ...             "memory": "12GB",
    ...             "io_big": False,
    ...             "resource_spec": "",
    ...             "resources": {"GPU":1},
    ...         },
    ...     }
    >>> cluster = SGEIdiapCluster(sge_job_spec=Q_1DAY_GPU_SPEC) # doctest: +SKIP
    >>> cluster.scale_up(10) # doctest: +SKIP
    >>> cluster.scale_up(1, sge_job_spec_key="gpu") # doctest: +SKIP
    >>> client = Client(cluster) # doctest: +SKIP


    Adaptive job allocation can also be used via `AdaptiveIdiap` extension:

    >>> cluster = SGEIdiapCluster(sge_job_spec=Q_1DAY_GPU_SPEC)  # doctest: +SKIP
    >>> cluster.adapt(Adaptive=AdaptiveIdiap,minimum=2, maximum=10) # doctest: +SKIP
    >>> client = Client(cluster)     # doctest: +SKIP


    """

    def __init__(
        self,
        log_directory="./logs",
        protocol="tcp://",
        dashboard_address=":8787",
        env_extra=None,
        sge_job_spec=Q_ALL_SPEC,
        **kwargs,
    ):

        # Defining the job launcher
        self.job_cls = SGEIdiapJob
        self.sge_job_spec = sge_job_spec

        self.protocol = protocol
        self.log_directory = log_directory

        silence_logs = "error"
        secutity = None
        interface = None
        host = None
        security = None

        if env_extra is None:
            env_extra = []
        elif not isinstance(env_extra, list):
            env_extra = [env_extra]
        self.env_extra = env_extra + ["export PYTHONPATH=" + ":".join(sys.path)]

        scheduler = {
            "cls": SchedulerIdiap,  # Use local scheduler for now
            "options": {
                "protocol": self.protocol,
                "interface": interface,
                "host": host,
                "dashboard_address": dashboard_address,
                "security": security,
            },
        }

        # Spec cluster parameters
        loop = None
        asynchronous = False
        name = None

        # Starting the SpecCluster constructor
        super(JobQueueCluster, self).__init__(
            scheduler=scheduler,
            worker={},
            loop=loop,
            silence_logs=silence_logs,
            asynchronous=asynchronous,
            name=name,
        )

    def _get_worker_spec_options(self, job_spec):
        """
        Craft a dask worker_spec to be used in the qsub command

        """

        def _get_key_from_spec(spec, key):
            return spec[key] if key in spec else ""

        new_resource_spec = _get_key_from_spec(job_spec, "resource_spec")

        # IO_BIG
        new_resource_spec += (
            "io_big=TRUE," if "io_big" in job_spec and job_spec["io_big"] else ""
        )

        queue = _get_key_from_spec(job_spec, "queue")
        if queue != "all.q":
            new_resource_spec += f"{queue}=TRUE"

        new_resource_spec = None if new_resource_spec == "" else new_resource_spec

        return {
            "queue": queue,
            "memory": _get_key_from_spec(job_spec, "memory"),
            "cores": 1,
            "processes": 1,
            "log_directory": self.log_directory,
            "local_directory": self.log_directory,
            "resource_spec": new_resource_spec,
            "interface": None,
            "protocol": self.protocol,
            "security": None,
            "resources": _get_key_from_spec(job_spec, "resources"),
            "env_extra": self.env_extra,
        }

    def scale(self, n_jobs, sge_job_spec_key="default"):
        """
        Launch an SGE job in the Idiap SGE cluster

        Parameters
        ----------

          n_jobs: int
            Quantity of jobs to scale

          sge_job_spec_key: str
             One of the specs `SGEIdiapCluster.sge_job_spec` 
        """

        if n_jobs == 0:
            # Shutting down all workers
            return super(JobQueueCluster, self).scale(0, memory=None, cores=0)

        job_spec = self.sge_job_spec[sge_job_spec_key]
        worker_spec_options = self._get_worker_spec_options(job_spec)
        n_cores = 1
        worker_spec = {"cls": self.job_cls, "options": worker_spec_options}

        # Defining a new worker_spec with some SGE characteristics
        self.new_spec = worker_spec

        return super(JobQueueCluster, self).scale(n_jobs, memory=None, cores=n_cores)

    def scale_up(self, n_jobs, sge_job_spec_key=None):
        """
        Scale cluster up. This is supposed to be used by the scheduler while dynamically allocating resources
        """
        return self.scale(n_jobs, sge_job_spec_key)

    async def scale_down(self, workers, sge_job_spec_key=None):
        """
        Scale cluster down. This is supposed to be used by the scheduler while dynamically allocating resources
        """
        await super().scale_down(workers)

    def adapt(self, *args, **kwargs):
        super().adapt(*args, Adaptive=AdaptiveIdiap, **kwargs)



class AdaptiveIdiap(Adaptive):
    """
    Custom mechanism to adaptively allocate workers based on scheduler load
    
    This custom implementation extends the `Adaptive.recommendations` by looking
    at the `distributed.scheduler.TaskState.resource_restrictions`.

    The heristics is:

    .. note ::
        If a certain task has the status `no-worker` and it has resource_restrictions, the scheduler should
        request a job matching those resource restrictions

    """

    async def recommendations(self, target: int) -> dict:
        """
        Make scale up/down recommendations based on current state and target
        """

        plan = self.plan
        requested = self.requested
        observed = self.observed

        # Get tasks with no worker associated due to
        # resource restrictions
        resource_restrictions = (
            await self.scheduler.get_no_worker_tasks_resource_restrictions()
        )

        # If the amount of resources requested is bigger
        # than what available and those jobs has restrictions
        if target > len(plan):
            self.close_counts.clear()
            if len(resource_restrictions) > 0:
                return {
                    "status": "up",
                    "n": target,
                    "sge_job_spec_key": list(resource_restrictions[0].keys())[0],
                }
            else:
                return {"status": "up", "n": target}

        # If the amount of resources requested is lower
        # than what is available, is time to downscale
        elif target < len(plan):
            to_close = set()

            # Get the worksers that can be closed.
            if target < len(plan) - len(to_close):
                L = await self.workers_to_close(target=target)
                to_close.update(L)

            firmly_close = set()
            # COUNTING THE AMOUNT OF SCHEDULER CYCLES THAT WE SHOULD KEEP
            # THIS WORKER BEFORE DESTROYING IT
            for w in to_close:
                self.close_counts[w] += 1
                if self.close_counts[w] >= self.wait_count:
                    firmly_close.add(w)

            for k in list(self.close_counts):  # clear out unseen keys
                if k in firmly_close or k not in to_close:
                    del self.close_counts[k]

            # Send message to destroy workers
            if firmly_close:
                return {"status": "down", "workers": list(firmly_close)}

        # If the amount of available workers is ok
        # for the current demand, BUT
        # there are tasks that need some special worker:
        # SCALE EVERYTHING UP
        if target == len(plan) and len(resource_restrictions) > 0:
            return {
                "status": "up",
                "n": target + 1,
                "sge_job_spec_key": list(resource_restrictions[0].keys())[0],
            }
        else:
            return {"status": "same"}

    async def scale_up(self, n, sge_job_spec_key="default"):
        await self.cluster.scale(n, sge_job_spec_key=sge_job_spec_key)

    async def scale_down(self, workers, sge_job_spec_key="default"):
        await super().scale_down(workers)



class SchedulerIdiap(Scheduler):
    """
    Idiap extended distributed scheduler

    This scheduler extends `Scheduler` by just adding a handler
    that fetches, at every scheduler cycle, the resource restrictions of 
    a task that has status `no-worker`
    """

    def __init__(self, *args, **kwargs):
        super(SchedulerIdiap, self).__init__(*args, **kwargs)
        self.handlers[
            "get_no_worker_tasks_resource_restrictions"
        ] = self.get_no_worker_tasks_resource_restrictions

    def get_no_worker_tasks_resource_restrictions(self, comm=None):
        """
        Get the a task resource restrictions for jobs that has the status 'no-worker'
        """

        resource_restrictions = []
        for k in self.tasks:
            if (
                self.tasks[k].state == "no-worker"
                and self.tasks[k].resource_restrictions is not None
            ):
                resource_restrictions.append(self.tasks[k].resource_restrictions)

        return resource_restrictions        
