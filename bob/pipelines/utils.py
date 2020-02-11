#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from dask.delayed import Delayed
import copy

class DaskDelayedTape:
    """
    Record dask delayed operations with a particular TAG.

    The dask.scheduler might use the information from this tape to place a taks to a particular worker (tagged with the same tag).
    Read https://distributed.dask.org/en/latest/resources.html#resources-with-collections for more information.

    
    Parameters
    ----------

      tag: str
        The name of the tag

    Example
    -------
    >>> with DaskDelayedTape("GPU") as tape:
    >>>     obj1 = dask.delayed(operation)()
    >>>     tape.tape(obj1)
   

    """

    def __init__(self, tag, old_tape=None):

        if old_tape is None:
            self._tape = dict()
        else:
            self._tape = copy.copy(old_tape.get_taped_delayeds)

        self.tag = tag

    def __enter__(self):
        return self

    def tape(self, obj):
        if isinstance(obj, Delayed):
            self._tape[tuple(obj.__dask_keys__())] = {self.tag: 1}
        else:
            ValueError("Only `dask.delayed.Delayed` objects can be taped")

    @property
    def get_taped_delayeds(self):
        return self._tape

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
