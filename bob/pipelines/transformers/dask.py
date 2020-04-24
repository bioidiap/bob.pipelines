import dask.bag
from sklearn.base import TransformerMixin, BaseEstimator


class ToDaskBag(TransformerMixin, BaseEstimator):
    """Transform an arbitrary iterator into a :py:class:`dask.bag`

    Paramters
    ---------

      npartitions: int
        Number of partitions used in :py:meth:`dask.bag.from_sequence`


    Example
    -------

    >>> transformer = DaskBagMixin()
    >>> dask_bag = transformer.transform([1,2,3])
    >>> dask_bag.map_partitions.....

    """
    def __init__(self, npartitions=None, **kwargs):
        super().__init__(**kwargs)
        self.npartitions = npartitions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return dask.bag.from_sequence(X, npartitions=self.npartitions)

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}
