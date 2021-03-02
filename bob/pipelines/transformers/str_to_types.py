from sklearn.preprocessing import FunctionTransformer


def str_to_types(samples, fieldtypes):
    for s in samples:
        for key in s.__dict__:
            if key not in fieldtypes:
                continue
            value = getattr(s, key)
            value = fieldtypes[key](value)
            setattr(s, key, value)
    return samples


def Str_To_Types(fieldtypes):
    """Converts str fields in samples to a different type

    Parameters
    ----------
    fieldtypes : dict
        A dict that specifies the functions to be used to convert strings to other types.

    Returns
    -------
    object
        A scikit-learn transformer that does the conversion.

    Example
    -------
    >>> from bob.pipelines import Sample
    >>> from bob.pipelines.transformers import Str_To_Types
    >>> samples = [Sample(None, id="1", flag="True"), Sample(None, id="2", flag="False")]
    >>> transformer = Str_To_Types(fieldtypes=dict(id=int, flag=bool))
    >>> transformer.transform(samples)
    [Sample(data=None, id=1, flag=True), Sample(data=None, id=2, flag=True)]
    """
    return FunctionTransformer(
        str_to_types, kw_args=dict(fieldtypes=fieldtypes), validate=False
    )
