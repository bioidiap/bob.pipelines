from sklearn.preprocessing import FunctionTransformer


def str_to_types(samples, fieldtypes):
    for s in samples:
        for key, func in fieldtypes.items():
            value = getattr(s, key)
            value = func(value)
            setattr(s, key, value)
    return samples


def str_to_bool(value):
    return value.lower() == "true"


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
    >>> from bob.pipelines.transformers import Str_To_Types, str_to_bool
    >>> samples = [Sample(None, id="1", flag="True"), Sample(None, id="2", flag="False")]
    >>> transformer = Str_To_Types(fieldtypes=dict(id=int, flag=str_to_bool))
    >>> transformer.transform(samples)
    [Sample(data=None, id=1, flag=True), Sample(data=None, id=2, flag=False)]
    """
    return FunctionTransformer(
        str_to_types, kw_args=dict(fieldtypes=fieldtypes), validate=False
    )
