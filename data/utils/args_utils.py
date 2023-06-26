from typing import Dict, Any


def unflatten(
    d: Dict[str, Any],
    base: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Convert any keys containing dotted paths to nested dicts

    >>> unflatten({'a': 12, 'b': 13, 'c': 14})  # no expansion
    {'a': 12, 'b': 13, 'c': 14}

    >>> unflatten({'a.b.c': 12})  # dotted path expansion
    {'a': {'b': {'c': 12}}}

    >>> unflatten({'a.b.c': 12, 'a': {'b.d': 13}})  # merging
    {'a': {'b': {'c': 12, 'd': 13}}}

    >>> unflatten({'a.b': 12, 'a': {'b': 13}})  # insertion-order overwrites
    {'a': {'b': 13}}

    >>> unflatten({'a': {}})  # insertion-order overwrites
    {'a': {}}
    """
    if base is None:
        base = {}

    for key, value in d.items():
        root = base

        ###
        # If a dotted path is encountered, create nested dicts for all but
        # the last level, then change root to that last level, and key to
        # the final key in the path.
        #
        # This allows one final setitem at the bottom of the loop.
        #
        if '.' in key:
            *parts, key = key.split('.')

            for part in parts:
                root.setdefault(part, {})
                root = root[part]

        if isinstance(value, dict):
            value = unflatten(value, root.get(key, {}))

        root[key] = value

    return base


def set_nonetype(dic: Dict):
    for k, v in dic.items():
        if isinstance(v, str) and v.lower() == 'none':
            dic[k] = None
        elif isinstance(v, dict):
            dic[k] = set_nonetype(v)
    return dic
