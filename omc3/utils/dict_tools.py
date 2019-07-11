"""
Module utils.dict_tools
-------------------------

Advanced dictionary functionalities.
"""
from utils import logging_tools
LOG = logging_tools.get_logger(__name__)


_TC = {  # Tree Characters
    '|': u'\u2502',  # Horizontal
    '-': u'\u2500',  # Vertical
    'L': u'\u2514',  # L-Shape
    'S': u'\u251C',  # Split
}


# Additional Dictionary Classes and Functions ##################################


class DotDict(dict):
    """ Make dict fields accessible by . """
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key in self:
            if isinstance(self[key], dict):
                self[key] = DotDict(self[key])

    # __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, key):
        """ Needed to raise the correct exceptions """
        try:
            return super(DotDict, self).__getitem__(key)
        except KeyError as e:
            raise AttributeError(e).with_traceback(e.__traceback__) from e

    def get_subdict(self, keys, strict=True):
        """ See get_subdict in dict_tools. """
        return DotDict(get_subdict(self, keys, strict))


def print_dict_tree(dictionary, name='Dictionary', print_fun=LOG.info):
    """ Prints a dictionary as a tree """
    def print_tree(tree, level_char):
        for i, key in enumerate(sorted(tree.keys())):
            if i == len(tree) - 1:
                node_char = _TC['L'] + _TC['-']
                level_char_pp = level_char + '   '
            else:
                node_char = _TC['S'] + _TC['-']
                level_char_pp = level_char + _TC['|'] + '  '

            if isinstance(tree[key], dict):
                print_fun(f"{level_char:s}{node_char:s} {str(key):s}")
                print_tree(tree[key], level_char_pp)
            else:
                print_fun(f"{level_char:s}{node_char:s} {str(key):s}: {str(tree[key]):s}")

    print_fun('{:s}:'.format(name))
    print_tree(dictionary, '')


def get_subdict(full_dict, keys, strict=True):
    """ Returns a sub-dictionary of ``full_dict`` containing only keys of ``keys``.

    Args:
        full_dict: Dictionary to extract from
        keys: keys to extract
        strict: If false it ignores keys not in full_dict. Otherwise it crashes on those.
                Default: True

    Returns: Extracted sub-dictionary

    """
    if strict:
        return {k: full_dict[k] for k in keys}
    return {k: full_dict[k] for k in keys if k in full_dict}
