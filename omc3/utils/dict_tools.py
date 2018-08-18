"""

"""
import copy
import six
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
            raise AttributeError(e)  # TODO: Adapt traceback to not link here (Python3 does that?)

    def get_subdict(self, keys, strict=True):
        """ See get_subdict in dict_tools. """
        return get_subdict(self, keys, strict)


def print_dict_tree(dictionary, name='Dictionary'):
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
                LOG.info(u"{:s}{:s} {:s}"
                         .format(level_char, node_char, str(key)))
                print_tree(tree[key], level_char_pp)
            else:
                LOG.info(u"{:s}{:s} {:s}: {:s}"
                         .format(level_char, node_char, str(key), str(tree[key])))

    LOG.info('{:s}:'.format(name))
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


# Dict Parser ##################################################################


class ParameterError(Exception):
    pass


class ArgumentError(Exception):
    pass


class Parameter(object):
    """ Helper Class for DictParser """
    def __init__(self, name, **kwargs):
        self.name = name
        self.required = kwargs.pop('required', False)
        self.default = kwargs.pop('default', None)
        self.help = kwargs.pop('help', '')
        self.type = kwargs.pop('type', None)
        self.nargs = kwargs.pop('nargs', None)
        self.subtype = kwargs.pop('subtype', None)
        self.choices = kwargs.pop('choices', None)

        if len(kwargs) > 0:
            ParameterError("'{:s}' are not valid parameters for Argument.".format(kwargs.keys()))

        self._validate()

    def _validate(self):
        if not isinstance(self.name, six.string_types):
            raise ParameterError("Parameter '{:s}': ".format(str(self.name)) +
                                 "Name is not a valid string.")

        if self.default and self.type and not isinstance(self.default, self.type):
            raise ParameterError("Parameter '{:s}': ".format(self.name) +
                                 "Default value not of specified type.")

        if self.choices:
            try:
                if self.default and self.default not in self.choices:
                    raise ParameterError("Parameter '{:s}': ".format(self.name) +
                                         "Default value not found in choices.")

                if self.type or self.subtype:
                    check = self.type if self.subtype is None else self.subtype
                    for choice in self.choices:
                        if not isinstance(choice, check):
                            raise ParameterError("Choice '{}' ".format(choice) +
                                                 "of parameter '{:s}': ".format(self.name) +
                                                 "is not of type '{:s}'.".format(check.__name__))
            except TypeError:
                raise ParameterError("Parameter '{:s}': ".format(self.name) +
                                     "Choices seem to be not iterable.")

        if self.nargs:
            if not isinstance(self.nargs, int):
                raise ParameterError("Parameter '{:s}': ".format(self.name) +
                                     "nargs needs to be an integer.")

            if not (self.type or self.type == list):
                raise ParameterError("Parameter '{:s}': ".format(self.name) +
                                     "'type' needs to be 'list' if 'nargs' is given.")

        if self.subtype and not (self.type or self.type == list):
            raise ParameterError("Parameter '{:s}': ".format(self.name) +
                                 "field 'subtype' is only accepted if 'type' is list.")

        if self.required and self.default is not None:
            LOG.warn("Parameter '{:s}': ".format(self.name) +
                     "Value is required but default value is given. The latter will be ignored.")


class DictParser(object):
    """ Provides functions to parse a dictionary.

    First build a dictionary structure with Arguments as leafs via add_argument or on init.
    A similar structured option dictionary with the values as leafs can then be parsed.
    """

    def __init__(self, dictionary=None, strict=False):
        """
        Initialize Class either empty or with preconfigured dictionary

        Args:
            dictionary: Preconfigured Dictionary for parsing
            strict: Strict Parsers don't accept unknown options. If False, it just logs the names.
        """
        self.strict = strict

        if dictionary:
            self._validate_parameters(dictionary)
            self.dictionary = dictionary
        else:
            self.dictionary = {}

    #########################
    # Static Methods (private)
    #########################

    @staticmethod
    def _validate_parameters(dictionary):
        """ Validates an input dictionary that can be used as parameters.

        Args:
            dictionary: Dictionary to validate
        """
        for key in dictionary:
            param = dictionary[key]
            if isinstance(param, dict):
                try:
                    DictParser._validate_parameters(param)
                except ParameterError as e:
                    e.message = "'{:s}.{:s}".format(key, e.message[1:])
                    e.args = (e.message,)
                    raise
            elif not isinstance(param, Parameter):
                raise ParameterError("'{:s}' is not a valid entry.".format(key))
            else:
                if key != param.name:
                    raise ParameterError("'{:s}': Key and name need to be the same.".format(key))

    @staticmethod
    def _check_value(key, arg_dict, param_dict):
        """ Checks if in arg_dict[key] satisfies param_dict[key]

        Args:
            key: key to check
            arg_dict: Arguments-structure. Can be None or empty.
            param_dict: Parameter-structure. Needs to contain 'key'

        Returns:
            The appropriate value for arg_dict[key]
        """
        param = param_dict[key]
        if not arg_dict or key not in arg_dict:
            if param.required:
                raise ArgumentError("'{:s}' required in options.\nHelp: {:s}".format(
                    key, param.help)
                )
            else:
                return param.default

        opt = arg_dict[key]
        if opt is None:
            if param.required:
                raise ArgumentError("'{:s}' required in options.\nHelp: {:s}".format(
                    key, param.help)
                )
        else:
            if param.type and not isinstance(opt, param.type):
                raise ArgumentError("'{:s}' is not of type {:s}.\nHelp: {:s}".format(
                    key, param.type.__name__, param.help)
                )
            if param.type == list:
                if param.nargs and not param.nargs == len(opt):
                    raise ArgumentError(
                        "'{:s}' should be list of length {:d},".format(key, param.nargs) +
                        " instead it was of length {:d}.\nHelp: {:s}".format(len(opt), param.help))
                if param.subtype:
                    for idx, item in enumerate(opt):
                        if not isinstance(item, param.subtype):
                            raise ArgumentError(
                                "Item {:d} of '{:s}' is not of type '{:s}' ".format(
                                    idx, key, param.subtype.__name__) +
                                ".\nHelp: {:s}".format(param.help))

                if param.choices and any([o for o in opt if o not in param.choices]):
                    raise ArgumentError(
                        "All elements of '{:s}' need to be one of {:s},".format(key,
                                                                                param.choices) +
                        " instead the list was {:s}.\nHelp: {:s}".format(str(opt), param.help)
                    )

            elif param.choices and opt not in param.choices:
                raise ArgumentError(
                    "'{:s}' needs to be one of {:s}, instead it was {:s}.\nHelp: {:s}".format(
                    key, param.choices, str(opt), param.help)
                )
        return opt

    def _parse_options(self, arg_dict, param_dict):
        """ Use parse_options()!

        This is a helper Function for parsing options. It does all the work. Called recursively.

        Args:
            arg_dict: Dictionary with the input parameter
            param_dict: Dictionary with the parameters to check the parameter against
        Returns:
            Dictionary with parsed options
        """
        checked_dict = DotDict()
        for key in param_dict:
            if isinstance(param_dict[key], Parameter):
                checked_dict[key] = DictParser._check_value(key, arg_dict, param_dict)
            elif isinstance(param_dict[key], dict):
                try:
                    if not arg_dict or not (key in arg_dict):
                        checked_dict[key] = DictParser._parse_options(None,
                                                                      param_dict[key])
                    else:
                        checked_dict[key] = DictParser._parse_options(arg_dict[key],
                                                                      param_dict[key])
                except ArgumentError as e:
                    old_msg = e.message[1:]
                    if old_msg.startswith("'"):
                        e.message = "'{:s}.{:s}".format(key, e.message[1:])
                    else:
                        e.message = "'{:s}' has {:s}".format(key, e.message)
                    e.args = (e.message,)
                    raise

            arg_dict.pop(key, None)  # Default value avoids KeyError

        if len(arg_dict) > 0:
            error_message = "Unknown Options: '{:s}'.".format(arg_dict.keys())
            if self.strict:
                raise ArgumentError(error_message)
            LOG.debug(error_message)

        if self.strict:
            return checked_dict
        else:
            return checked_dict, arg_dict

    #########################
    # Public Methods
    #########################

    def parse_arguments(self, arguments):
        """ Parse a given option dictionary and return parsed options.

        Args:
            arguments: Arguments to parse

        Return:
            Parsed options
        """
        return self._parse_options(copy.deepcopy(arguments), self.dictionary)

    def parse_config_items(self, items):
        """ Parse a list of (name, value) items, where the values are all strings.

        Args:
            items: list of (name, value) items.

        Returns:
            Parsed options
        """
        options = self._convert_config_items(items)
        return self._parse_options(options, self.dictionary)

    def add_parameter(self, param, **kwargs):
        """ Adds an parameter to the parser.

        If you want it to be an parameter of a sub-dictionary add
        the 'loc=subdict.subdict' keyword to the input.

        Args:
            param: Argument to add (either of object of class argument or string defining the name)
            kwargs: Any of the argument-fields (apart from 'name') and/or 'loc'

        Returns:
            This object
        """
        loc = kwargs.pop('loc', None)
        if not isinstance(param, Parameter):
            param = Parameter(param, **kwargs)
        self._add_param_to_dict(param, loc)
        return self

    def add_argument_dict(self, dictionary, loc):
        """ Appends a complete subdictionary to existing argument structure at node 'loc'.

        Args:
            loc: locination of the node to append the sub-dictionary
            dictionary: The dictionary to append

        Returns:
            This object
        """
        fields = loc.split('.')
        name = fields[-1]
        sub_dict = self._traverse_dict('.'.join(fields[:-1]))

        if name in sub_dict:
            raise ParameterError("'{:s}' already exists in parser!".format(name))

        self._validate_parameters(dictionary)
        sub_dict[name] = dictionary
        return self

    def help(self):
        # TODO: Print Help-Message
        pass

    def tree(self):
        """ Prints the current Parameter-Tree (I made dis :) ) """
        def print_tree(tree, level_char):
            for i, key in enumerate(sorted(tree.keys())):
                if i == len(tree) - 1:
                    node_char = _TC['L'] + _TC['-']
                    level_char_pp = level_char + '   '
                else:
                    node_char = _TC['S'] + _TC['-']
                    level_char_pp = level_char + _TC['|'] + '  '

                LOG.info(u"{:s}{:s} {:s}".format(level_char, node_char, key))
                if isinstance(tree[key], dict):

                    print_tree(tree[key], level_char_pp)
                else:
                    leaf = tree[key]
                    LOG.info(u"{:s}{:s} {:s}: {:s}".format(
                             level_char_pp, _TC['S'] + _TC['-'],
                             'Required', str(leaf.required)))
                    LOG.info(u"{:s}{:s} {:s}: {:s}".format(
                             level_char_pp, _TC['S'] + _TC['-'],
                             'Default', str(leaf.default)))
                    LOG.info(u"{:s}{:s} {:s}: {:s}".format(
                             level_char_pp, _TC['S'] + _TC['-'],
                             'Type', leaf.type.__name__ if leaf.type else 'None'))
                    LOG.info(u"{:s}{:s} {:s}: {:s}".format(
                             level_char_pp, _TC['S'] + _TC['-'],
                             'Choices', str(leaf.choices)))
                    LOG.info(u"{:s}{:s} {:s}: {:s}".format(
                             level_char_pp, _TC['L'] + _TC['-'],
                             'Help', leaf.help))

        LOG.info('Parameter Dictionary')
        print_tree(self.dictionary, '')

    #########################
    # Private Methods
    #########################

    def _add_param_to_dict(self, param, loc=None):
        """ Adds and parameter to the parameter dictionary.

        These will be used to parse an incoming option structure.

        Args:
            param: Argument to add
            loc: Path to sub-dictionary as string (e.g. subdict.subdict.loc[.arg])

        Returns:
            This object
        """
        sub_dict = self._traverse_dict(loc)
        if param.name in sub_dict:
            raise ParameterError("'{:s}' already exists in parser!".format(param.name))
        sub_dict[param.name] = param
        return self

    def _traverse_dict(self, loc=None):
        """ Traverses the dictionary to the subdict defined by loc.

        Adds non-existing substructures automatically.

        Args:
            loc: Path to sub-dictionary as string (e.g. argument.subparam.locination)

        Returns:
            Sub-dictionary
        """
        d = self.dictionary
        if loc:
            traverse = loc.split('.')
            for i, t in enumerate(traverse):
                try:
                    d = d[t]
                except KeyError:
                    d[t] = {}
                    d = d[t]
                if isinstance(d, Parameter):
                    raise ParameterError(
                        "'{:s}' is already an argument and hence cannot be a subdict.".format(
                            '.'.join(traverse[:i] + [t])))
        return d

    def _convert_config_items(self, items):
        """ Converts items list to a dictionary with types already in place """
        def list_check(value, level):
            s = value.replace(" ", "")
            if not (s.startswith("[" * (level+1)) or s.startswith(("["*level) + "range")):
                value = "[" + value + "]"
            return value

        def evaluate(name, item):
            try:
                return eval(item)  # sorry for using that
            except (NameError, SyntaxError):
                raise ArgumentError(
                    "Could not evaluate argument '{:s}', unknown '{:s}'".format(name, item))

        def eval_type(my_type, item):
            if issubclass(my_type, six.string_types):
                return my_type(item.strip("\'\""))
            if issubclass(my_type, bool):
                return bool(eval(item))
            else:
                return my_type(item)

        out = {}
        for name, value in items:
            if name in self.dictionary:
                arg = self.dictionary[name]
                if arg.type == list:
                    value = list_check(value, level=0)
                    if arg.subtype == list:
                        value = list_check(value, level=1)
                    value = evaluate(name, value)
                    if arg.subtype:
                        for idx, entry in enumerate(value):
                            value[idx] = eval_type(arg.subtype, entry)
                elif arg.type:
                    value = eval_type(arg.type, value)
                else:
                    value = evaluate(name, value)
                out[name] = value
            else:
                # could check self.strict here, but result is passed to get checked anyway
                out[name] = evaluate(name, value)
        return out


# Script Mode ##################################################################


if __name__ == '__main__':
    raise EnvironmentError("{:s} is not supposed to run as main.".format(__file__))
