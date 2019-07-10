"""
dict_parser
---------------------------
"""

import argparse
import copy

from utils.dict_tools import DotDict, _TC

from utils import logging_tools
LOG = logging_tools.get_logger(__name__)


# Parser #######################################################################


class DictParser(object):
    """ Provides functions to parse a dictionary.

    First, build a dictionary structure with Parameters or Parameter-like dicts as
    leafs via add_parameter or on init.
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
        # Helper ------------------------------------------
        def _check_key(key, param):
            """ Checks if key coincides with param.name. """
            if key != param.name:
                raise ParameterError(f"'{key:s}': Key and name need to be the same.")

        # Main --------------------------------------------
        if len(dictionary) == 0:
            raise ParameterError()

        for key in dictionary:
            param = dictionary[key]
            if isinstance(param, dict):
                try:
                    DictParser._validate_parameters(param)
                except ParameterError as e:
                    # Build error message recursively, to find key in structure
                    if len(e.args):
                        e.args = (f"'{key}.{e.args[0][1:]:s}",)
                        raise
                    raise ParameterError(f"'{key}' is not a valid entry.")
            elif not isinstance(param, Parameter):
                raise ParameterError(f"'{key}' is not a valid entry.")
            else:
                _check_key(key, param)

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
                raise ArgumentError(f"'{key:s}' required in options.\n"
                                    f"Help: {param.help:s}")
            return param.default

        opt = arg_dict[key]
        if opt is None:
            if param.required:
                raise ArgumentError(f"'{key:s}' required in options.\n"
                                    f"Help: {param.help:s}")

        if param.type and not isinstance(opt, param.type):
            raise ArgumentError(f"'{key:s}' is not of type {param.type.__name__:s}.\n"
                                f"Help: {param.help:s}")

        if param.type == list:
            if param.nargs:
                if isinstance(param.nargs, int) and not param.nargs == len(opt):
                    raise ArgumentError(f"'{key:s}' should be list of length {param.nargs:d},"
                                        f" instead it was of length {len(opt):d}.\n"
                                        f"Help: {param.help:s}")

                if param.nargs == argparse.ONE_OR_MORE and not len(opt):
                    raise ArgumentError(f"'{key:s}' should be list of length >= 1,"
                                        f" instead it was of length {len(opt):d}.\n"
                                        f"Help: {param.help:s}")

            if param.subtype:
                for idx, item in enumerate(opt):
                    if not isinstance(item, param.subtype):
                        raise ArgumentError(f"Item {idx:d} of '{key:s}'"
                                            f" is not of type '{param.subtype.__name__:s}'.\n"
                                            f"Help: {param.help:s}")

            if param.choices and any([o for o in opt if o not in param.choices]):
                raise ArgumentError(f"All elements of '{key:s}' need to be one of "
                                    f"'{param.choices:}', instead the list was {opt:s}.\n"
                                    f"Help: {param.help:s}")

        elif param.choices and opt not in param.choices:
            raise ArgumentError(f"'{key:s}' needs to be one of '{param.choices:}', "
                                f"instead it was {opt:s}.\n"
                                f"Help: {param.help:s}")
        return opt

    def _parse_arguments(self, arg_dict, param_dict):
        """ Use parse_arguments()!

        This is a helper Function for parsing arguments. It does all the work. Called recursively.

        Args:
            arg_dict: Dictionary with the input arguments
            param_dict: Dictionary with the parameters to check the parameter against
        Returns:
            Dictionary with parsed arguments, i.e. the options
        """
        checked_dict = DotDict()
        for key in param_dict:
            if isinstance(param_dict[key], Parameter):
                checked_dict[key] = DictParser._check_value(key, arg_dict, param_dict)
            elif isinstance(param_dict[key], dict):
                try:
                    if not arg_dict or not (key in arg_dict):
                        checked_dict[key] = self._parse_arguments({}, param_dict[key])[0]
                    else:
                        checked_dict[key] = self._parse_arguments(arg_dict[key], param_dict[key])[0]
                except ArgumentError as e:
                    old_msg = ""
                    if len(e.args):
                        old_msg = e.args[0][1:]
                    if old_msg.startswith("'"):
                        e.args = (f"'{key}.{old_msg[1:]:s}",)
                    else:
                        e.args = (f"'{key}' has {old_msg:s}",)
                    raise

            arg_dict.pop(key, None)  # Default value avoids KeyError

        if len(arg_dict) > 0:
            error_message = f"Unknown Options: '{list(arg_dict.keys())}'."
            if self.strict:
                raise ArgumentError(error_message)
            LOG.debug(error_message)

        return checked_dict, arg_dict

    #########################
    # Public Methods
    #########################

    def parse_arguments(self, arguments):
        """ Parse a given argument dictionary and return parsed options.

        Args:
            arguments: Arguments to parse

        Return:
            Options [, Unknown Options]
        """
        checked = self._parse_arguments(copy.deepcopy(arguments), self.dictionary)
        if self.strict:
            return checked[0]
        return checked

    def parse_config_items(self, items):
        """ Parse a list of (name, value) items, where the values are all strings.

        Args:
            items: list of (name, value) items.

        Returns:
            Parsed options
        """
        options = self._convert_config_items(items)
        return self.parse_arguments(options)

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

    def add_parameter_dict(self, dictionary, loc):
        """ Appends a complete subdictionary to existing argument structure at node 'loc'.

        Args:
            loc: location of the node to append the sub-dictionary
            dictionary: The dictionary to append

        Returns:
            This object
        """
        fields = loc.split('.')
        name = fields[-1]
        sub_dict = self._traverse_dict('.'.join(fields[:-1]))

        if name in sub_dict:
            raise ParameterError(f"'{name}' already exists in parser!")

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
                LOG.info(f"{level_char:s}{node_char:s} {key:s}")
                if isinstance(tree[key], dict):
                    print_tree(tree[key], level_char_pp)
                else:
                    leaf = tree[key]
                    LOG.info(f"{level_char_pp + _TC['S'] + _TC['-']:s}"
                             f" Required: {leaf.required}")

                    LOG.info(f"{level_char_pp + _TC['S'] + _TC['-']:s}"
                             f" Default: {leaf.default}")

                    LOG.info(f"{level_char_pp + _TC['S'] + _TC['-']:s}"
                             f" Type: {leaf.type.__name__ if leaf.type else 'None'}")

                    LOG.info(f"{level_char_pp + _TC['S'] + _TC['-']:s}"
                             f" Choices: {leaf.choices}")

                    LOG.info(f"{level_char_pp + _TC['L'] + _TC['-']:s}"
                             f" Help: {leaf.help:s}")
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
            raise ParameterError(f"'{param.name:s}' already exists in parser!")
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
        def evaluate(name, item):
            try:
                return eval(item)  # sorry for using that
            except (NameError, SyntaxError):
                raise ArgumentError(f"Could not evaluate argument '{name:s}', unknown '{item:s}'")

        def eval_type(my_type, item):
            if issubclass(my_type, str):
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


# Helper Classes ###############################################################


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
            ParameterError(f"'{kwargs.keys()}' are not valid parameters for Argument.")

        self._validate()

    def _validate(self):
        if not isinstance(self.name, str):
            raise ParameterError(f"Parameter '{self.name}': " +
                                 "Name is not a valid string.")

        if self.default and self.type and not isinstance(self.default, self.type):
            raise ParameterError(f"Parameter '{self.name:s}': " +
                                 "Default value not of specified type.")

        if self.subtype and not (self.type or self.type == list):
            raise ParameterError(f"Parameter '{self.name:s}': " +
                                 "field 'subtype' is only accepted if 'type' is list.")

        if self.nargs:
            if (not isinstance(self.nargs, int) and
                    self.nargs not in [argparse.ONE_OR_MORE, argparse.ZERO_OR_MORE]):
                raise ParameterError(f"Parameter '{self.name:s}': "
                                     "nargs needs to be an integer or either "
                                     f"'{argparse.ONE_OR_MORE}' or '{argparse.ZERO_OR_MORE}'. "
                                     f"Instead it was '{self.nargs}'")

            if not (self.type or self.type == list):
                raise ParameterError(f"Parameter '{self.name:s}': " +
                                     "'type' needs to be 'list' if 'nargs' is given.")

        if self.choices:
            try:
                [choice for choice in self.choices]
            except TypeError:
                raise ParameterError(f"Parameter '{self.name:s}': " +
                                     "'Choices' need to be iterable.")

            if self.default:
                if self.type == list:
                    not_a_choice = [d for d in self.default if d not in self.choices]

                    if len(not_a_choice) > 0:
                        raise ParameterError(f"Parameter '{self.name:s}': " +
                                             f"Default value(s) '{str(not_a_choice)}'"
                                             " not found in choices.")
                else:
                    if self.default not in self.choices:
                        raise ParameterError(f"Parameter '{self.name:s}': " +
                                             "Default value not found in choices.")

            if self.type or self.subtype:
                if self.nargs is None:
                    check = self.type if self.subtype is None else self.subtype
                else:
                    check = self.subtype

                if check is not None:
                    for choice in self.choices:
                        if not isinstance(choice, check):
                            raise ParameterError(f"Choice '{choice}' " +
                                                 f"of parameter '{self.name:s}': " +
                                                 f"is not of type '{check.__name__:s}'.")

        if self.required and self.default is not None:
            LOG.warning(f"Parameter '{self.name:s}': " +
                        "Value is required but default value is given. The latter will be ignored.")
