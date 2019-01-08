""" Entry Point Decorator

Allows a function to be decorated as entrypoint.
This function will then automatically accept console arguments, config files, json files,
kwargs and dictionaries as input and will parse it according to the parameters
given to the entrypoint-Decorator.

Terminology:
++++++++++++++++++++++++

    * **Parameter** - Items containing info on how to parse Arguments
    * **Arguments** - The input to the wrapped-function
    * **Options** - The parsed arguments and hence the options of the function

Hence, an :class:`ArgumentError` will be raised in case of something going wrong during parsing,
while :class:`ParameterErrors` will be raised if something goes wrong when
adding parameters to the list.

Usage:
++++++++++++++++++++++++

To be used as a decorator::
    @entrypoint(parameters)
    def some_function(options, unknown_options)

Using **strict** mode (see below)::
    @entrypoint(parameters, strict=True)
    def some_function(options)

It is also possible to use the EntryPoint Class similar to a normal parser::
    ep_parser = EntryPoint(parameters)
    options, unknown_options = ep_parser.parse(arguments)

Using **strict** mode (see below)::
    ep_parser = EntryPoint(parameters, strict=True)
    options = ep_parser.parse(arguments)


Parameters:
++++++++++++++++++++++++

Parameters need to be a list or a dictionary of dictionaries with the following keys:

| **name** (*required*): Name of the variable (e.g. later use options.NAME).
 If 'params' is a dictionary, the key will be used as name.
| **flags** (*required*): Commandline flag(s), e.g. ``--file``
| **required** (*optional*): ``bool``
| **default** (*optional*): Default value, if variable not present
| **help** (*optional*): ``str``
| **type** (*optional*): Value ``type`` (if nargs is given, set to list for dicts!)
| **choices** (*optional*): choices to choose from
 (choices need to be of ``type``, if given)
| **nargs** (*optional*): number of arguments to consume
 (commandline only, do not use ``REMAINDER``!)
| **action** (*optional*): either ``store_true`` or ``store_false``, will set ``type`` to bool
 and the default to ``False`` and ``True`` respectively.


The **strict** option changes the behaviour for unknown parameters:
``strict=True`` raises exceptions, ``strict=False`` loggs debug messages and returns the options.
Hence a wrapped function with ``strict=True`` must accept one input, with ``strict=False`` two.
Default: ``False``

"""

import copy
import json
import argparse
from argparse import ArgumentParser
from configparser import ConfigParser
from inspect import getfullargspec
from functools import wraps

from utils import logging_tools as logtools
from utils.dict_tools import DictParser
from utils.dict_tools import DotDict
from utils.dict_tools import ArgumentError
from utils.dict_tools import ParameterError

from utils.contexts import silence


LOG = logtools.get_logger(__name__)


ID_CONFIG = "entry_cfg"
ID_DICT = "entry_dict"
ID_JSON = "entry_json"
ID_SECTION = "section"


# EntryPoint Class #############################################################


class EntryPoint(object):
    def __init__(self, parameter, strict=False):
        """ Initialize decoration: Handle the desired input parameter. """
        self.strict = strict

        # add argument dictionary to EntryPoint
        self.remainder = None
        self.parameter = EntryPoint._dict2list_param(parameter)
        self._check_parameter()

        # add config-argparser
        self.configarg = self._create_config_argument()

        # create parsers from parameter
        self.argparse = self._create_argument_parser()
        self.dictparse = self._create_dict_parser()
        self.configparse = self._create_config_parser()

    def parse(self, *args, **kwargs):
        """ Parse whatever input parameter come.

            This is the heart of EntryPoint and will recognize the input and parse it
            accordingly.
            Allowed inputs are:
                - Dictionary with arguments as key-values
                - Key-Value arguments
                - Path to a one-section config file
                - Commandline Arguments
                - Commandline Arguments in string-form (as list)
                - Special Key-Value arguments are:
                    entry_dict: Value is a dict of arguments
                    entry_cfg: Path to config file
                    entry_json: Path to json file
                    section: Section to use in config file, or subdirectory to use in json file.
                             Only works with the key-value version of config file.
                             If not given only one-section config files are allowed.
         """
        if len(args) > 0 and len(kwargs) > 0:
            raise ArgumentError("Cannot combine positional parameter with keyword parameter.")

        if len(args) > 1:
            raise ArgumentError("Only one positional argument allowed (dict or config file).")

        if args and args[0] is not None:
            # LOG.info("Entry input: {:s}".format(args[0]))  # activate for debugging
            options = self._handle_arg(args[0])
        elif len(kwargs) > 0:
            # LOG.info("Entry input: {:s}".format(kwargs))  # activate for debugging
            options = self._handle_kwargs(kwargs)
        else:
            # LOG.info("Entry input: {:s}".format(" ".join(sys.argv))  # activate for debugging
            options = self._handle_commandline()

        return options  # options might include known and unknown options

    #########################
    # Create Parsers
    #########################

    def _create_config_argument(self):
        """ Creates the config-file argument parser """
        parser = ArgumentParser()
        parser.add_argument('--{}'.format(ID_CONFIG), type=str, dest=ID_CONFIG, required=True,)
        parser.add_argument('--{}'.format(ID_SECTION), type=str, dest=ID_SECTION,)
        return parser

    def _create_argument_parser(self):
        """ Creates the ArgumentParser from parameter. """
        parser = ArgumentParser()
        parser = add_params_to_generic(parser, self.parameter)
        return parser

    def _create_dict_parser(self):
        """ Creates the DictParser from parameter. """
        parser = DictParser(strict=self.strict)
        parser = add_params_to_generic(parser, self.parameter)
        return parser

    def _create_config_parser(self):
        """ Creates the config parser. Maybe more to do here later with parameter. """
        parser = ConfigParser()
        return parser

    #########################
    # Handlers
    #########################

    def _handle_commandline(self, args=None):
        """ No input to function """
        try:
            # check for config file first
            with silence():
                options = self.configarg.parse_args(args)
        except SystemExit:
            # parse regular options
            options, unknown_opts = self.argparse.parse_known_args(args)
            options = DotDict(vars(options))
            if self.strict:
                if unknown_opts:
                    raise ArgumentError(f"Unknown options: {unknown_opts}")
                return options
            else:
                if unknown_opts:
                    LOG.debug(f"Unknown options: {unknown_opts}")
                return options, unknown_opts
        else:
            # parse config file
            return self.dictparse.parse_config_items(self._read_config(vars(options)[ID_CONFIG]))

    def _handle_arg(self, arg):
        """ *args has been input """
        if isinstance(arg, str):
            # assume config file
            options = self.dictparse.parse_config_items(self._read_config(arg))
        elif isinstance(arg, dict):
            # dictionary
            options = self.dictparse.parse_arguments(arg)
        elif isinstance(arg, list):
            # list of commandline parameter
            options = self._handle_commandline(arg)
        else:
            raise ArgumentError("Only dictionary or configfiles "
                                "are allowed as positional arguments")
        return options  # options might include known and unknown options

    def _handle_kwargs(self, kwargs):
        """ **kwargs been input """
        if ID_CONFIG in kwargs:
            if len(kwargs) > 2 or (len(kwargs) == 2 and ID_SECTION not in kwargs):
                raise ArgumentError(
                    "Only '{:s}' and '{:s}'".format(ID_CONFIG, ID_SECTION) +
                    " arguments are allowed, when using a config file.")
            options = self._read_config(kwargs[ID_CONFIG],
                                        kwargs.get(ID_SECTION, None))
            options = self.dictparse.parse_config_items(options)

        elif ID_DICT in kwargs:
            if len(kwargs) > 1:
                raise ArgumentError("Only one argument allowed when using a dictionary")
            options = self.dictparse.parse_arguments(kwargs[ID_DICT])

        elif ID_JSON in kwargs:
            if len(kwargs) > 2 or (len(kwargs) == 2 and ID_SECTION not in kwargs):
                raise ArgumentError(
                    "Only '{:s}' and '{:s}'".format(ID_JSON, ID_SECTION) +
                    " arguments are allowed, when using a json file.")
            with open(kwargs[ID_JSON], 'r') as json_file:
                json_dict = json.load(json_file)

            if ID_SECTION in kwargs:
                json_dict = json_dict[kwargs[ID_SECTION]]

            options = self.dictparse.parse_arguments(json_dict)

        else:
            options = self.dictparse.parse_arguments(kwargs)

        return options   # options might include known and unknown options

    #########################
    # Helpers
    #########################

    def _check_parameter(self):
        """ EntryPoint specific checks for parameter """
        for param in self.parameter:
            arg_name = param.get("name", None)
            if arg_name is None:
                raise ParameterError("A Parameter needs a Name!")

            if param.get("nargs", None) == argparse.REMAINDER:
                raise ParameterError("Parameter '{:s}' is set as remainder.".format(arg_name) +
                                     "This method is really buggy, hence it is forbidden.")

            if param.get("flags", None) is None:
                raise ParameterError("Parameter '{:s}'".format(arg_name) +
                                     "does not have flags.")

    def _read_config(self, cfgfile_path, section=None):
        """ Get content from config file"""
        cfgparse = self.configparse

        with open(cfgfile_path) as config_file:
            cfgparse.readfp(config_file)

        sections = cfgparse.sections()
        if not section and len(sections) == 1:
            section = sections[0]
        elif not section:
            raise ArgumentError("'{:s}' contains multiple sections. Please specify one!")

        return cfgparse.items(section)

    @staticmethod
    def _dict2list_param(param):
        """ Convert dictionary to list and add name by key """
        if isinstance(param, dict):
            out = []
            for key in param:
                item = param[key]
                item["name"] = key
                out.append(item)
            return out
        else:
            return param


# entrypoint Decorator #########################################################


class entrypoint(EntryPoint):
    """ Decorator extension of EntryPoint.

    Implements the __call__ method needed for decorating.
    Lowercase looks nicer if used as decorator """

    def __call__(self, func):
        """ Builds wrapper around the function 'func' (called on decoration)

        Whenever the decorated function is called, actually this wrapper is called.
        The Number of arguments is checked for compliance with instance- and class- methods,

        Meaning: if there is one more argument as there should be, we pass it on as it is
        (should be) either ``self`` or ``cls``.
        One could check that there are no varargs and keywords, but let's assume the user
        is doing the right things.
        """
        nargs = len(getfullargspec(func).args)

        if self.strict:
            if nargs == 1:
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(self.parse(*args, **kwargs))
            elif nargs == 2:
                @wraps(func)
                def wrapper(other, *args, **kwargs):
                    return func(other, self.parse(*args, **kwargs))
            else:
                raise ArgumentError("In strict mode, only one option-structure will be passed."
                                    " The entrypoint needs to have the following structure: "
                                    " ([self/cls,] options)."
                                    " Found: {:s}".format(getfullargspec(func).args))
        else:
            if nargs == 2:
                @wraps(func)
                def wrapper(*args, **kwargs):
                    options, unknown_options = self.parse(*args, **kwargs)
                    return func(options, unknown_options)
            elif nargs == 3:
                @wraps(func)
                def wrapper(other, *args, **kwargs):
                    options, unknown_options = self.parse(*args, **kwargs)
                    return func(other, options, unknown_options)
            else:
                raise ArgumentError("Two option-structures will be passed."
                                    " The entrypoint needs to have the following structure: "
                                    " ([self/cls,] options, unknown_options)."
                                    " Found: {:s}".format(getfullargspec(func).args))
        return wrapper


# EntryPoint Arguments #########################################################


class EntryPointParameters(DotDict):
    """ Helps to build a simple dictionary structure via add_argument functions.

    You really don't need that, but old habits die hard."""
    def add_parameter(self, **kwargs):
        """ Add parameter """
        name = kwargs.pop("name")
        if name in self:
            raise ParameterError("'{:s}' is already a parameter.".format(name))
        else:
            self[name] = kwargs

    def help(self):
        """ Prints current help. Usable to paste into docstrings. """
        optional_param = ""
        required_param = ""

        for name in sorted(self.keys()):
            item_str = ""
            item = self[name]
            try:
                name_type = "{n:s} ({t:s})".format(n=name, t=item["type"].__name__)
            except KeyError:
                name_type = "{n:s}".format(n=name)

            try:
                item_str += "{n:s}: {h:s}".format(n=name_type, h=item["help"])
            except KeyError:
                item_str += "{n:s}: -Help not available- ".format(n=name_type)

            space = " " * (len(name_type) + 2)

            try:
                item_str += f"\n{space:s}**Flags**: {item['flags']:s}"
            except KeyError:
                pass

            try:
                item_str += f"\n{space:s}**Choices**: {item['choices']:s}"
            except KeyError:
                pass

            try:
                item_str += f"\n{space:s}**Default**: ``{item['default']:s}``"
            except KeyError:
                pass

            try:
                item_str += "\n{s:s}**Action**: ``{d:s}``".format(s=space, d=str(item["action"]))
            except KeyError:
                pass

            if item.get("required", False):
                required_param += item_str + "\n"
            else:
                optional_param += item_str + "\n"

        if required_param:
            LOG.info("Required")
            LOG.info(required_param)

        if optional_param:
            LOG.info("Optional")
            LOG.info(optional_param)


# Public Helpers ###############################################################


def add_params_to_generic(parser, params):
    """ Adds entry-point style parameter to either
    ArgumentParser, DictParser or EntryPointArguments
    """
    params = copy.deepcopy(params)

    if isinstance(params, dict):
        params = EntryPoint._dict2list_param(params)

    if isinstance(parser, EntryPointParameters):
        for param in params:
            parser.add_parameter(param)

    elif isinstance(parser, ArgumentParser):
        for param in params:
            param["dest"] = param.pop("name", None)
            flags = param.pop("flags", None)
            if flags is None:
                parser.add_argument(**param)
            else:
                if isinstance(flags, str):
                    flags = [flags]
                parser.add_argument(*flags, **param)

    elif isinstance(parser, DictParser):
        for param in params:
            if "nargs" in param:
                if param["nargs"] != "?":
                    param["subtype"] = param.get("type", None)
                    param["type"] = list

                if isinstance(param["nargs"], str):
                    param.pop("nargs")

            if "action" in param:
                if param["action"] in ("store_true", "store_false"):
                    param["type"] = bool
                    param["default"] = not param["action"][6] == "t"
                else:
                    raise ParameterError("Action '{:s}' not allowed in EntryPoint")
                param.pop("action")

            param.pop("flags", None)

            name = param.pop("name")
            parser.add_parameter(name, **param)
    else:
        raise TypeError("Parser not recognised.")
    return parser


def split_arguments(args, *param_list):
    """ Divide remaining arguments into a list of argument-dicts,
        fitting to the params in param_list.

        Args:
            args: Input arguments, either as list of strings or dict
            param_list: list of sets of entry-point parameters (either dict, or list)

        Returns:
            A list of dictionaries containing the arguments for each set of parameters,
            plus one more entry for unknown parameters.
            If the input was a list of argument-strings, the parameters will already be parsed.

        .. warning:: Unless you know what you are doing, run this function only on
                     remaining-arguments from entry point parsing, not on the actual arguments

        .. warning:: Adds each argument only once, to the set of params who claim it first!
    """
    split_args = []
    if isinstance(args, list):
        # strings of commandline parameters, has to be parsed twice
        # (as I don't know how to handle flags properly)
        for params in param_list:
            parser = argparse.ArgumentParser()
            parser = add_params_to_generic(parser, params)
            this_args, args = parser.parse_known_args(args)
            split_args.append(DotDict(this_args.__dict__))
        split_args.append(args)
    else:
        # should be a dictionary of params, so do it the manual way
        for params in param_list:
            params = param_names(params)
            split_args.append(DotDict([(key, args.pop(key)) for key in args if key in params]))
        split_args.append(DotDict(args))
    return split_args


def param_names(params):
    """ Get the names of the parameters, no matter if they are a dict or list of dicts """
    try:
        names = params.keys()
    except AttributeError:
        names = [p["name"] for p in params]
    return names


class CreateParamHelp(object):
    """ Print params help quickly but changing the logging format first.

    Usage Example::

        import amplitude_detuning_analysis
        help = CreateParamHelp()
        help(amplitude_detuning_analysis)
        help(amplitude_detuning_analysis, "_get_plot_params")

    """
    def __init__(self):
        logtools.getLogger("").handlers = []  # remove all handlers from root-logger
        logtools.get_logger("__main__", fmt="%(message)s")  # set up new

    def __call__(self, module, param_fun=None):
        if param_fun is None:
            try:
                module.get_params().help()
            except AttributeError:
                module._get_params().help()
        else:
            getattr(module, param_fun)().help()
