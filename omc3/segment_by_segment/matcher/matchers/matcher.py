# pylint: disable=E1101
import os
import logging
import shutil
from functools import partial
from omc3.segment_by_segment.matcher.segment_by_segment import SegmentBeatings
from omc3.utils import iotools
from omc3.model import manager

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
LOGGER = logging.getLogger(__name__)


class MatcherFactory(object):
    """
    Use this class to create matchers.
    The factory instance will have a set_{parameter} function for each of the
    items in the PARAMETERS constant. All parameters specified should be set
    or an MatcherFactoryError will be throws. The method create() will give
    the matcher instance.

    Args:
        matcher_like: A subclass of the Matcher class to instantiate.
    """
    PARAMETERS = (
        "lhc_mode", "beam", "name",
        "var_classes", "match_path", "label",
        "use_errors", "propagation", "measurement_path",
        "excluded_constraints", "excluded_variables",
    )

    def __init__(self, matcher_like):
        self._matcher_like = matcher_like.__new__(matcher_like)
        for parameter in MatcherFactory.PARAMETERS:
            setattr(
                self,
                "set_" + parameter,
                partial(self._set_function_tmpl, parameter),
            )

    def _set_function_tmpl(self, parameter, value):
        setattr(self, "_is_{}_set".format(parameter), True)
        setattr(self, "_{}".format(parameter), value)
        return self

    def create(self):
        """
        Checks that all the parameters have been set and creates the instance
        of the requested matcher.

        Returns:
            An instace of the required Matcher.
        """
        for parameter in MatcherFactory.PARAMETERS:
            if not getattr(self, "_is_{}_set".format(parameter)):
                raise MatcherFactoryError(
                    "Parameter {} not defined.".format(parameter)
                )
            setattr(self._matcher_like,
                    parameter,
                    getattr(self, "_{}".format(parameter)))
        self._matcher_like._create()
        return self._matcher_like


class MatcherFactoryError(Exception):
    """
    Thrown when the creation of a matcher using the MatcherFactory fails.
    """
    pass


class Matcher(object):

    def __init__(self):
        raise NotImplementedError(
            "Use the MatcherFactory to implement matcher classes."
        )

    def _create(self):
        assert self.propagation in ("front", "back", "f", "b")
        self.matcher_path = os.path.join(
            self.match_path,
            self.name,
        )
        Matcher._copy_measurement_files(
            self.label,
            self.measurement_path,
            self.matcher_path,
        )
        self.beatings = SegmentBeatings(os.path.join(self.matcher_path, "sbs"),
                                        self.label)
        self.segment = self._get_segment()
        self.propagation = self.propagation[0]
        self.ini_end = "ini" if self.propagation == "f" else "end"

    def get_variables(self, exclude=True):
        """
        Returns the variables names to use to match. If exclude is true,
        it will not return the variables in the excluded variables list.
        """
        raise NotImplementedError

    def get_sequence_name(self):
        """
        Returns the name of the sequence this matcher will generate.
        """
        return "seq_" + self.name

    def get_initvals_name(self):
        """
        Returns the name of the initial values variable this matcher will
        generate and use in: twiss, beta0=...;
        """
        return "bini_" + self.name

    def get_macro_name(self):
        """
        Returns the name of the macro this matcher will use to match.
        """
        return "match_" + self.name

    def define_aux_vars(self):
        """Returns the MAD-X string to define the auxiliary values to use
        during the matching"""
        raise NotImplementedError

    def define_constraints(self):
        """Returns two MAD-X strings to define the matching constraints for this
        matcher."""
        raise NotImplementedError

    def update_constraints_values(self):
        """Returns the MAD-X string that updates the value of the constraints to
        let MAD-X reevaluate in every iteration."""
        raise NotImplementedError

    def update_variables_definition(self):
        """Returns the MAD-X string that updates the definition of the variables
        that may have been override by the modifiers file."""
        raise NotImplementedError

    def generate_changeparameters(self):
        """Returns the MAD-X string that selects the variables to dump to the
        changeparameters file."""
        raise NotImplementedError

    def apply_correction(self):
        """Returns the MAD-X string that applies the final correction to the
        variables, in order to get the corrected twiss files"""
        raise NotImplementedError

    def _get_constraint_instruction(self, constr_name, value, error):
        weight = 1.0
        if self.use_errors:
            if error == 0.0:
                return("    ! Ignored constraint {}\n".format(constr_name))
            weight = 1 / error
        constr_string = '    constraint, weight = {weight}, '
        constr_string += 'expr = {constr_name} = {value};\n'
        constr_string = constr_string.format(
            constr_name=constr_name,
            value=value,
            weight=weight
        )
        return constr_string

    def _get_constraints_block(self, names, values, errors):
        constr_block = ""
        for name, value, error in zip(names, values, errors):
            weight = 1.0
            if self.use_errors:
                if error == 0.0:
                    constr_block += "    ! Ignored constraint {}\n".format(name)
                weight = 1. / error
            constr_line = '    constraint, weight = {weight}, '
            constr_line += 'expr = {name} = {value};\n'
            constr_block += constr_line.format(name=name,
                                               value=value,
                                               weight=weight)
        return constr_block

    def _get_nominal_table_name(self, beam=None):
        if beam is None:
            beam = self.segment.get_beam()
        return self.name + ".twiss.b" + str(beam)

    @staticmethod
    def override(parent_cls):
        """
        Decorator that checks if the decorated method overrides some method
        in the given parent class.
        Similar to java @override annotation.
        To use it:
        @Matcher.override(Superclass)
        def some_method():
            pass
        """
        def override_decorator(method):
            if not (method.__name__ in dir(parent_cls)):
                raise TypeError(method.__name__ + " must override a method from class " + parent_cls.__name__ + ".")
            else:
                method.__doc__ = getattr(parent_cls, method.__name__).__doc__
            return method
        return override_decorator

    def _get_segment(self):
        range_start, range_end = _get_match_bpm_range(self.beatings.phase["x"])
        LOGGER.info("Matching range for Beam " + str(self.beam) + ": " +
                    range_start + " " + range_end)
        accel_cls = manager.get_accel_class(
            accel="lhc", lhc_mode=self.lhc_mode, beam=self.beam
        )
        optics_file = os.path.join(
            os.path.join(self.matcher_path, "sbs"), "modifiers.madx"
        )
        segment = accel_cls.get_segment(self.label,
                                        range_start,
                                        range_end,
                                        optics_file,
                                        None)
        return segment

    @staticmethod
    def _copy_measurement_files(label, measurement_path, match_math):
        iotools.create_dirs(match_math)
        iotools.create_dirs(os.path.join(match_math, "sbs"))
        # GetLLM output files:
        _copy_files_with_extension(measurement_path,
                                   match_math, ".out")
        _copy_files_with_extension(measurement_path,
                                   match_math, ".dat")
        # SbS output files for the given label:
        _copy_files_which_contains(
            os.path.join(measurement_path, "sbs"),
            os.path.join(match_math, "sbs"),
            label
        )
        # SbS MAD-X files (not necessary but useful):
        _copy_files_with_extension(
            os.path.join(measurement_path, "sbs"),
            os.path.join(match_math, "sbs"),
            ".madx"
        )


def _copy_files_with_extension(src, dest, ext):
    _copy_files_with_filter(
        src, dest,
        lambda file_name: file_name.endswith(ext)
    )


def _copy_files_which_contains(src, dest, substring):
    _copy_files_with_filter(
        src, dest,
        lambda file_name: substring in file_name
    )


def _copy_files_with_filter(src, dest, filter_function):
    src_files = _get_filtered_file_list(src, filter_function)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        shutil.copy(full_file_name, dest)


def _get_filtered_file_list(src, filter_function):
    filtered_file_list = []
    original_file_list = os.listdir(src)
    for file_name in original_file_list:
        if (os.path.isfile(os.path.join(src, file_name)) and
                filter_function(file_name)):
            filtered_file_list.append(file_name)
    return filtered_file_list


def _get_match_bpm_range(some_sbs_df):
    return some_sbs_df.NAME.iloc[0], some_sbs_df.NAME.iloc[-1]
