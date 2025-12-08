import os
import sys
import datetime
import pandas as pd
from omc3.segment_by_segment.matcher import (
    amplitude,
    coupling,
    kmod,
    phase,
)
from omc3.segment_by_segment.matcher.template_processor import TemplateProcessor
from omc3 import sbs_propagation
from utils.contexts import silence
import tfs
import logging

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

MATCHER_TYPES = {
    "phase": phase.PhaseMatcher,
    "coupling": coupling.CouplingMatcher,
    "kmod": kmod.KmodMatcher,
    "amp": amplitude.AmpMatcher,
}


LOGGER = logging.getLogger(__name__)


def start_matching(lhc_mode, match_path, minimize, matchers_list):
    LOGGER.info("+++ Segment-by-segment general matcher +++")
    input_data = InputData(
        lhc_mode,
        match_path,
        minimize,
        matchers_list
    )
    # log_handler.add_file_handler(input_data.match_path)
    run_full_madx_matching(input_data)


def _run_madx_matching(input_data, just_twiss=False):
    template_processor = TemplateProcessor(input_data.matchers,
                                           input_data.match_path,
                                           input_data.lhc_mode,
                                           input_data.minimize, )
    if not just_twiss:
        template_processor.run()
    else:
        template_processor.run_just_twiss()


def run_full_madx_matching(input_data):
    _run_madx_matching(input_data)
    _write_sbs_data_for_matchers(input_data)
    _build_changeparameters_file(input_data)


def run_twiss_and_sbs(input_data):
    _run_madx_matching(input_data, just_twiss=True)
    _write_sbs_data_for_matchers(input_data)
    _build_changeparameters_file(input_data)


def _write_sbs_data_for_matchers(input_data):
    with silence():
        for this_matcher in input_data.matchers:
            _write_sbs_data(
                this_matcher.segment,
                this_matcher.matcher_path,
            )


def _write_sbs_data(segment_inst, temporary_path):
    save_path = os.path.join(temporary_path, "sbs")
    input_data = SegmentBySegmentMain._InputData(temporary_path)
    prop_models = SegmentBySegmentMain._PropagatedModels(
        save_path,
        segment_inst.label,
        '',
    )
    SegmentBySegmentMain.getAndWriteData(
        segment_inst.label, input_data, None, prop_models, save_path,
        False, False, True, False,
        segment_inst,
        None, None, None, "",
    )


def _build_changeparameters_file(input_data):
    original_file =\
        os.path.join(input_data.match_path, "changeparameters.madx")
    output_dir = os.path.join(input_data.match_path, "results")
    os.mkdir(output_dir)
    vars_dict = OrderedDict()
    with open(original_file, "r") as original_file_data:
        for original_line in original_file_data:
            parts = original_line.split("=")
            variable_name = parts[0].replace("d", "", 1).strip()
            variable_value = float(parts[1].replace(";", "").strip())
            vars_dict[variable_name] = variable_value
    tfs.write(
        os.path.join(output_dir, "changeparameters.tfs"),
        pd.DataFrame(data={"NAME": vars_dict.keys(),
                           "DELTA": vars_dict.values()}).loc[:, ["NAME", "DELTA"]],
        headers_dict={"DATE": datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")},
    )
    changeparameters_correct =\
        os.path.join(output_dir, "changeparameters_correct.madx")
    changeparameters_match =\
        os.path.join(output_dir, "changeparameters.madx")
    with open(changeparameters_correct, "w") as correct_data,\
         open(changeparameters_match, "w") as match_data:
        for varname in vars_dict:
            value = vars_dict[varname]
            sign = "+" if value >= 0 else "-"
            sign_correct = "-" if value >= 0 else "+"  # Flip sign to correct
            correct_data.write(
                "{name} = {name} {sign} {value};\n"
                .format(name=varname, sign=sign_correct, value=abs(value))
            )
            match_data.write(
                "{name} = {name} {sign} {value};\n"
                .format(name=varname, sign=sign, value=abs(value))
            )


class InputData():

    def __init__(self, lhc_mode, match_path, minimize, matchers_list):
        self.lhc_mode = lhc_mode
        self.match_path = match_path
        self.matchers = matchers_list
        self.minimize = minimize

    def _get_matchers_list(self, input_data):
        raw_matchers_list = input_data["matchers"]
        for matcher_name, matcher_data in raw_matchers_list.iteritems():
            matcher_type = matcher_data["type"]
            matcher_beam = matcher_data["beam"]
            matcher_variables = matcher_data["variables"]
            MatcherClass = MATCHER_TYPES.get(matcher_type, None)
            if MatcherClass is None:
                raise ValueError('Unknown matcher type: ' + matcher_type +
                                 ' must be in: ' + str(MATCHER_TYPES.keys()))
            self.matchers.append(
                MatcherClass(self.lhc_mode, matcher_beam,
                             matcher_name, matcher_data, matcher_variables,
                             self.match_path)
            )

    def _check_and_assign_attribute(self, input_data, attribute_name):
        setattr(self, attribute_name, input_data[attribute_name])

    # This transforms annoying unicode string into common byte string
    @staticmethod
    def _byteify(input_data):
        if isinstance(input_data, dict):
            return dict([(InputData._byteify(key), InputData._byteify(value))
                         for key, value in input_data.iteritems()])
        elif isinstance(input_data, list):
            return [InputData._byteify(element) for element in input_data]
        elif isinstance(input_data, str):
            return input_data.encode('utf-8')
        else:
            return input_data


def _parse_args():
    if len(sys.argv) >= 2:
        first_arg = sys.argv[1]
        if first_arg == "gui":
            LOGGER.info("Running GUI...")
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "gui", help="Run GUI mode.",
                type=str,
            )
            parser.add_argument(
                "--lhc_mode", help="LHC mode.",
                type=str,
            )
            parser.add_argument(
                "--input", help="Beta-beating GUI output directory.",
                type=str,
            )
            parser.add_argument(
                "--match", help="Match ouput directory.",
                type=str,
            )
            args = parser.parse_args()
            _run_gui(args.lhc_mode, args.match, args.input)
        elif os.path.isfile(first_arg):
            LOGGER.info("Given input is a file, matching from JSON file...")
            main(os.path.abspath(first_arg))
    elif len(sys.argv) == 1:
        LOGGER.info("No given input, running GUI...")
        _run_gui()


if __name__ == "__main__":
    # log_handler.set_up_console_logger(logging.getLogger(""))
    _parse_args()
