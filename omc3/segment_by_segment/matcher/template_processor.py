import os
import logging
from omc3.model import manager
from omc3 import madx_wrapper

LOGGER = logging.getLogger(__name__)

DEFAULT_TEMPLATE_PATH = os.path.join(
    "..", "..", "madx", "templates", "lhc_super_matcher.madx"
)

EXTRACT_SEQUENCES_TEMPLATE = """
!!!! Extract sequence for matcher {matcher_name} !!!!
SEQEDIT, SEQUENCE={base_seq};
FLATTEN;
CYCLE, START={start_from};
ENDEDIT;

EXTRACT, SEQUENCE={base_seq},
         FROM={start_from}, TO={end_at},
         NEWNAME={new_seq};

SEQEDIT, SEQUENCE={new_seq};
FLATTEN;
IF ({is_back} == 1) {{
    REFLECT; ! reverse command
}}
ENDEDIT;

EXEC, BEAM_{base_seq}({new_seq});
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

SET_INITIAL_VALUES_TEMPLATE = """

!!!!!! Matcher {matcher_name} initial values !!!!!!
OPTION, -ECHO, -INFO;
CALL, FILE = "{modifiers_optics}";
OPTION, ECHO, INFO;

CALL, FILE = "{values_file}";

IF({is_back} == 0){{
    USE, PERIOD={segment_seq}, RANGE={startfrom}/{startfrom};
    SAVEBETA, LABEL={bininame}, PLACE={startfrom};
    TWISS, CHROM, betx=betx_ini, alfx=alfx_ini, bety=bety_ini, alfy=alfy_ini,
                  dx=dx_ini, dy=dy_ini, dpx=dpx_ini, dpy=dpy_ini,
                  wx=wx_ini, phix=phix_ini, wy=wy_ini, phiy=phiy_ini,
                  r11=ini_r11 ,r12=ini_r12, r21=ini_r21, r22=ini_r22;
    EXEC, TWISS_SEGMENT({segment_seq},
                        "{matcher_path}/twiss_{label}.dat",
                        {bininame});
}}ELSE{{
    USE, PERIOD={segment_seq}, RANGE={endat}/{endat};
    SAVEBETA, LABEL={bininame}, PLACE={endat};
    TWISS, CHROM, betx=betx_end, alfx=alfx_end, bety=bety_end, alfy=alfy_end,
                  dx=dx_end, dy=dy_end, dpx=dpx_end, dpy=dpy_end,
                  wx=wx_end, phix=phix_end, wy=wy_end, phiy=phiy_end,
                  r11=end_r11 ,r12=end_r12, r21=end_r21, r22=end_r22;
    EXEC, TWISS_SEGMENT({segment_seq},
                        "{matcher_path}/twiss_{label}_back.dat",
                        {bininame});
}}
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

START_MATCH = "match, use_macro;"
MATCHING_MACRO_TEMPLATE = """

!!!!!! Matcher {matcher_name} macro !!!!!!
{macro_name}: MACRO = {{
USE, PERIOD={segment_seq};

OPTION, -ECHO, -INFO;
CALL, FILE = "{modifiers_optics}";

{update_variables}

TWISS, BETA0={bininame}, CHROM;

{update_constraints}

OPTION, ECHO, INFO;
PRINT, TEXT = "=========== step for {macro_name} ===========";
}};
{define_constraints}
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

"""
END_MATCH = ("lmdif, tolerance:=1e-34, calls:=12000;\n"
             "endmatch;")

RUN_CORRECTED_TWISS_TEMPLATE = """
OPTION, -ECHO, -INFO;
CALL, FILE = "{modifiers_optics}";
{apply_corrections}
OPTION, ECHO, INFO;
if ({is_back} == 0) {{
    EXEC, TWISS_SEGMENT({segment_seq},
                        "{matcher_path}/twiss_{label}_cor.dat",
                        {bininame});
}}else{{
    EXEC, TWISS_SEGMENT({segment_seq},
                        "{matcher_path}/twiss_{label}_cor_back.dat",
                        {bininame});
}}
"""


SAVE_CHANGEPARAMETERS = 'SAVE, FILE="{match_path}/changeparameters.madx";'
CALL_CHANGEPARAMETERS = 'CALL, FILE="{match_path}/changeparameters.madx";'


class TemplateProcessor(object):

    def __init__(self, matchers_list, match_path, lhc_mode, minimize):
        self._matchers_list = matchers_list
        self._match_path = match_path
        self._accel_cls = manager.get_accel_class(accel="lhc",
                                                  lhc_mode=lhc_mode)
        self._minimize = minimize
        self._set_up_collections()
        self._log_file = os.path.join(match_path, "match_madx_log.out")
        self._output_file = os.path.join(match_path, "resolved_madx_match.madx")
        tmpl_path = os.path.join(os.path.dirname(__file__),
                                 "lhc_general_matcher.madx")
        with open(tmpl_path, "r") as tmpl_file:
            self._main_template = tmpl_file.read()

    def _set_up_collections(self):
        self._variables = set()
        self._extract_sequences_list = []
        self._set_initial_values_list = []
        self._aux_var_definition_list = []
        self._define_variables_list = []
        self._set_matching_macros_list = []
        self._gen_changeparameters_list = []
        self._run_corrected_twiss_list = []

    def run(self):
        self._process_matchers()
        madx_script = self._main_template.format(
            LIB=self._accel_cls.MACROS_NAME,
            MAIN_SEQ=self._accel_cls.load_main_seq_madx(),
            EXTRACT_SEQUENCES="\n".join(self._extract_sequences_list),
            SET_INITIAL_VALUES="\n".join(self._set_initial_values_list),
            DEFINE_CONSTRAINTS_AUX_VALS="\n".join(self._aux_var_definition_list),
            START_MATCH=START_MATCH,
            DEFINE_VARIABLES="\n".join(self._define_variables_list),
            SET_MATCHING_MACROS="\n".join(self._set_matching_macros_list),
            END_MATCH=END_MATCH,
            GEN_CHANGEPARAMETERS="\n".join(self._gen_changeparameters_list),
            SAVE_CHANGEPARAMETERS=SAVE_CHANGEPARAMETERS.format(match_path=self._match_path),
            RUN_CORRECTED_TWISS="\n".join(self._run_corrected_twiss_list),
        )
        madx_wrapper.resolve_and_run_string(madx_script,
                                            output_file=self._output_file,
                                            log_file=self._log_file)

    def run_just_twiss(self):
        for matcher in self._matchers_list:
            self._extract_sequences(matcher)
            self._set_initial_values(matcher)
            self._define_aux_vars(matcher)
            self._run_corrected_twiss(matcher)
        madx_script = self._main_template.format(
            LIB=self._accel_cls.MACROS_NAME,
            MAIN_SEQ=self._accel_cls.load_main_seq_madx(),
            EXTRACT_SEQUENCES="\n".join(self._extract_sequences_list),
            SET_INITIAL_VALUES="\n".join(self._set_initial_values_list),
            DEFINE_CONSTRAINTS_AUX_VALS="\n".join(self._aux_var_definition_list),
            START_MATCH="",
            DEFINE_VARIABLES="\n".join(self._define_variables_list),
            SET_MATCHING_MACROS="\n".join(self._set_matching_macros_list),
            END_MATCH="",
            GEN_CHANGEPARAMETERS="\n".join(self._gen_changeparameters_list),
            SAVE_CHANGEPARAMETERS=CALL_CHANGEPARAMETERS.format(match_path=self._match_path),
            RUN_CORRECTED_TWISS="\n".join(self._run_corrected_twiss_list),
        )
        madx_wrapper.resolve_and_run_string(madx_script,
                                            output_file=self._output_file,
                                            log_file=self._log_file)

    def _process_matchers(self):
        for matcher in self._matchers_list:
            LOGGER.info("Processing matcher: " + matcher.name)
            LOGGER.info(" - Ignoring variables: " +
                        str(matcher.excluded_variables))
            LOGGER.info(" - Ignoring constraints: " +
                        str(matcher.excluded_constraints))
            self._collect_variables(matcher)
            self._extract_sequences(matcher)
            self._set_initial_values(matcher)
            self._define_aux_vars(matcher)
            self._set_matching_macros(matcher)
            self._generate_changeparameters()
            self._run_corrected_twiss(matcher)
        self._define_variables()

    def _extract_sequences(self, matcher):
        extract_sequences_str = ""
        beam = matcher.segment.get_beam()
        extract_sequences_str += EXTRACT_SEQUENCES_TEMPLATE.format(
            matcher_name=matcher.name,
            base_seq="lhcb{}".format(beam),
            new_seq=matcher.get_sequence_name(),
            is_back=0 if matcher.propagation == "f" else "1",
            start_from=matcher.segment.start.name,
            end_at=matcher.segment.end.name,
        )
        self._extract_sequences_list.append(extract_sequences_str)

    def _set_initial_values(self, matcher):
        set_initial_values_str = ""
        matcher_path = os.path.join(matcher.matcher_path, "sbs")
        set_initial_values_str += SET_INITIAL_VALUES_TEMPLATE.format(
            matcher_name=matcher.name,
            values_file=os.path.join(
                matcher_path,
                "measurement_{}.madx".format(matcher.label)
            ),
            modifiers_optics=matcher.segment.optics_file,
            segment_seq=matcher.get_sequence_name(),
            matcher_path=matcher_path,
            label=matcher.segment.label,
            is_back=0 if matcher.propagation == "f" else "1",
            startfrom=matcher.segment.start.name,
            endat=matcher.segment.end.name,
            bininame=matcher.get_initvals_name(),
        )
        self._set_initial_values_list.append(set_initial_values_str)

    def _define_aux_vars(self, matcher):
        self._aux_var_definition_list.append(matcher.define_aux_vars())

    def _collect_variables(self, matcher):
        for variable in matcher.get_variables():
            self._variables.add(variable)

    def _define_variables(self):
        def_variables_string = ""
        for variable in self._variables:
            def_variables_string += '    vary, name=d' + variable.replace("->", "")
            def_variables_string += ', step := 1e-6;\n'
        self._define_variables_list.append(def_variables_string)

    def _set_matching_macros(self, matcher):
        matching_macros = ""
        define_constr_str = matcher.define_constraints()
        if self._minimize:
            define_constr_str += self._minimize_variables()
        matching_macros += MATCHING_MACRO_TEMPLATE.format(
            matcher_name=matcher.name,
            modifiers_optics=matcher.segment.optics_file,
            macro_name=matcher.get_macro_name(),
            segment_seq=matcher.get_sequence_name(),
            update_constraints=matcher.update_constraints_values(),
            update_variables=matcher.update_variables_definition(),
            bininame=matcher.get_initvals_name(),
            define_constraints=define_constr_str,
        )
        matching_macros += "\n"
        self._set_matching_macros_list.append(matching_macros)

    def _minimize_variables(self):
        variables = self._variables
        minimize_vars_str = ""
        minimize_vars_str += '    constraint, weight = 1.0, expr = sqrt('
        for variable in variables:
            minimize_vars_str += variable + "^2 +"
        minimize_vars_str = minimize_vars_str[:-1]
        minimize_vars_str += ') = 0.0; \n'
        return minimize_vars_str

    def _generate_changeparameters(self):
        changeparameters_str = ""
        for variable in self._variables:
            changeparameters_str += ('select, flag=save, '
                                     'pattern=\"d{}\";\n'.format(variable.replace("->", "")))
        self._gen_changeparameters_list.append(changeparameters_str)

    def _run_corrected_twiss(self, matcher):
        run_corrected_twiss_str = ""

        apply_correction_str = ""
        for variable in self._variables:
            apply_correction_str += (
                '{variable} = {matcher_name}.{variable}_0 + d{variable};\n'
            ).format(matcher_name=matcher.name, variable=variable)

        run_corrected_twiss_str += RUN_CORRECTED_TWISS_TEMPLATE.format(
            modifiers_optics=matcher.segment.optics_file,
            apply_corrections=apply_correction_str,
            matcher_path=os.path.join(matcher.matcher_path, "sbs"),
            label=matcher.segment.label,
            is_back=0 if matcher.propagation == "f" else "1",
            segment_seq=matcher.get_sequence_name(),
            bininame=matcher.get_initvals_name(),
        )
        self._run_corrected_twiss_list.append(run_corrected_twiss_str)
