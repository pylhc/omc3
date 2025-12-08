import numpy as np
from omc3.segment_by_segment.matcher.generic import Matcher

DEF_CONSTR_AUX_VALUES_TEMPLATE = """
use, period=%(SEQ)s;
twiss, beta0=%(INIT_VALS)s, chrom, table=%(TABLE_NAME)s;

%(S_VARIABLES)s
%(D_VARIABLES)s
"""

USE_ABS = False
USE_F1010 = False


class CouplingMatcher(Matcher):

    COUP_CORR_CLASSES = ["MQSX"]

    @Matcher.override(Matcher)
    def get_variables(self, exclude=True):
        variables = self.segment.get_segment_vars(
            classes=CouplingMatcher.COUP_CORR_CLASSES,
        )
        if exclude:
            variables = [
                var for var in variables
                if var not in self.excluded_variables
            ]
        return variables

    @Matcher.override(Matcher)
    def define_aux_vars(self):
        variables_s_str = ""
        variables_d_str = ""

        for variable in self.get_variables():
            variables_s_str += self.name + '.' + variable + '_0' + ' = ' + variable + ';\n'
            variables_d_str += variable + ' := ' + self.name + "." + variable + '_0' + ' + d' + variable + ';\n'

        return DEF_CONSTR_AUX_VALUES_TEMPLATE % {
            "SEQ": self.get_sequence_name(),
            "INIT_VALS": self.get_initvals_name(),
            "TABLE_NAME": self._get_nominal_table_name(),
            "S_VARIABLES": variables_s_str,
            "D_VARIABLES": variables_d_str,
        }

    @Matcher.override(Matcher)
    def define_constraints(self):
        sbs_data = self.beatings.coupling.copy()
        sbs_data = sbs_data[~sbs_data.index.isin(self.excluded_constraints)]
        sbs_data.loc[:, "F1001ABS"] = np.sqrt(sbs_data.F1001REMEAS ** 2 +
                                              sbs_data.F1001IMMEAS ** 2)
        sbs_data.loc[:, "F1010ABS"] = np.sqrt(sbs_data.F1010REMEAS ** 2 +
                                              sbs_data.F1010IMMEAS ** 2)

        constr_tpl = ""
        if USE_ABS:
            constr_tpl += '   constraint, weight = {weight}, '
            constr_tpl += 'expr = {cls_name}.{name}_f1001abs = {f1001abs}; \n'
            if USE_F1010:
                constr_tpl += '   constraint, weight = {weight}, '
                constr_tpl += 'expr = {cls_name}.{name}_f1010abs = {f1010abs}; \n'
        else:
            constr_tpl += '   constraint, weight = {weight}, '
            constr_tpl += 'expr = {cls_name}.{name}_f1001r = {f1001r}; \n'
            constr_tpl += '   constraint, weight = {weight}, '
            constr_tpl += 'expr = {cls_name}.{name}_f1001i = {f1001i}; \n'
            if USE_F1010:
                constr_tpl += '   constraint, weight = {weight}, '
                constr_tpl += 'expr = {cls_name}.{name}_f1010r = {f1010r}; \n'
                constr_tpl += '   constraint, weight = {weight}, '
                constr_tpl += 'expr = {cls_name}.{name}_f1010i = {f1010i}; \n'
        constr_tpl += '!   S = {s}\n'

        def _to_line(line):
            return constr_tpl.format(
                cls_name=self.name, name=line.NAME, weight=1.0, s=line.S,
                f1001abs=line.F1001ABS, f1010abs=line.F1010ABS,
                f1001r=line.F1001REMEAS, f1001i=line.F1001IMMEAS,
                f1010r=line.F1010REMEAS, f1010i=line.F1010IMMEAS,
            )

        return "\n".join(sbs_data.apply(_to_line, axis=1))

    @Matcher.override(Matcher)
    def update_constraints_values(self):
        update_constraints_str = ""
        update_constraints_str += self._get_f_terms_strings()
        return update_constraints_str

    @Matcher.override(Matcher)
    def update_variables_definition(self):
        update_vars_str = ""
        for variable in self.get_variables():
            update_vars_str += "        " + variable + ' := ' + self.name + "." + variable + '_0 + d' + variable + ';\n'
        return update_vars_str

    @Matcher.override(Matcher)
    def generate_changeparameters(self):
        changeparameters_str = ""
        for variable in self.get_variables():
            changeparameters_str += 'select,flag=save,pattern=\"d' + variable + '\";\n'
        return changeparameters_str

    @Matcher.override(Matcher)
    def apply_correction(self):
        apply_correction_str = ""
        for variable in self.get_variables():
            apply_correction_str += variable + ' = ' + self.name + "." + variable + '_0 + d' + variable + ';\n'
        return apply_correction_str

    def _get_f_terms_strings(self):
        sbs_data = self.beatings.coupling
        sbs_data = sbs_data[~sbs_data.index.isin(self.excluded_constraints)]
        f_terms_tpl = "exec, get_f_terms_for(twiss, {name});\n"
        if USE_ABS:
            f_terms_tpl += "{cls_name}.{name}_f1001abs = {name}_f1001abs;\n"
            f_terms_tpl += "{cls_name}.{name}_f1010abs = {name}_f1010abs;\n"
        else:
            f_terms_tpl += "{cls_name}.{name}_f1001r = {name}_f1001r;\n"
            f_terms_tpl += "{cls_name}.{name}_f1001i = {name}_f1001i;\n"
            f_terms_tpl += "{cls_name}.{name}_f1010r = {name}_f1010r;\n"
            f_terms_tpl += "{cls_name}.{name}_f1010i = {name}_f1010i;\n"

        def _to_line(line):
            return f_terms_tpl.format(cls_name=self.name, name=line.NAME)

        return "\n".join(sbs_data.apply(_to_line, axis=1))
