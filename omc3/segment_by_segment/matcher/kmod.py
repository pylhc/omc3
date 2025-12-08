from omc3.segment_by_segment.matcher.generic import Matcher
from omc3.segment_by_segment.matcher.phase import PhaseMatcher


class KmodMatcher(PhaseMatcher):

    BETA_BEATING_CONSTR_WEIGHT = 1.

    BETA_BEATING_TMPL = (
        "{varname} := ((table(twiss, {bpm_name}, bet{plane}) - table({nominal_table_name}, {bpm_name}, bet{plane})) / "
        "table({nominal_table_name}, {bpm_name}, bet{plane}));"
    )

    @Matcher.override(PhaseMatcher)
    def define_aux_vars(self):
        beatings_str = ""
        for plane in ["x", "y"]:
            for name in self.beatings.beta_kmod[plane].NAME:
                beatings_str += KmodMatcher.BETA_BEATING_TMPL.format(
                    varname=self.name + self._get_suffix() + plane + name,
                    bpm_name=name,
                    nominal_table_name=self._get_nominal_table_name(),
                    plane=plane,
                ) + "\n"
        variables_s_str = ""
        for variable in self.get_variables():
            variables_s_str += self.name + '.' + variable + '_0' + ' = ' + variable + ';\n'

        aux_vars_str = PhaseMatcher.SEGMENT_TWISS_TMPL.format(
            seq=self.get_sequence_name(),
            init_vals=self.get_initvals_name(),
            table_name=self._get_nominal_table_name(),
        )
        aux_vars_str += beatings_str
        aux_vars_str += variables_s_str
        return aux_vars_str

    @Matcher.override(PhaseMatcher)
    def define_constraints(self):
        constr_string = ""
        is_back = "b" in self.propagation
        for plane in ("x", "y"):
            this_kmod_data = self.beatings.beta_kmod[plane]
            const_names = [self.name + self._get_suffix() + plane + name
                           for name in this_kmod_data.NAME]
            beta_beatings = this_kmod_data.loc[
                :,
                ("BETABEAT{}" if not is_back else "BETABEATBACK{}").format(plane.upper())
            ]
            err_beta_beatings = this_kmod_data.loc[
                :,
                ("ERRBETABEAT{}" if not is_back else "ERRBETABEATBACK{}").format(plane.upper())
            ]
            constr_string += self._get_constraints_block(const_names, beta_beatings, err_beta_beatings)
        return constr_string

    def _get_suffix(self):
        return ".kmodbeating"
