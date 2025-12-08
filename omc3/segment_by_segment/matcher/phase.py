from omc3.segment_by_segment.matcher.generic import Matcher


class PhaseMatcher(Matcher):

    @Matcher.override(Matcher)
    def get_variables(self, exclude=True):
        variables = self.segment.get_segment_vars(
            classes=self.var_classes,
        )
        if exclude:
            variables = [
                var for var in variables
                if var not in self.excluded_variables
            ]
        return variables

    PH_ERR_TMPL = (
        "{matcher_name}.dmu{plane}{name} := "
        "{sign}((table(twiss, {name}, mu{plane}) - "
        "table({nominal_table_name}, {name}, mu{plane})));\n"
    )

    SEGMENT_TWISS_TMPL = (
        "use, period={seq};\n"
        "twiss, beta0={init_vals}, chrom, table={table_name};\n"
    )

    @Matcher.override(Matcher)
    def define_aux_vars(self):
        phases_str = ""
        sign = "" if "f" in self.propagation else "-"
        for plane in ("x", "y"):
            for name in self.beatings.phase[plane].NAME:
                phases_str += PhaseMatcher.PH_ERR_TMPL.format(
                    matcher_name=self.name, sign=sign, name=name, plane=plane,
                    nominal_table_name=self._get_nominal_table_name(),
                )

        variables_s_str = ""
        for variable in self.get_variables():
            variables_s_str += self.name + '.' + variable.replace("->", "") + '_0' + ' = ' + variable + ';\n'

        aux_vars_str = PhaseMatcher.SEGMENT_TWISS_TMPL.format(
            seq=self.get_sequence_name(),
            init_vals=self.get_initvals_name(),
            table_name=self._get_nominal_table_name(),
        )
        aux_vars_str += phases_str
        aux_vars_str += variables_s_str
        return aux_vars_str

    @Matcher.override(Matcher)
    def define_constraints(self):
        constr_string = ""
        for plane in ("x", "y"):
            sbs_data = self.beatings.phase[plane]
            sbs_data = sbs_data[~sbs_data.index.isin(self.excluded_constraints)]
            is_back = "b" in self.propagation
            phases = sbs_data.loc[:, (("PROPPHASE{}" if not is_back else "BACKPHASE{}")
                                      .format(plane.upper()))]
            errors = sbs_data.loc[:, (("ERRPROPPHASE{}" if not is_back else "ERRBACKPHASE{}")
                                      .format(plane.upper()))]
            for name, phase, error in zip(sbs_data.NAME, phases, errors):
                constr_string += self._get_constraint_instruction(
                    self.name + '.dmu' + plane + name,
                    phase, error)
        return constr_string

    @Matcher.override(Matcher)
    def update_constraints_values(self):
        return ""

    @Matcher.override(Matcher)
    def update_variables_definition(self):
        update_vars_str = ""
        for variable in self.get_variables():
            update_vars_str += "        " + variable + ' := ' + self.name + "." + variable.replace("->", "") + '_0 + d' + variable.replace("->", "") + ';\n'
        return update_vars_str

    @Matcher.override(Matcher)
    def generate_changeparameters(self):
        changeparameters_str = ""
        for variable in self.get_variables():
            changeparameters_str += 'select,flag=save,pattern=\"d' + variable.replace("->", "") + '\";\n'
        return changeparameters_str

    @Matcher.override(Matcher)
    def apply_correction(self):
        apply_correction_str = ""
        for variable in self.get_variables():
            apply_correction_str += variable + ' = ' + self.name + "." + variable.replace("->", "") + '_0 + d' + variable.replace("->", "") + ';\n'
        return apply_correction_str
