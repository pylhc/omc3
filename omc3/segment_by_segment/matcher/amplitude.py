from omc3.segment_by_segment.matcher.generic import Matcher
from omc3.segment_by_segment.matcher.kmod import KmodMatcher


class AmpMatcher(KmodMatcher):

    BETA_BEATING_CONSTR_WEIGHT = 1.

    @Matcher.override(KmodMatcher)
    def define_constraints(self):
        constr_string = ""
        is_back = "b" in self.propagation
        for plane in ("x", "y"):
            this_amp_data = self.beatings.beta_amp[plane]
            const_names = [self.name + self._get_suffix() + plane + name
                           for name in this_amp_data.NAME]
            beta_beatings = this_amp_data.loc[
                :,
                ("BETABEATAMP{}" if not is_back else "BETABEATAMPBACK{}").format(plane.upper())
            ]
            err_beta_beatings = this_amp_data.loc[
                :,
                ("ERRBETABEATAMP{}" if not is_back else "ERRBETABEATAMPBACK{}").format(plane.upper())
            ]
            constr_string += self._get_constraints_block(const_names, beta_beatings, err_beta_beatings)
        return constr_string

    @Matcher.override(KmodMatcher)
    def _get_suffix(self):
        return ".ampbeating"
