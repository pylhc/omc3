import os
from sbs_general_matcher.gui.matchers_models.matcher_model_default import (
    MatcherModelDefault,
    MatcherPlotterDefault,
)
from sbs_general_matcher.matchers.matcher import MatcherFactory
from sbs_general_matcher.matchers.coupling_matcher import CouplingMatcher
from tfs_files import tfs_pandas


class MatcherModelCoupling(MatcherModelDefault):

    def create_matcher(self, lhc_mode, match_path):
        factory = MatcherFactory(CouplingMatcher)
        self._matcher = (
            factory
            .set_lhc_mode(lhc_mode)
            .set_beam(self._beam)
            .set_name(self._name)
            .set_var_classes(MatcherModelDefault.BETA_CORR_CLASSES)
            .set_match_path(match_path)
            .set_label(self._label)
            .set_use_errors(self._use_errors)
            .set_propagation(self._propagation)
            .set_measurement_path(self._meas_path)
            .set_excluded_constraints([])
            .set_excluded_variables([])
        ).create()

    def get_plotter(self, figure):
        if self._plotter is None:
            self._plotter = MatcherPlotterCoupling(figure, self)
        return self._plotter


class MatcherPlotterCoupling(MatcherPlotterDefault):

    def plot(self):
        self._figure.clear()
        self._axes_to_data = {}
        label = self._matcher_model.get_label()

        axes_f1001 = self._figure.add_subplot(2, 1, 1)
        axes_f1010 = self._figure.add_subplot(2, 1, 2)
        outpath = self._matcher_model.get_output_path()
        file_coup = tfs_pandas.read_tfs(os.path.join(
            outpath, "sbs",
            "sbscouple_" + label + ".out")
        )
        self._axes_to_data[axes_f1001] = file_coup
        self._axes_to_data[axes_f1010] = file_coup
        self._plot_match(axes_f1001, file_coup, "f1001")
        self._plot_match(axes_f1010, file_coup, "f1010")

        self._figure.tight_layout()
        self._figure.patch.set_visible(False)
        self._figure.canvas.draw()

    def _plot_match(self, axes, sbs_file, term):
        if self._matcher_model.get_propagation() == "front":
            MatcherPlotterCoupling._plot_front(axes, sbs_file, term)
        elif self._matcher_model.get_propagation() == "back":
            MatcherPlotterCoupling._plot_back(axes, sbs_file, term)

        axes.legend(loc="lower left", prop={'size': 16})
        if term == "f1001":
            axes.set_ylabel(r"$abs(f_{1001})$")
        elif term == "f1010":
            axes.set_ylabel(r"$abs(f_{1010})$")

    @staticmethod
    def _plot_front(axes, sbs_file, term):
        up_term = term.upper()
        axes.errorbar(sbs_file.S,
                      getattr(sbs_file, up_term + "ABSMEAS").values,
                      getattr(sbs_file, "ERR" + up_term + "ABSMEAS").values,
                      label=r"$abs(f_{" + up_term[1:] + "})$ measured",
                      color="blue")
        axes.plot(sbs_file.S,
                  getattr(sbs_file, up_term + "ABSMEAS").values,
                  marker="o", markersize=7., color="blue")

        axes.plot(sbs_file.S,
                  getattr(sbs_file, up_term + "ABSCOR").values,
                  label=r"$abs(f_{" + up_term[1:] + "})$ model",
                  color="green")
        axes.plot(sbs_file.S,
                  getattr(sbs_file, up_term + "ABSCOR").values,
                  marker="o", markersize=7., color="green")

    @staticmethod
    def _plot_back(axes, sbs_file, term):
        up_term = term.upper()
        axes.errorbar(sbs_file.S,
                      getattr(sbs_file, up_term + "ABSBACK").values,
                      getattr(sbs_file, "ERR" + up_term + "ABSBACK").values,
                      label=r"$abs(f_{" + up_term[1:] + "})$ measured",
                      color="blue")
        axes.plot(sbs_file.S,
                  getattr(sbs_file, up_term + "ABSBACK").values,
                  marker="o", markersize=7., color="blue")

        axes.plot(sbs_file.S,
                  getattr(sbs_file, up_term + "ABSBACKCOR").values,
                  label=r"$abs(f_{" + up_term[1:] + "})$ model",
                  color="green")
        axes.plot(sbs_file.S,
                  getattr(sbs_file, up_term + "ABSBACKCOR").values,
                  marker="o", markersize=7., color="green")
