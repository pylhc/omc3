import os
from sbs_general_matcher.gui.matchers_models.matcher_model_default import (
    MatcherModelDefault,
    MatcherPlotterDefault,
)
from sbs_general_matcher.matchers.matcher import MatcherFactory
from sbs_general_matcher.matchers.kmod_matcher import KmodMatcher
from tfs_files import tfs_pandas


class MatcherModelKmod(MatcherModelDefault):

    def create_matcher(self, lhc_mode, match_path):
        factory = MatcherFactory(KmodMatcher)
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
            self._plotter = MatcherPlotterKmod(figure, self)
        return self._plotter


class MatcherPlotterKmod(MatcherPlotterDefault):

    def plot(self):
        self._figure.clear()
        self._axes_to_data = {}
        label = self._matcher_model.get_label()

        axes_x = self._figure.add_subplot(2, 1, 1)
        kmod_bb_horizontal = self._get_kmod_beta_beat_file(label, "X")
        bb_horizontal = self._get_beta_beat_file(label, "X")
        self._axes_to_data[axes_x] = bb_horizontal
        self._plot_match(axes_x, kmod_bb_horizontal, bb_horizontal, "X")

        axes_y = self._figure.add_subplot(2, 1, 2)
        kmod_bb_vertical = self._get_kmod_beta_beat_file(label, "Y")
        bb_vertical = self._get_beta_beat_file(label, "Y")
        self._axes_to_data[axes_y] = bb_vertical
        self._plot_match(axes_y, kmod_bb_vertical, bb_vertical, "Y")

        self._figure.tight_layout()
        self._figure.patch.set_visible(False)
        self._figure.canvas.draw()

    def _get_kmod_beta_beat_file(self, label, plane):
        outpath = self._matcher_model.get_output_path()
        file_data = tfs_pandas.read_tfs(os.path.join(
            outpath, "sbs",
            "sbskmodbetabeat" + plane.lower() + "_" + label + ".out")
        )
        return file_data

    def _get_beta_beat_file(self, label, plane):
        outpath = self._matcher_model.get_output_path()
        file_data = tfs_pandas.read_tfs(os.path.join(
            outpath, "sbs",
            "sbsbetabeat" + plane.lower() + "_" + label + ".out")
        )
        return file_data

    def _plot_match(self, axes, kmod_bb_data, bb_data, plane):
        if self._matcher_model.get_propagation() == "front":
            MatcherPlotterKmod._plot_front(axes, kmod_bb_data, bb_data, plane)
        elif self._matcher_model.get_propagation() == "back":
            MatcherPlotterKmod._plot_back(axes, kmod_bb_data, bb_data, plane)

        axes.legend(loc="upper left", prop={'size': 16})
        axes.set_ylabel(r"$\Delta\beta_{" + plane.lower() + r"} / {\beta_{model}}$")
        axes.set_xlabel("S along the segment [m]")

    @staticmethod
    def _plot_front(axes, kmod_bb_data, bb_data, plane):
        axes.plot(bb_data.S, getattr(bb_data, "BETABEATCOR" + plane).values,
                  label=r"$\Delta\beta / {\beta_{model}}$ model", color="green")
        axes.plot(bb_data.S, getattr(bb_data, "BETABEATCOR" + plane).values,
                  marker="o", markersize=7., color="green")

        axes.errorbar(kmod_bb_data.S,
                      getattr(kmod_bb_data, "BETABEAT" + plane).values,
                      getattr(kmod_bb_data, "ERRBETABEAT" + plane).values,
                      fmt='o', markersize=8.,
                      label=r"$\Delta\beta / {\beta_{model}}$ k-mod",
                      color="blue")

    @staticmethod
    def _plot_back(axes, kmod_bb_data, bb_data, plane):
        axes.plot(bb_data.S, getattr(bb_data, "BETABEATCORBACK" + plane).values,
                  label=r"$\Delta\beta / {\beta_{model}}$ model", color="green")
        axes.plot(bb_data.S, getattr(bb_data, "BETABEATCORBACK" + plane).values,
                  marker="o", markersize=7., color="green")

        axes.errorbar(kmod_bb_data.S,
                      getattr(kmod_bb_data, "BETABEATBACK" + plane).values,
                      getattr(kmod_bb_data, "ERRBETABEATBACK" + plane).values,
                      fmt='o', markersize=8.,
                      label=r"$\Delta\beta / {\beta_{model}}$ k-mod",
                      color="blue")
