import sys
import os
import shutil
import logging

LOGGER = logging.getLogger(__name__)


class MatcherModelDefault(object):

    BETA_CORR_CLASSES = ("MQX", "MQXT", "MQXNEW", "MQT", "MQM", "MQTL")
    COUP_CORR_CLASSES = ("MQSX")

    def __init__(self, match_path, name, beam, meas_path,
                 label, use_errors, propagation):
        self._match_path = match_path
        self._name = name
        self._beam = beam
        self._meas_path = meas_path
        self._label = label
        self._use_errors = use_errors
        self._propagation = propagation
        self._matcher = None
        self._elements_positions = None
        self.ignored_vars = []
        self.disabled_constr = {1: [], 2: []}
        self._plotter = None

    def get_name(self):
        return self._name

    def get_beam(self):
        return self._beam

    def get_meas_path(self):
        return self._meas_path

    def get_output_path(self):
        return os.path.join(self._match_path,
                            self._name)

    def get_label(self):
        return self._label

    def get_use_errors(self):
        return self._use_errors

    def get_propagation(self):
        return self._propagation

    def get_ignore_vars_list(self):
        return self._ignore_vars_list

    def set_var_active(self, var_name, active):
        if self._matcher is None:
            raise ValueError(
                "Cannot set excluded variables of not created matcher"
            )
        if active and var_name in self._matcher.excluded_variables:
            self._matcher.excluded_variables.remove(var_name)
            LOGGER.info(self._name + " -> Activated var: " + var_name)
        elif (not active and var_name not in
                self._matcher.excluded_variables):
            self._matcher.excluded_variables.append(var_name)
            LOGGER.info(self._name + " -> Disabled var: " + var_name)

    def disable_all_vars(self):
        self._matcher.excluded_variables =\
            self.get_matcher().get_variables(exclude=False)

    def toggle_constr(self, constr_name):
        if self._matcher is None:
            raise ValueError(
                "Cannot set excluded variables of not created matcher"
            )
        if (constr_name in self._matcher.excluded_constraints):
            self._matcher.excluded_constraints.remove(constr_name)
            LOGGER.info(self._name + " -> Activated constr: " + constr_name)
        elif (constr_name not in self._matcher.excluded_constraints):
            self._matcher.excluded_constraints.append(constr_name)
            LOGGER.info(self._name + " -> Disabled constr: " + constr_name)

    def create_matcher(self, match_path):
        raise NotImplementedError

    def delete_matcher(self):
        shutil.rmtree(self.get_output_path())

    def get_plotter(self, figure):
        raise NotImplementedError

    def get_matcher(self):
        return self._matcher

    def get_match_results(self):
        match_results = {}
        if self._matcher is not None:
            corr_file = os.path.join(
                self._matcher.match_path,
                "changeparameters.madx"
            )
            try:
                with open(corr_file, "r") as changeparameters:
                    for line in changeparameters:
                        parts = line.split("=")
                        variable_name = parts[0].replace("d", "", 1).strip()
                        variable_value = float(
                            parts[1].replace(";", "").strip()
                        )
                        match_results[variable_name] = variable_value
            except IOError:
                LOGGER.exception("Cannot parse corrections file: %s",
                                 corr_file)
        return match_results


class MatcherPlotterDefault(object):

    DISTANCE_THRESHOLD2 = 10 ** 2
    BOX_STYLE = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)

    def __init__(self, figure, matcher_model):
        self._figure = figure
        self._matcher_model = matcher_model
        self._axes_to_data = {}
        self._latest_annotation = None
        self.update_vars_funct = lambda color_dict: None
        figure.canvas.mpl_connect(
            'motion_notify_event',
            lambda event: self.on_move_event(event),
        )
        figure.canvas.mpl_connect(
            'button_press_event',
            lambda event: self.on_click_event(event),
        )

    def update_vars(self):
        results_dict = self._matcher_model.get_match_results()
        self.update_vars_funct(results_dict)

    def on_move_event(self, event):
        for axes in self._figure.axes:
            del axes.texts[:]
        annotation_was_none = True
        if self._latest_annotation is not None:
            self._latest_annotation = None
            annotation_was_none = False
        element_name, element_position = self.get_element_within_range(event)
        if element_name is not None and element_position is not None:
            x, y = element_position
            new_text = (element_name + "\n" +
                        "S = " + str(x))
            self._latest_annotation = event.inaxes.text(
                x, y,
                new_text,
                bbox=MatcherPlotterDefault.BOX_STYLE
            )
        # TODO: Only redraw on element transition (no repeat same element)
        have_to_redraw = (
            (self._latest_annotation is not None) or
            (self._latest_annotation is None and not annotation_was_none)
        )
        if have_to_redraw:
            self._redraw_figure(event)

    def on_click_event(self, event):
        selected_point = self._get_point_within_range(event)
        if selected_point is not None:
            x, _ = selected_point
            elements_data = self._axes_to_data[event.inaxes].set_index("S")
            element_name = str(elements_data.loc[x, "NAME"])
            self._matcher_model.toggle_constr(element_name)
        self._redraw_figure(event)

    def get_element_within_range(self, event):
        selected_point = self._get_point_within_range(event)
        if selected_point is not None:
            elements_data = self._axes_to_data[event.inaxes].set_index("S")
            x, y = selected_point
            element_name = elements_data.loc[x, "NAME"]
            element_position = (x, y)
            return element_name, element_position
        return None, None

    def _get_points_for_element(self, element_name, axes):
        points = []
        try:
            element_x = self._axes_to_data[axes].set_index("NAME").loc[
                element_name, "S"
            ]
        except KeyError:
            return points
        for line in axes.get_lines():
            xydata_in_plot = line.get_xydata()
            for data_point in xydata_in_plot:
                x, _ = data_point
                if x == element_x:
                    points.append(data_point)
        return points

    def _redraw_figure(self, event):
        for axes in self._figure.axes:
            del axes.texts[:]
            for element_name in self._matcher_model._matcher.excluded_constraints:
                points = self._get_points_for_element(element_name, axes)
                for point in points:
                    x, y = point
                    axes.text(x, y, "X",
                              horizontalalignment='center',
                              verticalalignment='center',
                              fontsize=15, color='red')
        if event.inaxes is not None and self._latest_annotation is not None:
            event.inaxes.texts.append(self._latest_annotation)
        self._figure.canvas.draw()

    def _get_point_within_range(self, event):
        axes = event.inaxes
        if axes is not None:
            x_fig, y_fig = event.x, event.y
            for line in axes.get_lines():
                xydata_in_plot = line.get_xydata()
                min_distance2 = sys.float_info.max
                selected_point_plot = None
                for data_point in xydata_in_plot:
                    fig_point_x, fig_point_y = axes.transData.transform(data_point)
                    distance2 = (fig_point_x - x_fig) ** 2 + (fig_point_y - y_fig) ** 2
                    if distance2 < min_distance2:
                        min_distance2 = distance2
                        selected_point_plot = data_point
                if min_distance2 < MatcherPlotterDefault.DISTANCE_THRESHOLD2:
                    return selected_point_plot
        return None


if __name__ == "__main__":
    print >> sys.stderr, "This module is meant to be imported."
    sys.exit(-1)
