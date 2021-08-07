from PyQt5 import QtWidgets
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
import matplotlib.pyplot as plt


class SbSGuiMatchResultView(QtWidgets.QWidget):

    def __init__(self, variables, parent=None):
        super(SbSGuiMatchResultView, self).__init__(parent)

        # Sent triggers
        self.toggle_var_action = lambda name, active, all: None

        self._variables = variables
        self._build_gui()

    def _build_gui(self):
        main_layout = QtWidgets.QHBoxLayout()
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        self._figure = plt.figure()
        figure_widget = QtWidgets.QWidget()
        figure_widget.setLayout(self._get_new_canvas_layout(self._figure))
        main_splitter.addWidget(figure_widget)

        variables_layout = QtWidgets.QVBoxLayout()
        variables_frame = _BorderedGroupBox("Variables")
        variables_frame.setLayout(variables_layout)

        self._vars_layout = QtWidgets.QVBoxLayout()
        if not len(self._variables) == 0:
            widget = QtWidgets.QWidget()
            widget.setLayout(self._vars_layout)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            vars_frame = _BorderedGroupBox("")
            vars_frame_layout = QtWidgets.QHBoxLayout()
            vars_frame_layout.addWidget(scroll)
            vars_frame.setLayout(vars_frame_layout)
            variables_layout.addWidget(vars_frame)
            for variable in self._variables:
                self._vars_layout.addWidget(
                    _CustomCheckBox(variable, self)
                )

        self._select_all_checkbox = QtWidgets.QCheckBox("Toggle select all")
        self._select_all_checkbox.mousePressEvent = self._toogle_select_all
        variables_layout.addWidget(self._select_all_checkbox)

        main_splitter.addWidget(variables_frame)
        self.setLayout(main_layout)

    def _get_new_canvas_layout(self, figure):
        layout = QtWidgets.QVBoxLayout()
        canvas = FigureCanvas(figure)
        toolbar = _CustomNavigationBar(
            canvas,
            figure,
            self
        )
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        return layout

    def get_figure(self):
        return self._figure

    def _toogle_select_all(self, event):
        self._select_all_checkbox.setChecked(
            not self._select_all_checkbox.isChecked()
        )

        def toggle_checkbox(checkbox):
            checkbox.set_state(event, self._select_all_checkbox.isChecked())

        self._loop_through_checkboxes(toggle_checkbox)

    def update_variables(self, name_value_dict):
        def update(checkbox):
            checkbox.set_variable_value_from_dict(name_value_dict)

        self._loop_through_checkboxes(update)

    def set_var_active(self, varname, active):
        def toggle(checkbox):
            if str(checkbox.text()) == varname:
                checkbox.setChecked(active)

        self._loop_through_checkboxes(toggle)

    def _loop_through_checkboxes(self, function):
        for index in range(self._vars_layout.count()):
            checkbox = self._vars_layout.itemAt(index).widget()
            if issubclass(checkbox.__class__, QtWidgets.QCheckBox):
                function(checkbox)


class _CustomNavigationBar(NavigationToolbar):
    LEGEND_LOCATIONS = [1, 2, 3, 4, None]

    def __init__(self, canvas, figure, parent=None):
        self.toolitems = list(self.toolitems)
        self.toolitems.append((
            "Move legend", "Move or hide the legend", "hand", "move_legend"
        ))
        super(
            _CustomNavigationBar,
            self
        ).__init__(canvas, parent)

        self._figure = figure
        self._legend_location_index = 0

    def move_legend(self, *args):
        self._legend_location_index = (
            self._legend_location_index + 1
        ) % len(_CustomNavigationBar.LEGEND_LOCATIONS)
        for axes in self._figure.axes:
            legend = axes.get_legend()
            loc = _CustomNavigationBar.LEGEND_LOCATIONS[
                self._legend_location_index
            ]
            if loc is None:
                legend.set_visible(False)
            else:
                legend.set_visible(True)
                legend._set_loc(loc)
        self.draw()


class _CustomCheckBox(QtWidgets.QCheckBox):
    STRENGH_COLOR_LIMIT = 1e-3
    BACKGROUND_COLOR_TEMPLATE = "QCheckBox { background-color: %(COLOR)s;}"
    DISABLED_CSS_COLOR = "rgb(224, 224, 224)"

    def __init__(self, text, view):
        super(_CustomCheckBox, self).__init__(text)
        self.setMouseTracking(True)
        self._tooltip_text = ""
        self._view = view

    def mousePressEvent(self, event):
        self.set_state(event, not self.isChecked())

    def set_state(self, event, checked):
        all = True
        if event.button() == QtCore.Qt.RightButton:
            all = False
        self.setChecked(checked)
        self._view.toggle_var_action(
            str(self.text()),
            checked,
            all,
        )

    def mouseMoveEvent(self, event):
        super(_CustomCheckBox, self).mouseMoveEvent(event)
        QtWidgets.QToolTip.showText(event.globalPos(),
                                    self._tooltip_text,
                                    widget=self)

    def set_variable_value_from_dict(self, name_value_dict):
        try:
            variable_value = name_value_dict[str(self.text())]
            tooltip_text = str(variable_value)
            color_string = _CustomCheckBox._get_css_color_for_value(
                variable_value
            )
        except KeyError:
            tooltip_text = "Disabled"
            color_string = _CustomCheckBox.DISABLED_CSS_COLOR
        self._tooltip_text = str(tooltip_text)
        style = _CustomCheckBox.BACKGROUND_COLOR_TEMPLATE % {
            "COLOR": color_string,
        }
        self.setStyleSheet(style)
        self.update()

    @staticmethod
    def _get_css_color_for_value(value):
        absolute_strength = abs(value)
        limit = _CustomCheckBox.STRENGH_COLOR_LIMIT
        if absolute_strength > limit:
            red = 255
        else:
            red = int((absolute_strength / limit) * 255)
        green = 255 - red
        blue = 0
        return "rgb(" + str(red) + ", " + str(green) + ", " + str(blue) + ")"


class _BorderedGroupBox(QtWidgets.QGroupBox):

    def __init__(self, label, parent=None):
        super(_BorderedGroupBox, self).__init__(label, parent)
