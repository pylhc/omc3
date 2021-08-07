import sys
import os
import subprocess
import logging
from contextlib import contextmanager
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, Qt, QFileSystemWatcher, pyqtSignal
from omc3 import sbs_matcher
from omc3.segment_by_segment.matcher.gui.sbs_gui_matcher_selection import SbSGuiMatcherTypeSelection
from omc3.segment_by_segment.matcher.gui.widgets import InitialConfigPopup, LogDialog
from omc3.segment_by_segment.matcher.gui.sbs_gui_match_result_view import SbSGuiMatchResultView


LOGGER = logging.getLogger(__name__)


class SbSGuiMain(QtWidgets.QMainWindow):

    WINDOW_TITLE = "Segment-by-segment general matcher GUI"

    def __init__(self, controller, parent=None):
        super(SbSGuiMain, self).__init__(parent)

        self._controller = controller
        self._active_background_dialog = None
        self._log_dialog = LogDialog(parent=self)
        logging.getLogger("").addHandler(self._log_dialog)
        self._build_gui()

    def _build_gui(self):

        self.setWindowTitle(SbSGuiMain.WINDOW_TITLE)
        screen_shape = QtWidgets.QDesktopWidget().screenGeometry()
        self.resize(2 * screen_shape.width() / 3,
                    2 * screen_shape.height() / 3)

        self._main_widget = SbSGuiMain.SbSMainWidget(self._controller)
        self.setCentralWidget(self._main_widget)

        main_menu = self.menuBar()
        main_menu.setNativeMenuBar(False)
        matchers_menu = main_menu.addMenu("Matchers")
        matchers_menu.addAction(self._get_new_matcher_action())
        matchers_menu.addAction(self._get_clone_matcher_action())
        matchers_menu.addAction(self._get_remove_matcher_action())
        main_menu.addAction(self._get_show_log_action())
        view_menu = main_menu.addMenu("View")
        view_menu.addAction(self._get_tile_windows_action())
        view_menu.addAction(self._get_cascade_windows_action())

    def add_subwindow(self, matcher_model, matcher_resuts_view):
        subwindow = MatcherSubwindow(
            matcher_model,
            matcher_resuts_view,
            self._controller.remove_matcher,
        )
        self._main_widget.mdi_area.addSubWindow(subwindow)
        subwindow.setWindowTitle(matcher_model.get_name())
        subwindow.showMaximized()
        if len(self.get_subwindows_list()) > 1:
            self._main_widget.mdi_area.tileSubWindows()

    def remove_subwindow(self, subwindow):
        self._main_widget.mdi_area.removeSubWindow(subwindow)

    def get_selected_subwindow(self):
        return self._main_widget.mdi_area.activeSubWindow()

    def get_subwindows_list(self):
        return self._main_widget.mdi_area.subWindowList()

    def show_background_task_dialog(self, message):
        self._active_background_dialog = SbSGuiMain.BackgroundTaskDialog(
            message, parent=self
        )
        self._active_background_dialog.setVisible(True)

    def hide_background_task_dialog(self):
        if self._active_background_dialog is not None:
            self._active_background_dialog.setVisible(False)
            self._active_background_dialog = None

    def show_error_dialog(self, title, message):
        message_box = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Critical,
            title,
            message,
            QtWidgets.QMessageBox.Ok,
            self
        )
        message_box.exec_()

    def show_confirm_dialog(self, title, message):
        message_box = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Question,
            title,
            message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            self
        )
        ret = message_box.exec_()
        if ret == QtWidgets.QMessageBox.Yes:
            return True
        return False

    def is_minimize_selected(self):
        return self._main_widget.is_minimize_selected()

    def _get_new_matcher_action(self):
        new_matcher_action = QtWidgets.QAction("New matcher...", self)
        new_matcher_action.triggered.connect(self._controller.new_matcher)
        return new_matcher_action

    def _get_remove_matcher_action(self):
        remove_matcher_action = QtWidgets.QAction("Remove matcher", self)
        remove_matcher_action.triggered.connect(
            self._controller.remove_matcher
        )
        return remove_matcher_action

    def _get_show_log_action(self):
        show_log_action = QtWidgets.QAction("Show log", self)
        show_log_action.triggered.connect(
            self._show_log
        )
        return show_log_action

    def _get_tile_windows_action(self):
        tile_windows_action = QtWidgets.QAction("Tile windows", self)
        tile_windows_action.triggered.connect(
            self._main_widget.mdi_area.tileSubWindows
        )
        return tile_windows_action

    def _get_cascade_windows_action(self):
        tile_windows_action = QtWidgets.QAction("Cascade windows", self)
        tile_windows_action.triggered.connect(
            self._main_widget.mdi_area.cascadeSubWindows
        )
        return tile_windows_action

    def _get_clone_matcher_action(self):
        clone_matcher_action = QtWidgets.QAction("Clone matcher", self)
        clone_matcher_action.triggered.connect(self._controller.clone_matcher)
        return clone_matcher_action

    def _show_log(self):
        self._log_dialog.show()

    class SbSMainWidget(QtWidgets.QWidget):
        def __init__(self, controller, parent=None):
            super(SbSGuiMain.SbSMainWidget, self).__init__(parent)
            self._controller = controller
            self._build_gui()

        def is_minimize_selected(self):
            return self._minimize_checkbox.isChecked()

        def _build_gui(self):
            main_layout = QtWidgets.QVBoxLayout()
            self.setLayout(main_layout)

            self.mdi_area = QtWidgets.QMdiArea()
            self.mdi_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.mdi_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            main_layout.addWidget(self.mdi_area)

            lower_panel_layout = QtWidgets.QHBoxLayout()
            buttons_layout = QtWidgets.QVBoxLayout()
            global_options_layout = QtWidgets.QVBoxLayout()
            lower_panel_layout.addLayout(buttons_layout, stretch=3)
            lower_panel_layout.addLayout(global_options_layout)
            main_layout.addLayout(lower_panel_layout)

            self._minimize_checkbox = QtWidgets.QCheckBox("Minimize variables")
            global_options_layout.addWidget(self._minimize_checkbox)

            run_button = QtWidgets.QPushButton("Run matching")
            run_button.clicked.connect(self._controller.run_matching)
            buttons_layout.addWidget(run_button)

            edit_corr_button = QtWidgets.QPushButton("Edit corrections file")
            edit_corr_button.clicked.connect(
                self._controller.edit_corrections_file
            )
            buttons_layout.addWidget(edit_corr_button)

    class BackgroundTaskDialog(QtWidgets.QMessageBox):
        def __init__(self, message, parent=None):
            super(SbSGuiMain.BackgroundTaskDialog, self).__init__(
                QtWidgets.QMessageBox.NoIcon,
                "Please wait...",
                message
            )
            self.setWindowFlags(Qt.CustomizeWindowHint)
            self.setStandardButtons(QtWidgets.QMessageBox.NoButton)
            self.resize(520, 340)
            self.setModal(True)


class SbSGuiMainController(object):

    def __init__(self):
        self._view = SbSGuiMain(self)
        self._match_path = None
        self._possible_measurements = {1: [], 2: []}
        self._input_dir = None
        self._current_thread = None
        self._active_watcher = None

    @staticmethod
    def ask_for_initial_config(lhc_mode, match_path):
        initial_config_popup = InitialConfigPopup(lhc_mode, match_path)
        initial_config_popup.setWindowTitle("Please choose an output path")
        result_code = initial_config_popup.exec_()
        if result_code == QtWidgets.QDialog.Accepted:
            return (initial_config_popup.get_selected_lhc_mode(),
                    initial_config_popup.get_selected_file())
        else:
            return None, None

    def set_match_path(self, match_path):
        self._match_path = match_path
        self._corrections_file = os.path.join(match_path,
                                              "changeparameters.madx")

    def set_lhc_mode(self, lhc_mode):
        self._lhc_mode = lhc_mode

    def set_input_dir(self, input_dir):
        if input_dir is None:
            return
        self._input_dir = os.path.abspath(input_dir)
        self._find_measurements()

    def _find_measurements(self):
        self._possible_measurements = {1: [], 2: []}
        if self._input_dir is None:
            return
        for lhcb1or2 in os.listdir(self._input_dir):
            if lhcb1or2 == "LHCB1":
                beam = 1
            elif lhcb1or2 == "LHCB2":
                beam = 2
            else:
                continue
            results_path = os.path.join(self._input_dir, lhcb1or2,
                                        "Results")
            if not os.path.isdir(results_path):
                continue
            for individual_result in os.listdir(results_path):
                individual_result_path = os.path.join(results_path,
                                                      individual_result)
                if os.path.isdir(individual_result_path):
                    self._possible_measurements[beam].append(
                        individual_result_path
                    )

    def get_posible_measurements(self, beam):
        return self._possible_measurements[beam]

    def get_match_path(self):
        return self._match_path

    def show_view(self):
        self._view.show()

    def new_matcher(self):
        self._new_matcher_from_chooser()

    def clone_matcher(self):
        self._new_matcher_from_chooser(
            self._view.get_selected_subwindow().model
        )

    def _new_matcher_from_chooser(self, matcher_to_clone=None):
        selection_dialog = SbSGuiMatcherTypeSelection(
            self, clone_matcher=matcher_to_clone
        )
        result_code = selection_dialog.exec_()
        if result_code == QtWidgets.QDialog.Accepted:
            matcher_model = selection_dialog.get_selected_matcher()
            with self._heavy_task("Copying files..."):
                try:
                    matcher_model.create_matcher(self._lhc_mode,
                                                 self._match_path)
                except IOError as err:
                    self._view.show_error_dialog(
                        "Exception happened copying the files.",
                        str(err),
                    )
                    LOGGER.exception("Exception happened copying the files.")
                    return
            vars = matcher_model.get_matcher().get_variables(exclude=False)
            matcher_model.disable_all_vars()
            matcher_view = SbSGuiMatchResultView(vars)
            matcher_view.toggle_var_action = (
                lambda name, active, all: self._on_var_toggle(
                    matcher_model, matcher_view, name, active, all)
            )
            self._view.add_subwindow(matcher_model, matcher_view)

    @contextmanager
    def _heavy_task(self, message):
        self._view.show_background_task_dialog(message)
        try:
            yield
        finally:
            self._view.hide_background_task_dialog()

    def is_this_matcher_name_ok(self, matcher_name):
        for matcher_subw in self._view.get_subwindows_list():
            model_name = matcher_subw.model.get_name()
            if matcher_name == model_name:
                return False
        return True

    def remove_matcher(self):
        matcher_subw = self._view.get_selected_subwindow()
        if matcher_subw is None:
            return
        do_remove = self._view.show_confirm_dialog(
            "Removing matcher",
            "Will remove matcher " +
            matcher_subw.model.get_name() +
            ". Are you sure?"
        )
        if do_remove:
            with self._heavy_task("Deleting matcher..."):
                matcher_subw.model.delete_matcher()
                self._view.remove_subwindow(matcher_subw)
        return do_remove

    def run_matching(self, just_twiss=False):
        matchers_list = []
        for matcher_subw in self._view.get_subwindows_list():
            matchers_list.append(matcher_subw.model.get_matcher())
        minimize = self._view.is_minimize_selected()
        input_data = sbs_matcher.InputData(
            self._lhc_mode, self._match_path, minimize, matchers_list
        )

        if not just_twiss:
            def background_task():
                had_active_watcher = False
                if self._active_watcher is not None:
                    had_active_watcher = True
                    self._active_watcher.removePath(self._match_path)
                    self._active_watcher = None
                sbs_matcher.run_full_madx_matching(input_data)
                if had_active_watcher:
                    self._watch_dir(self._match_path)
        else:
            def background_task():
                sbs_matcher.run_twiss_and_sbs(input_data)

        self._current_thread = BackgroundThread(
            self._view,
            background_task,
            message="Running matching...",
            on_end_function=self._on_match_end,
            on_exception_function=self._on_match_exception
        )
        self._current_thread.start()

    def _on_match_end(self):
        for matcher_subw in self._view.get_subwindows_list():
            matcher_model = matcher_subw.model
            view = matcher_subw.widget()
            figure = view.get_figure()
            model_plotter = matcher_model.get_plotter(figure)
            model_plotter.plot()
            model_plotter.update_vars_funct = view.update_variables
            model_plotter.update_vars()
        self._current_thread = None

    def _on_match_exception(self, message):
        self._current_thread = None

    def _on_var_toggle(self, this_model, this_view, name, active, all):
        if all:
            for matcher_subw in self._view.get_subwindows_list():
                matcher_subw.model.set_var_active(name, active)
                matcher_subw.widget().set_var_active(name, active)
        else:
            this_model.set_var_active(name, active)
            this_view.set_var_active(name, active)

    def edit_corrections_file(self):
        if not os.path.isfile(self._corrections_file):
            open(self._corrections_file, "a").close()  # Create empty file
        self._launch_text_editor(self._corrections_file)
        if (self._active_watcher is not None and
                self._corrections_file in self._active_watcher.files()):
            return
        self._watch_dir(self._match_path)

    def _watch_dir(self, directory):
        self._active_watcher = QFileSystemWatcher([directory])
        self._active_watcher.directoryChanged.connect(self._match_dir_changed)

    def _launch_text_editor(self, file_path):
        if sys.platform.startswith('darwin'):
            subprocess.call(('open', file_path))
        elif os.name == 'nt':
            os.startfile(file_path)
        elif os.name == 'posix':
            subprocess.call(('xdg-open', file_path))

    def _match_dir_changed(self, path):
        if os.path.samefile(path, self._match_path):
            if self._current_thread is None:
                self.run_matching(just_twiss=True)


class MatcherSubwindow(QtWidgets.QMdiSubWindow):
    def __init__(self, matcher_model, matcher_results_view, onclose_function):
        super(MatcherSubwindow, self).__init__()
        self.setWidget(matcher_results_view)
        self.model = matcher_model
        self._onclose = onclose_function

    def closeEvent(self, event):
        if self._onclose():
            event.accept()
        else:
            event.ignore()


class BackgroundThread(QThread):

    on_exception = pyqtSignal([str])

    def __init__(self, view, function, message=None,
                 on_end_function=None, on_exception_function=None):
        QThread.__init__(self)
        self._view = view
        self._function = function
        self._message = message
        self._on_end_function = on_end_function
        self._on_exception_function = on_exception_function

    def run(self):
        try:
            self._function()
        except Exception as e:
            LOGGER.exception(str(e))
            self.on_exception.emit(str(e))

    def start(self):
        self.finished.connect(self._on_end)
        self.on_exception.connect(self._on_exception)
        super(BackgroundThread, self).start()
        self._view.show_background_task_dialog(self._message)

    def _on_end(self):
        self._view.hide_background_task_dialog()
        self._on_end_function()

    def _on_exception(self, exception_message):
        self._view.hide_background_task_dialog()
        self._view.show_error_dialog("Error", exception_message)
        self._on_exception_function(exception_message)
