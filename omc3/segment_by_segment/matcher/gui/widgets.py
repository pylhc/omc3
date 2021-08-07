import logging
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt
from omc3.segment_by_segment.matcher.gui import constants


class FileSelectionDialogWidget(QtWidgets.QWidget):
    def __init__(self, label_text="", parent=None):
        super(FileSelectionDialogWidget, self).__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        label = QtWidgets.QLabel(label_text)
        layout.addWidget(label)
        self._text_area = QtWidgets.QLineEdit()
        layout.addWidget(self._text_area)
        self._select_button = QtWidgets.QPushButton("Choose...")
        self._select_button.clicked.connect(self._select_file_action)
        layout.addWidget(self._select_button)

    def get_selected_file(self):
        return self._text_area.text()

    def set_selected_file(self, text):
        self._text_area.setText(text)

    def _select_file_action(self):
        file_selection_dialog = QtWidgets.QFileDialog()
        self._text_area.setText(file_selection_dialog.getExistingDirectory())


class FileSelectionComboWidget(QtWidgets.QWidget):

    added_items = []

    def __init__(self, label_text="", initial_list=[], parent=None):
        super(FileSelectionComboWidget, self).__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        label = QtWidgets.QLabel(label_text)
        layout.addWidget(label)
        self._items_combo = QtWidgets.QComboBox()
        layout.addWidget(self._items_combo)
        self._items_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToContents
        )
        self._items_combo.addItems(initial_list + self.added_items)
        self._items_combo.currentTextChanged.connect(self._item_change_action)
        self._select_button = QtWidgets.QPushButton("Add...")
        self._select_button.clicked.connect(self._select_file_action)
        # To be replaced by some action:
        self.on_item_change = lambda selected_text: None
        layout.addWidget(self._select_button)

    def get_selected_file(self):
        return self._items_combo.currentText()

    def add_item(self, text):
        self.added_items.append(text)
        self._items_combo.insertItem(0, text)
        self._items_combo.setCurrentIndex(0)

    def reload_items(self, text_list):
        self._items_combo.clear()
        self._items_combo.addItem("")
        for text in text_list:
            self._items_combo.addItem(text)
        self._items_combo.setCurrentIndex(0)

    def _item_change_action(self, selected_text):
        if not selected_text.strip() == "":
            self.on_item_change(selected_text)

    def _select_file_action(self):
        file_selection_dialog = QtWidgets.QFileDialog()
        self.add_item(file_selection_dialog.getExistingDirectory())


class InitialConfigPopup(QtWidgets.QDialog):

    def __init__(self, lhc_mode, match_path, parent=None):
        super(InitialConfigPopup, self).__init__(parent)
        self.resize(655, 90)
        main_layout = QtWidgets.QVBoxLayout(self)

        self._lhc_mode_combo = QtWidgets.QComboBox()
        self._lhc_mode_combo.addItems(constants.LHC_MODES)
        if lhc_mode is not None:
            if lhc_mode not in constants.LHC_MODES:
                raise ValueError("Invalid lhc mode, must be one of " +
                                 str(constants.LHC_MODES))
            else:
                self._lhc_mode_combo.setCurrentIndex(
                    constants.LHC_MODES.index(lhc_mode)
                )
        main_layout.addWidget(self._lhc_mode_combo)

        self._file_selector = FileSelectionDialogWidget()
        main_layout.addWidget(self._file_selector)
        if match_path is not None:
            self._file_selector.set_selected_file(match_path)

        buttons_layout = QtWidgets.QHBoxLayout()
        accept_button = QtWidgets.QPushButton("Accept")
        accept_button.clicked.connect(self.accept)
        buttons_layout.addWidget(accept_button)
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_button)
        main_layout.addLayout(buttons_layout)

    def get_selected_file(self):
        return str(self._file_selector.get_selected_file())

    def get_selected_lhc_mode(self):
        return str(self._lhc_mode_combo.currentText())


class LogDialog(QtWidgets.QDialog, logging.Handler):

    _update_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent=parent)
        logging.Handler.__init__(self)

        self.resize(855, 655)
        layout = QtWidgets.QVBoxLayout()
        self._log_text = QtWidgets.QPlainTextEdit(parent)
        self._log_text.setReadOnly(True)
        layout.addWidget(self._log_text)
        self.setLayout(layout)

        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"
        )
        formatter.datefmt = '%d/%m/%Y %H:%M:%S'
        self.setFormatter(formatter)
        self.setLevel(logging.DEBUG)

        self._update_signal.connect(self._update_log, Qt.QueuedConnection)

    def _update_log(self, msg):
        self._log_text.appendPlainText(msg)

    def emit(self, record):
        msg = self.format(record)
        self._update_signal.emit(msg)


class MinimalButton(QtWidgets.QPushButton):
    def __init__(self, text, parent=None):
        super(MinimalButton, self).__init__(text, parent=parent)
        double = text.count('&&')
        text = text.replace('&', '') + ('&' * double)
        width = self.fontMetrics().boundingRect(text).width() + 7
        self.setMaximumWidth(width)
