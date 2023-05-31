"""
Plot Window
------------

Create QT-based windows that can store plots and allow switching between them
via tabs.

"""
from typing import List, Tuple, Dict

import matplotlib
from matplotlib.figure import Figure

from omc3.utils import logging_tools


matplotlib.use('qt5agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

LOG = logging_tools.get_logger(__name__)

try:
    # Beware: this also raises an import error, if the LD_LIBRARY_PATH variable is not set correctly
    # LD_LIBRARY_PATH=YOUR_PYTHON_ENV/lib/python3.XX/site-packages/PySide2/Qt/lib/
    from PySide2.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout
except ImportError as e:
    LOG.debug(f"Could not import PySide2: {str(e)}")
    QMainWindow, QApplication, QVBoxLayout, QWidget, QTabWidget = None, None, None, object, object


class PlotWidget(QWidget):
    """ A derived QWidget that contains Matplotlib-Figures,
     stacked vertically, each with its own toolbar."""

    def __init__(self, *figures: Figure, title: str):
        """Creates Widget.

        Args:
            *figures (Figure): Figures to be contained in the widget
            title (str): Name of the widget
        """
        super().__init__()
        self.title: str = title
        self.figures: Tuple[Figure] = figures
        self._canvas_toolbar_handles: List[Tuple[FigureCanvas, NavigationToolbar]] = []

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        for figure in figures:
            canvas = FigureCanvas(figure)
            toolbar = NavigationToolbar(canvas, self)

            self._layout.addWidget(canvas)
            self._layout.addWidget(toolbar)
            self._canvas_toolbar_handles.append((canvas, toolbar))


class TabWidget(QTabWidget):
    """A simple tab widget, that in addition
    to QTabWidget keeps track of the tabs via dictionary
    and can have a title."""

    def __init__(self, title: str = None):
        super().__init__()
        self.title: str = title
        self.tabs: Dict[str, QWidget] = {}

    def add_tab(self, widget: QWidget):
        self.tabs[widget.title] = widget
        self.addTab(widget, widget.title)


class SimpleTabWindow:
    """ A simple window that contains a single Tab-Widget,
    allowing the user to add tabs to it. """
    def __init__(self, title: str = "Simple Tab Window", size=(1280, 900)):
        """

        Args:
            title (str): Title of the created Window
            size (Tuple[int, int]): Size of the created window.
        """
        self.app = QApplication([])
        self.main_window = QMainWindow()
        self.main_window.__init__()
        self.main_window.setWindowTitle(title)
        self.current_window = -1
        self.tabs_widget = TabWidget()
        self.main_window.setCentralWidget(self.tabs_widget)
        self.main_window.resize(*size)
        self.main_window.show()

    def add_tab(self, widget: QWidget):
        """ Add a new tab made from the given widget.

        Args:
            widget (QWidget): Widget to add.
        """
        self.tabs_widget.add_tab(widget)

    def show(self):
        self.app.exec_()


class VerticalTabWindow(SimpleTabWindow):
    """ A Window in which the tabs are aligned vertically on the left-hand side.
    This window assumes that you may want to have tabs within tabs,
    so the convenience function `add_to_tab` is implemented, which allows
    you to add widgets directly to an already existing tab. """

    def __init__(self, name: str = "Vertical Tab Window", size=(1280, 900)):
        super().__init__(name, size)
        self.tabs_widget.setTabPosition(QTabWidget.West)

    def add_to_tab(self, widget: QWidget, tab: str):
        """ Add `widget` to the tab with the given
        `tab` identifier.

        Args:
            widget (QWidget): Widget to add.
            tab (str): Title/Name of the tab .

        """
        self.tabs_widget.tabs[tab].add_tab(widget)

    def show(self):
        self.app.exec_()
