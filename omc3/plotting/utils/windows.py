"""
Plot Window
------------

In this module different classes are defined, allowing to put plots 
manually into windows.
These windows are QT-based, created with ``qtpy'' which allows to use 
either PySide(2 or 6) or PyQt(5 or 6), depending on which it installed on the system.
As the ``qtpy`` library is optional, there are some checks to make sure 
the imports do not fail, but then the classes cannot be used.
To check if QtPy is installed, either run :meth:`omc3.plotting.plot_window.is_qtpy_installed`
or try to initialize one of the windows, which will fail with a TypeError, 
as they want to call QApplication which is set to `None`.

An exception to all of this is  :meth:`omc3.plotting.plot_window.create_pyplot_window_from_fig`,
which allows to create a `pyplot` handled window from an already existing figure.
This way, figures can be created without manager (which makes them resource friendlier)
and can either be added to a QT-Window or, if not installed, opened by `pyplot`.
"""
import sys
from typing import Dict, List, Tuple

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)

try:
    from qtpy.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget)
except ImportError as e:
    LOG.debug(f"Could not import QtPy: {str(e)}")
    QMainWindow, QApplication, QVBoxLayout, QWidget, QTabWidget = None, None, None, object, object
else:
    matplotlib.use('Qt5agg')


def is_qtpy_installed():
    """Returns True if QtPy is installed."""
    return QMainWindow is not None


def log_no_qtpy_many_windows():
    """Logs a warning that QtPy is not installed and many windows will be opened."""

    LOG.warning(
        "QtPy is not installed. "
        "Plots will be shown in individual windows. "
        "Install QtPy for a more organized representation. "
    )


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
        self.app = QApplication(sys.argv)
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


def create_pyplot_window_from_fig(fig: Figure, title: str = None):
    """Creates a window from the given figure, which is managed by pyplot. 
    This is similar to how figures behave when created with `pyplot.figure()`,
    but you can crate the figure instance first and the manager later.

    Caveat: Uses private functions of pyplot.


    Args:
        fig (Figure): figure to be managed by pyplot. 
        title (str): Title of the window.
    """
    if fig.canvas.manager is not None:
        raise AttributeError('Figure already has a manager, cannot create a new one')

    allnums = plt.get_fignums()
    next_num = max(allnums) + 1 if allnums else 1
    manager = plt._get_backend_mod().new_figure_manager_given_figure(next_num, fig)
    plt._pylab_helpers.Gcf._set_new_active_manager(manager)
    if title is not None:
        manager.set_window_title(title)
