import pytest
from matplotlib.figure import Figure

from omc3.plotting.utils.windows import PlotWidget, SimpleTabWindow, TabWidget, VerticalTabWindow


@pytest.mark.basic
def test_plot_widget(monkeypatch):
    # Preparation ---
    monkeypatch.setattr("omc3.plotting.utils.windows.QVBoxLayout", MockLayout)
    monkeypatch.setattr("omc3.plotting.utils.windows.FigureCanvas", MockFigureCanvas)
    monkeypatch.setattr("omc3.plotting.utils.windows.NavigationToolbar", MockNavigationToolbar)

    class MockPlotWidget(PlotWidget, MockQWidget):
        pass

    figures = (Figure(), Figure(), Figure())
    my_title = "Hello OMC!"

    # Execution ---
    widget = MockPlotWidget(*figures, title=my_title)

    # Assert ---
    assert widget.title == my_title
    assert isinstance(widget._layout, MockLayout)
    assert len(widget._layout.widgets) == len(figures) * 2
    for idx, w in enumerate(widget._layout.widgets):
        if idx % 2:
            assert isinstance(w, MockNavigationToolbar)
        else:
            assert isinstance(w, MockFigureCanvas)
            assert w.figure == figures[idx // 2]


@pytest.mark.basic
def test_tab_widget():
    # Preparation ---
    class MockTabWidget(TabWidget, MockQTabWidget):
        pass

    tabs = [MockQWidget("tab1"), MockQWidget("tab2"), MockQWidget("tab3")]
    my_title = "Hello OMC!"

    # Execution ---
    widget = MockTabWidget(title=my_title)
    for tab in tabs:
        widget.add_tab(tab)

    # Assert ---
    assert widget.title == my_title
    assert len(widget.tabs) == len(tabs)
    assert len(widget.added_tabs) == len(tabs)
    for idx, (title, tab) in enumerate(widget.added_tabs.items()):
        assert tabs[idx] == widget.tabs[title]
        assert tabs[idx] == tab


@pytest.mark.basic
@pytest.mark.parametrize('WindowClass', (SimpleTabWindow, VerticalTabWindow))
def test_tab_window(monkeypatch, WindowClass):
    # Preparation ---
    monkeypatch.setattr("omc3.plotting.utils.windows.QApplication", MockQApplication)
    monkeypatch.setattr("omc3.plotting.utils.windows.QMainWindow", MockQMainWindow)
    monkeypatch.setattr("omc3.plotting.utils.windows.QTabWidget", MockQTabWidget)

    class MockTabWidget(TabWidget, MockQTabWidget):
        pass

    monkeypatch.setattr("omc3.plotting.utils.windows.TabWidget", MockTabWidget)


    tabs = [MockQWidget("tab1"), MockQWidget("tab2"), MockQWidget("tab3")]
    my_title = "Hello OMC!"
    my_size = (800, 600)

    # Execution ---
    window = WindowClass(title=my_title, size=my_size)
    for tab in tabs:
        window.add_tab(tab)

    # Assert Main Window ---
    assert window.main_window.title == my_title
    assert window.main_window.size == my_size
    assert window.main_window.shown

    # Assert Tab Widget ---
    assert isinstance(window.tabs_widget, TabWidget)
    assert len(window.tabs_widget.added_tabs) == len(tabs)
    for tab, tab_added in zip(tabs, window.tabs_widget.added_tabs.values()):
        assert tab == tab_added
    assert window.tabs_widget.position == (MockQTabWidget.West if (WindowClass == VerticalTabWindow) else 0)

    # Assert App ---
    assert not window.app.executed
    window.show()
    assert window.app.executed


# Mock Classes -----------------------------------------------------------------

class MockQWidget:

    def __init__(self, title = None):
        self.layout = None
        self.title = title

    def setLayout(self, layout):
        self.layout = layout


class MockQTabWidget:
    West = 1
    Center = 2
    East = 3

    def __init__(self):
        self.added_tabs = {}
        self.position = 0

    def addTab(self, tab, tab_title):
        self.added_tabs[tab_title] = tab

    def setTabPosition(self, position):
        self.position = position


class MockLayout:

    def __init__(self) -> None:
        self.widgets = []


    def addWidget(self, widget):
        self.widgets.append(widget)


class MockFigureCanvas:

    def __init__(self, figure):
        self.figure = figure


class MockNavigationToolbar:

    def __init__(self, canvas, parent=None):
        self.canvas = canvas
        self.parent = parent


class MockQApplication:

    def __init__(self, *args, **kwargs):
        self.executed = False

    def exec(self):
        self.executed = True


class MockQMainWindow:

    def __init__(self):
        self.title = None
        self.central_widget = None
        self.size = None
        self.shown = False

    def setWindowTitle(self, title):
        self.title = title

    def setCentralWidget(self, widget):
        self.central_widget = widget

    def resize(self, width, height):
        self.size = (width, height)

    def show(self):
        self.shown = True
