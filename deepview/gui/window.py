import logging
import sys

from PySide6.QtWidgets import QMessageBox, QMainWindow
from PySide6 import QtCore
from PySide6 import QtWidgets

# --------------own packages------------------

# import deeplabcut
# from deeplabcut import VERSION
# from deeplabcut.gui import BASE_DIR, utils
# from deeplabcut.gui.tabs import *
from deepview.gui.widgets import StreamReceiver, StreamWriter
# from napari_deeplabcut import misc

# from deepview.gui.tabs.visualize_gps import GPSDisplayer
# from deepview.gui.tabs.IMU_GPS_interact import GPSIMU_Interaction
# from deepview.gui.tabs.evaluate_network import EvaluateNetwork
# from deepview.gui.tabs.label_data import LabelData
# from deepview.gui.tabs.interaction_plot import InteractionPlot
from deepview.gui.window_menu import WindowMenuMixin
from deepview.gui.window_project import WindowProjectMixin
from deepview.gui.window_shell import WindowShellMixin
from deepview.gui.window_tabs import WindowTabsMixin
from deepview.gui.window_theme import WindowThemeMixin


class MainWindow(
    WindowMenuMixin,
    WindowProjectMixin,
    WindowShellMixin,
    WindowTabsMixin,
    WindowThemeMixin,
    QMainWindow,
):
    config_loaded = QtCore.Signal()
    video_type_ = QtCore.Signal(str)
    video_files_ = QtCore.Signal(set)

    def __init__(self, app):
        super(MainWindow, self).__init__()
        self.app = app  # style.qss
        screen_size = app.screens()[0].size()
        self.screen_width = screen_size.width()
        self.screen_height = screen_size.height()

        self.logger = logging.getLogger("GUI")

        self.config = None
        self.loaded = False

        # self.shuffle_value = 1  # 可以改成testfile
        self.testfile = ''  # 默认为空
        self.trainingset_index = 0
        self.filetype = "csv"  # todo
        self.files = set()

        # self.default_set()  # default text in the textbox

        self._generate_welcome_page()
        self.window_set()
        self.default_set()  # default text in the textbox

        names = ["new_project.png", "open.png", "help.png"]
        self.create_actions(names)
        self.create_menu_bar()
        self.load_settings()
        self._toolbar = None
        self.create_toolbar()

        # Thread-safe Stdout redirector
        self.writer = StreamWriter()
        sys.stdout = self.writer
        self.receiver = StreamReceiver(self.writer.queue)
        self.receiver.new_text.connect(self.print_to_status_bar)

        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setMaximum(0)
        self._progress_bar.hide()
        self.status_bar.addPermanentWidget(self._progress_bar)

    def _ask_for_help(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Ask for help")
        dlg.setText(
            """Ask our community for help on <a href='https://forum.image.sc/tag/deepview'>the forum</a>!"""
        )
        _ = dlg.exec()

    def _learn_dlc(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Learn DLC")
        dlg.setText(
            """Learn DLC with <a href='https://deepview.github.io/UseOverviewGuide.html'>our docs and how-to guides</a>!"""
        )
        _ = dlg.exec()
