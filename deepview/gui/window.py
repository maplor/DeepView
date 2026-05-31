import os
import logging
import subprocess
import sys
from typing import List
import qdarkstyle

from PySide6.QtWidgets import QMessageBox, QMainWindow
from PySide6 import QtCore
from PySide6 import QtWidgets

# --------------own packages------------------
from deepview.gui.tabs import ProjectCreator

# import deeplabcut
from deepview import auxiliaryfunctions
# from deeplabcut import VERSION
# from deeplabcut.gui import BASE_DIR, utils
# from deeplabcut.gui.tabs import *
from deepview.gui.widgets import StreamReceiver, StreamWriter
# from napari_deeplabcut import misc

from deepview.gui.tabs.open_project import OpenProject
# from deepview.gui.tabs.visualize_gps import GPSDisplayer
# from deepview.gui.tabs.IMU_GPS_interact import GPSIMU_Interaction
# from deepview.gui.tabs.evaluate_network import EvaluateNetwork
# from deepview.gui.tabs.label_data import LabelData
# from deepview.gui.tabs.interaction_plot import InteractionPlot
from deepview.gui.window_menu import WindowMenuMixin
from deepview.gui.window_shell import WindowShellMixin
from deepview.gui.window_tabs import WindowTabsMixin


class MainWindow(WindowMenuMixin, WindowShellMixin, WindowTabsMixin, QMainWindow):
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

    @property
    def cfg(self):
        try:
            cfg = auxiliaryfunctions.read_config(self.config)
        except TypeError:
            cfg = {}
        return cfg

    @property
    def project_folder(self) -> str:
        return self.cfg.get("project_path", os.path.expanduser("~\Desktop"))

    # @property
    # def is_multianimal(self) -> bool:
    #     return bool(self.cfg.get("multianimalproject"))
    #
    # @property
    # def all_bodyparts(self) -> List:
    #     if self.is_multianimal:
    #         return self.cfg.get("multianimalbodyparts")
    #     else:
    #         return self.cfg["bodyparts"]

    # @property
    # def all_individuals(self) -> List:
    #     if self.is_multianimal:
    #         return self.cfg.get("individuals")
    #     else:
    #         return [""]

    # @property
    # def pose_cfg_path(self) -> str:
    #     try:
    #         return os.path.join(
    #             self.cfg["project_path"],
    #             auxiliaryfunctions.get_model_folder(
    #                 self.cfg["TrainingFraction"][int(self.trainingset_index)],
    #                 int(self.shuffle_value),
    #                 self.cfg,
    #             ),
    #             "train",
    #             "pose_cfg.yaml",
    #         )
    #     except FileNotFoundError:
    #         return str(Path(deepview.__file__).parent / "pose_cfg.yaml")

    @property
    def inference_cfg_path(self) -> str:
        return os.path.join(
            self.cfg["project_path"],
            auxiliaryfunctions.get_model_folder(
                self.cfg["TrainingFraction"][int(self.trainingset_index)],
                int(self.testfile),
                self.cfg,
            ),
            "test",
            "inference_cfg.yaml",
        )

    def update_cfg(self, text):
        self.root.config = text
        self.unsupervised_id_tracking.setEnabled(self.is_transreid_available())

    # def update_shuffle(self, value):
    #     self.shuffle_value = value
    #     self.logger.info(f"Shuffle set to {self.shuffle_value}")

    def update_testfile(self, value):
        self.testdata = value
        self.logger.info(f"Select test set {self.testdata}")

    @property
    def file_type(self):
        return self.file_type

    @file_type.setter
    def file_type(self, ext):
        self.filetype = ext
        self.file_type_.emit(ext)
        self.logger.info(f"File type set to {self.file_type}")

    @property
    def text_files(self):
        return self.files

    @text_files.setter
    def video_files(self, video_files):
        self.files = set(video_files)
        self.video_files_.emit(self.files)
        self.logger.info(f"Files (.csv) selected to analyze:\n{self.files}")

    def _update_project_state(self, config, loaded):
        self.config = config
        self.loaded = loaded
        if loaded:
            self.add_recent_filename(self.config)
            self.add_tabs()

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

    def _create_project(self):
        dlg = ProjectCreator(self)
        dlg.show()

    def _open_project(self):
        open_project = OpenProject(self)
        open_project.load_config()
        if not open_project.config:
            return

        open_project.loaded = True
        self._update_project_state(
            open_project.config,
            open_project.loaded,
        )
        # print('Todo: open an existing project...')

    # def _goto_superanimal(self):
    #     self.tab_widget = QtWidgets.QTabWidget()
    #     self.tab_widget.setContentsMargins(0, 20, 0, 0)
    #     self.modelzoo = ModelZoo(
    #         root=self, parent=None, h1_description="DeepLabCut - Model Zoo"
    #     )
    #     self.tab_widget.addTab(self.modelzoo, "Model Zoo")
    #     self.setCentralWidget(self.tab_widget)

    def load_config(self, config):
        self.config = config
        self.config_loaded.emit()
        print(f'Project "{self.cfg["Task"]}" successfully loaded.')

    def darkmode(self):
        dark_stylesheet = qdarkstyle.load_stylesheet_pyside2()
        self.app.setStyleSheet(dark_stylesheet)
        try:
            self.label_with_interactive_plot.update_theme('dark')
            self.supervised_cl.update_theme('dark')
        except AttributeError:
            pass

        names = ["new_project2.png", "open2.png", "help2.png"]
        self.remove_action()
        self.create_actions(names)
        self.update_menu_bar()
        self.create_toolbar()

    def lightmode(self):
        from qdarkstyle.light.palette import LightPalette

        style = qdarkstyle.load_stylesheet(palette=LightPalette)
        self.app.setStyleSheet(style)
        try:
            self.label_with_interactive_plot.update_theme('light')
            self.supervised_cl.update_theme('light')
        except AttributeError:
            pass

        names = ["new_project.png", "open.png", "help.png"]
        self.remove_action()
        self.create_actions(names)
        self.create_toolbar()
        self.update_menu_bar()
