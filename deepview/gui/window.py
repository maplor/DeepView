import os
import logging
import subprocess
import sys
from pathlib import Path
from typing import List
import qdarkstyle

from PySide6.QtWidgets import QMessageBox, QWidget, QMainWindow
from PySide6 import QtCore
from PySide6.QtGui import QIcon
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt

# --------------own packages------------------
from deepview.gui import components
from deepview.gui.tabs import ProjectCreator
from deepview.gui import BASE_DIR

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
from deepview.gui.window_tabs import WindowTabsMixin


class MainWindow(WindowMenuMixin, WindowTabsMixin, QMainWindow):
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

    def print_to_status_bar(self, text):
        self.status_bar.showMessage(text)
        self.status_bar.repaint()


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

    def window_set(self):
        self.setWindowTitle("DeepView")

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#ffffff"))
        self.setPalette(palette)

        icon = os.path.join(BASE_DIR, "assets", "logo.png")
        self.setWindowIcon(QIcon(icon))

        self.status_bar = self.statusBar()
        self.status_bar.setObjectName("Status Bar")
        self.status_bar.showMessage("www.Todo.org")

    def _generate_welcome_page(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        self.layout.setSpacing(30)

        title = components._create_label_widget(
            f"Welcome to the DeepView Project Manager GUI TODO:version!",
            "font:bold; font-size:18px;",
            margins=(0, 30, 0, 0),
        )
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title)

        image_widget = QtWidgets.QLabel(self)
        image_widget.setAlignment(Qt.AlignCenter)
        image_widget.setContentsMargins(0, 0, 0, 0)
        logo = os.path.join(BASE_DIR, "assets", "third.png")
        pixmap = QtGui.QPixmap(logo)
        image_widget.setPixmap(
            pixmap.scaledToHeight(400, QtCore.Qt.SmoothTransformation)
        )
        self.layout.addWidget(image_widget)

        description = "DeepView™ is an open source tool for activity recognition using time-series data with deep learning.\nMaekawa, Otsuka, and Xia | http://www.hara.org\n\n To get started,  create a new project, load an existing one."
        label = components._create_label_widget(
            description,
            "font-size:12px; text-align: center;",
            margins=(0, 0, 0, 0),
        )
        label.setMinimumWidth(400)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(label)

        self.layout_buttons = QtWidgets.QHBoxLayout()
        self.layout_buttons.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        self.create_project_button = QtWidgets.QPushButton("Create New Project")
        self.create_project_button.setFixedWidth(200)
        self.create_project_button.clicked.connect(self._create_project)  # Create-New-Project action

        self.load_project_button = QtWidgets.QPushButton("Load Project")
        self.load_project_button.setFixedWidth(200)
        self.load_project_button.clicked.connect(self._open_project)  # Load-Project action

        # self.run_superanimal_button = QtWidgets.QPushButton("Model Zoo")
        # self.run_superanimal_button.setFixedWidth(200)
        # self.run_superanimal_button.clicked.connect(self._goto_superanimal)

        self.layout_buttons.addWidget(self.create_project_button)
        self.layout_buttons.addWidget(self.load_project_button)
        # self.layout_buttons.addWidget(self.run_superanimal_button)

        self.layout.addLayout(self.layout_buttons)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

    def default_set(self):
        self.name_default = ""
        self.proj_default = ""
        self.exp_default = ""
        self.loc_default = str(Path.home())


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


    def closeEvent(self, event):
        print("Exiting...")
        answer = QtWidgets.QMessageBox.question(
            self,
            "Quit",
            "Are you sure you want to quit?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self.receiver.terminate()
            event.accept()
            self.save_settings()
        else:
            event.ignore()
            print("")
