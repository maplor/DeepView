import os
import logging
import subprocess
import sys
from functools import cached_property
from pathlib import Path
from typing import List
import qdarkstyle

from PySide6.QtWidgets import QMessageBox, QMenu, QWidget, QMainWindow
from PySide6 import QtCore
from PySide6.QtGui import QIcon, QAction
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

from deepview.gui.tabs.create_training_dataset import CreateTrainingDataset
from deepview.gui.tabs.open_project import OpenProject
from deepview.gui.tabs.train_network import TrainNetwork
# from deepview.gui.tabs.visualize_gps import GPSDisplayer
# from deepview.gui.tabs.IMU_GPS_interact import GPSIMU_Interaction
# from deepview.gui.tabs.evaluate_network import EvaluateNetwork
# from deepview.gui.tabs.label_data import LabelData
# from deepview.gui.tabs.interaction_plot import InteractionPlot
from deepview.gui.tabs.label_with_interactive_plot import LabelWithInteractivePlotTab
from deepview.gui.tabs.supervised_learning_new_labels import SupervisedLearningNewLabels
from deepview.gui.tabs.supervised_cl import SupervisedCLTab


class MainWindow(QMainWindow):
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
    def toolbar(self):
        if self._toolbar is None:
            self._toolbar = self.addToolBar("File")
        return self._toolbar

    @cached_property
    def settings(self):
        return QtCore.QSettings()

    def load_settings(self):
        filenames = self.settings.value("recent_files") or []
        for filename in filenames:
            self.add_recent_filename(filename)

    def save_settings(self):
        recent_files = []
        for action in self.recentfiles_menu.actions()[::-1]:
            recent_files.append(action.text())
        self.settings.setValue("recent_files", recent_files)

    def add_recent_filename(self, filename):
        actions = self.recentfiles_menu.actions()
        filenames = [action.text() for action in actions]
        if filename in filenames:
            return
        action = QAction(filename, self)
        before_action = actions[0] if actions else None
        self.recentfiles_menu.insertAction(before_action, action)

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

    def create_actions(self, names):
        # Creating action using the first constructor
        self.newAction = QAction(self)
        self.newAction.setText("&New Project...")

        self.newAction.setIcon(
            QIcon(os.path.join(BASE_DIR, "assets", "icons", names[0]))
        )
        self.newAction.setShortcut("Ctrl+N")
        self.newAction.setStatusTip("Create a new project...")

        self.newAction.triggered.connect(self._create_project)

        # Creating actions using the second constructor
        self.openAction = QAction("&Open...", self)
        self.openAction.setIcon(
            QIcon(os.path.join(BASE_DIR, "assets", 'first.jpeg'))
        )
        self.openAction.setShortcut("Ctrl+O")
        self.openAction.setStatusTip("Open a project...")
        self.openAction.triggered.connect(self._open_project)

        self.saveAction = QAction("&Save", self)
        self.exitAction = QAction("&Exit", self)

        self.lightmodeAction = QAction("&Light theme", self)
        self.lightmodeAction.triggered.connect(self.lightmode)
        self.darkmodeAction = QAction("&Dark theme", self)
        self.darkmodeAction.triggered.connect(self.darkmode)

        self.helpAction = QAction("&Help", self)
        self.helpAction.setIcon(
            QIcon(os.path.join(BASE_DIR, "assets", "second.jpeg"))
        )
        self.helpAction.setStatusTip("Ask for help...")
        self.helpAction.triggered.connect(self._ask_for_help)

        self.aboutAction = QAction("&Learn DLC", self)
        self.aboutAction.triggered.connect(self._learn_dlc)

        # self.check_updates = QAction("&Check for Updates...", self)
        # self.check_updates.triggered.connect(_check_for_updates)

    def create_menu_bar(self):
        menu_bar = self.menuBar()

        # File menu
        self.file_menu = QMenu("&File", self)
        menu_bar.addMenu(self.file_menu)

        self.file_menu.addAction(self.newAction)
        self.file_menu.addAction(self.openAction)

        self.recentfiles_menu = self.file_menu.addMenu("Open Recent")
        self.recentfiles_menu.triggered.connect(
            lambda a: self._update_project_state(a.text(), True)
        )
        self.file_menu.addAction(self.saveAction)
        self.file_menu.addAction(self.exitAction)

        # View menu
        view_menu = QMenu("&View", self)
        mode = view_menu.addMenu("Appearance")
        menu_bar.addMenu(view_menu)
        mode.addAction(self.lightmodeAction)
        mode.addAction(self.darkmodeAction)

        # Help menu
        help_menu = QMenu("&Help", self)
        menu_bar.addMenu(help_menu)
        help_menu.addAction(self.helpAction)
        help_menu.adjustSize()
        # help_menu.addAction(self.check_updates)
        help_menu.addAction(self.aboutAction)

    def update_menu_bar(self):
        self.file_menu.removeAction(self.newAction)
        self.file_menu.removeAction(self.openAction)

    def create_toolbar(self):
        self.toolbar.addAction(self.newAction)
        self.toolbar.addAction(self.openAction)
        self.toolbar.addAction(self.helpAction)

    def remove_action(self):
        self.toolbar.removeAction(self.newAction)
        self.toolbar.removeAction(self.openAction)
        self.toolbar.removeAction(self.helpAction)

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

    def refresh_active_tab(self):
        active_tab = self.tab_widget.currentWidget()

        tab_label = self.tab_widget.tabText(self.tab_widget.currentIndex())


        widget_to_attribute_map = {
            QtWidgets.QSpinBox: "setValue",
            components.TestfileSpinBox: "setValue",
            components.TrainingSetSpinBox: "setValue",
            QtWidgets.QLineEdit: "setText",
        }

        def _attempt_attribute_update(widget_name, updated_value):
            try:
                widget = getattr(active_tab, widget_name)
                method = getattr(widget, widget_to_attribute_map[type(widget)])
                self.logger.debug(
                    f"Setting {widget_name}={updated_value} in tab '{tab_label}'"
                )
                method(updated_value)
            except AttributeError:
                pass

        _attempt_attribute_update("testfile", self.testfile)
        _attempt_attribute_update("cfg_line", self.config)

    def add_tabs(self):
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setContentsMargins(0, 20, 0, 0)
        self.create_training_dataset = CreateTrainingDataset(
            root=self,
            parent=None,
            h1_description="DeepView - Step 1. Create training dataset",
        )
        self.train_network = TrainNetwork(
            root=self, parent=None,
            h1_description="Step 2. Train unsupervised learning network",
        )

        self.label_with_interactive_plot = LabelWithInteractivePlotTab(
            root=self,
            parent=None,
            h1_description="Step 3. Label with Interaction Plot",
        )
        self.supervised_contrastive_learning = LabelWithInteractivePlotTab(
            root=self,
            parent=None,
            h1_description="Step 4. Apply Supervised Contrastive Learning",
        )
        self.supervised_learning_gui = SupervisedLearningNewLabels(
            root=self,
            parent=None,
            h1_description="Step 6. Label with Interaction Plot",
        )
        self.supervised_cl = SupervisedCLTab(
            root=self,
            parent=None,
            h1_description="Step 7. SupervisedCL",
        )

        # self.tab_widget.addTab(self.manage_project, "Manage project")
        # self.tab_widget.addTab(self.extract_frames, "Extract frames")
        # self.tab_widget.addTab(self.label_frames, "Label frames")
        self.tab_widget.addTab(self.create_training_dataset, "Create training dataset")
        self.tab_widget.addTab(self.train_network, "Train network")
        # self.tab_widget.addTab(self.evaluate_network, "Evaluate network")
        # self.tab_widget.addTab(self.mad_gui, "Label data")
        # self.tab_widget.addTab(self.interaction_plot, "Interaction plot")
        # self.tab_widget.addTab(self.show_gps, "Display GPS on the map")
        # self.tab_widget.addTab(self.imu_gps_interact, "IMU GPS interaction")
        self.tab_widget.addTab(self.label_with_interactive_plot, "Label with interactive plot")
        self.tab_widget.addTab(self.supervised_learning_gui, "Supervised learning with new labels")
        self.tab_widget.addTab(self.supervised_cl, "SupervisedCL")
        # self.tab_widget.addTab(self.analyze_videos, "Analyze videos")
        # self.tab_widget.addTab(
        #     self.unsupervised_id_tracking, "Unsupervised ID Tracking (*)"
        # )
        # self.tab_widget.addTab(self.create_videos, "Create videos")
        # self.tab_widget.addTab(
        #     self.extract_outlier_frames, "Extract outlier frames (*)"
        # )
        # self.tab_widget.addTab(self.refine_tracklets, "Refine tracklets (*)")
        # self.tab_widget.addTab(self.modelzoo, "Model Zoo")
        # self.tab_widget.addTab(self.video_editor, "Video editor (*)")

        # if not self.is_multianimal:
        #     self.refine_tracklets.setEnabled(False)
        # self.unsupervised_id_tracking.setEnabled(self.is_transreid_available())

        self.setCentralWidget(self.tab_widget)
        self.tab_widget.currentChanged.connect(self.refresh_active_tab)

    def is_transreid_available(self):
        if self.is_multianimal:
            try:
                # from deeplabcut.pose_tracking_pytorch import transformer_reID

                return True
            except ModuleNotFoundError:
                return False
        else:
            return False

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
