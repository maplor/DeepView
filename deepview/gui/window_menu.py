import os
from functools import cached_property

from PySide6 import QtCore
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QMenu

from deepview.gui import BASE_DIR


class WindowMenuMixin:
    @cached_property
    def settings(self):
        return QtCore.QSettings()

    @property
    def toolbar(self):
        if self._toolbar is None:
            self._toolbar = self.addToolBar("File")
        return self._toolbar

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