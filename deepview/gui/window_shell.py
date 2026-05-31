import os
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget

from deepview.gui import BASE_DIR, components


class WindowShellMixin:
    def print_to_status_bar(self, text):
        self.status_bar.showMessage(text)
        self.status_bar.repaint()

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

    def closeEvent(self, event):
        self.logger.info("Exiting")
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
