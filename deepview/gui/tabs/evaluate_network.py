

#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import os
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QFileDialog, QLineEdit


from deepview.gui.components import (
    DefaultTab,
    TestfileSpinBox,
    _create_horizontal_layout,
    _create_label_widget,
    _create_vertical_layout,
)
# from deepview.gui.widgets import ConfigEditor
import deepview
# from deepview.utils import auxiliaryfunctions
from deepview.utils.auxiliaryfunctions import get_evaluation_folder

'''
button1:evaluate 单天或者某些天（todo）的数据，生成数据的representation，完成后保存representation
button2:在evaluate文件夹下作图：tsne，acc+cluster labels
'''

class EvaluateNetwork(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(EvaluateNetwork, self).__init__(root, parent, h1_description)

        # self.bodyparts_to_use = self.root.all_bodyparts

        self._set_page()

    def _set_page(self):
        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_horizontal_layout()
        # shuffle button is created here, change to selecting filename
        self._generate_layout_attributes(self.layout_attributes)  # 实现bottun
        self.main_layout.addLayout(self.layout_attributes)

        self.main_layout.addWidget(_create_label_widget(""))  # dummy text
        self.layout_additional_attributes = _create_vertical_layout()
        self._generate_additional_attributes(self.layout_additional_attributes)  # 实现bottun
        self.main_layout.addLayout(self.layout_additional_attributes)

        self.ev_nw_button = QtWidgets.QPushButton("Evaluate Network")
        self.ev_nw_button.setMinimumWidth(150)
        self.ev_nw_button.clicked.connect(self.evaluate_network)  # 实现bottun

        self.opt_button = QtWidgets.QPushButton("Plot example data")
        self.opt_button.setMinimumWidth(150)
        self.opt_button.clicked.connect(self.plot_maps)  # 实现bottun

        # self.edit_inferencecfg_btn = QtWidgets.QPushButton("Edit inference_cfg.yaml")
        # self.edit_inferencecfg_btn.setMinimumWidth(150)
        # self.edit_inferencecfg_btn.clicked.connect(self.open_inferencecfg_editor)

        self.main_layout.addWidget(self.ev_nw_button, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.opt_button, alignment=Qt.AlignRight)

    def _generate_layout_attributes(self, layout):
        # opt_text = QtWidgets.QLabel("Select test data")
        # self.shuffle = TestfileSpinBox(root=self.root, parent=self)

        # Create a QLineEdit to display the selected file path
        self.file_path_edit = QLineEdit(self)
        self.file_path_edit.setPlaceholderText("Selected File from labeled-data/iteration/CollectedData_*.pkl")
        self.file_path_edit.setReadOnly(True)

        # Create a QPushButton to open a file dialog for file selection
        self.select_file_button = QPushButton("Select test File", self)
        self.select_file_button.clicked.connect(self.open_file_dialog)

        layout.addWidget(self.file_path_edit)
        layout.addWidget(self.select_file_button)

    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_name, _ = QFileDialog.getOpenFileName(self, "Select test data", "", "All Files (*);;Text Files (*.txt)", options=options)

        if file_name:
            self.file_path_edit.setText(file_name)  # file_name为全路径
            # print(f"Selected File Path: {file_name}")

    def _generate_additional_attributes(self, layout):
        tmp_layout = _create_horizontal_layout(margins=(0, 0, 0, 0))

        self.plot_predictions = QtWidgets.QCheckBox(
            "Plot predictions (as in standard DeepView projects)"
        )
        self.plot_predictions.stateChanged.connect(self.update_plot_predictions)

        tmp_layout.addWidget(self.plot_predictions)

        # self.bodyparts_list_widget = BodypartListWidget(root=self.root, parent=self)
        # self.use_all_bodyparts = QtWidgets.QCheckBox("Compare all bodyparts")
        # self.use_all_bodyparts.stateChanged.connect(self.update_bodypart_choice)
        # self.use_all_bodyparts.setCheckState(Qt.Checked)
        #
        # tmp_layout.addWidget(self.use_all_bodyparts)
        layout.addLayout(tmp_layout)

        # layout.addWidget(self.bodyparts_list_widget, alignment=Qt.AlignLeft)

    def update_plot_predictions(self, s):
        if s == Qt.Checked:
            self.root.logger.info("Plot predictions ENABLED")
        else:
            self.root.logger.info("Plot predictions DISABLED")

    def evaluate_network(self):
        config = self.root.config

        # plot_predictions只是check，在_generate_additional_attributes生成
        plotting = self.plot_predictions.checkState() == Qt.Checked

        # file_path_edit: select data path in evaluation component
        try:
            testdata_path = self.file_path_edit.text()
        except:
            testdata_path = ''
        deepview.evaluate_network(
            config,
            testdata_path,
            plotting=plotting,
            show_errors=True,
        )
        # print('evaluation finished...')
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("The meta data is successfully created and ready to plot.")
        msg.setInformativeText(
            "Click the button 'plot' to visualize data."
        )

    def plot_maps(self):
        # shuffle = self.root.shuffle_value  #=1
        config = self.root.config  # project/config.yaml
        # test set生成图像并保存
        deepview.extract_save_all_maps(config, Indices=[0, 1, 2])

        # Display all images
        dest_folder = os.path.join(
            self.root.project_folder,
            str(
                get_evaluation_folder( self.root.cfg)
            ),
            "maps",
        )
        fig_paths = [
            os.path.join(dest_folder, file)
            for file in os.listdir(dest_folder)
            if file.endswith(".png")
        ]
        # 在这里plot figure并且展示在对话框内。
        canvas = GridCanvas(fig_paths, parent=self)
        canvas.show()

class GridCanvas(QtWidgets.QDialog):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        layout = QtWidgets.QVBoxLayout(self)
        self.figure = Figure()
        self.figure.patch.set_facecolor("None")
        self.grid = self.figure.add_gridspec(3, 3)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        for image_path, gridspec in zip(image_paths[:9], self.grid):
            ax = self.figure.add_subplot(gridspec)
            ax.set_axis_off()
            img = mpimg.imread(image_path)
            ax.imshow(img)