import pyqtgraph as pg
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox,
    QCheckBox,
    QDialog,
    QRadioButton,
    QSplitter,
    QFrame,
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QComboBox, QPushButton, QSpacerItem, QSizePolicy, QLineEdit,
    QMessageBox, QSpinBox
)
from deepview.gui.supervised_cl.ui.new_scatter_training import NewScatterTrainingMixin


class NewScatterMapWidget(NewScatterTrainingMixin, QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window  # 保存对主界面的引用

        self.repre_tsne_CLR, self.flag_concat_CLR, self.label_concat_CLR = None, None, None
        self.repre_tsne_SimCLR, self.flag_concat_SimCLR, self.label_concat_SimCLR = None, None, None

        # Create a vertical layout
        self.layout = QHBoxLayout(self)

        self.map_1 = pg.PlotWidget()
        self.map_2 = pg.PlotWidget()
        self.existing_labels_status = True
        self.manual_labels_status = True


        # TODO 需要在主界面中添加按钮，绑定到display data按钮
        # self.stop_button = QPushButton("Stop")
        # self.stop_button.setEnabled(False)
        # self.stop_button.clicked.connect(self.stop_thread)


        self.layout.addWidget(self.map_1)
        self.layout.addWidget(self.map_2)


    # def display_data(self, data, model_name, data_length, column_names):
    #     self.generate_test_data(data, model_name, data_length, column_names)

    def update_existing_labels_status(self, status):
        self.existing_labels_status = status

    def update_manual_labels_status(self, status):
        self.manual_labels_status = status

    def add_data_to_plot(self, repre_tsne_CLR, flag_concat_CLR, label_concat_CLR,
                         repre_tsne_SimCLR, flag_concat_SimCLR, label_concat_SimCLR):

        # 重新初始化两张图
        self.map_1.clear()
        self.map_2.clear()

        '''
        当点击 Apply SCL按钮后运行下面程序
        '''
        '''
        第一张New scatter map生成方式
        '''
       
        real_label_idxs_unlabeled_1 = np.where(flag_concat_CLR == 0)[0]
        x_unlabeled_1 = repre_tsne_CLR[real_label_idxs_unlabeled_1, 0]
        y_unlabeled_1 = repre_tsne_CLR[real_label_idxs_unlabeled_1, 1]
        self.scatter1 = pg.ScatterPlotItem(x=x_unlabeled_1, y=y_unlabeled_1, pen=pg.mkPen(None),
                                    brush=pg.mkBrush(128, 128, 128, 128), size=5,
                                    data=real_label_idxs_unlabeled_1)

        self.map_1.addItem(self.scatter1)

        real_label_idxs_labeled_1 = np.where(flag_concat_CLR == 1)[0]
        x_labeled_1 = repre_tsne_CLR[real_label_idxs_labeled_1, 0]
        y_labeled_1 = repre_tsne_CLR[real_label_idxs_labeled_1, 1]
        # plt.scatter(x, y, color='blue', alpha=0.5, label='labeled')
        real_label_concat = label_concat_CLR[real_label_idxs_labeled_1]
        color_dict = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow'}
        brushes = [pg.mkBrush(color_dict[label]) for label in real_label_concat]
        self.scatter2 = pg.ScatterPlotItem(x=x_labeled_1, y=y_labeled_1, pen=pg.mkPen(None),
                                    brush=brushes, size=5, data=real_label_idxs_labeled_1)
        self.map_1.addItem(self.scatter2)


        '''
        第2张New scatter map生成方式
        '''
        # self.map_2 = self.map_2.addPlot(title=f"Contrast,{epoch}")


        real_label_idxs_unlabeled_2 = np.where(flag_concat_SimCLR == 0)[0]
        x_unlabeled_2 = repre_tsne_SimCLR[real_label_idxs_unlabeled_2, 0]
        y_unlabeled_2 = repre_tsne_SimCLR[real_label_idxs_unlabeled_2, 1]
        self.scatter3 = pg.ScatterPlotItem(x=x_unlabeled_2, y=y_unlabeled_2, pen=pg.mkPen(None),
                                    brush=pg.mkBrush(128, 128, 128, 128), size=5,
                                    data=real_label_idxs_unlabeled_2)

        self.map_2.addItem(self.scatter3)

        real_label_idxs_labeled_2 = np.where(flag_concat_SimCLR == 1)[0]
        x_labeled_2 = repre_tsne_SimCLR[real_label_idxs_labeled_2, 0]
        y_labeled_2 = repre_tsne_SimCLR[real_label_idxs_labeled_2, 1]
    
        real_label_concat = label_concat_SimCLR[real_label_idxs_labeled_2]
        color_dict = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow'}
        brushes2 = [pg.mkBrush(color_dict[label]) for label in real_label_concat]
        self.scatter4 = pg.ScatterPlotItem(x=x_labeled_2, y=y_labeled_2, pen=pg.mkPen(None),
                                    brush=brushes2, size=5, data=real_label_idxs_labeled_2)
        self.map_2.addItem(self.scatter4)

        # Connect signals
        self.scatter1.sigClicked.connect(lambda plot, points: self.on_click(self.scatter1, points))
        self.scatter2.sigClicked.connect(lambda plot, points: self.on_click(self.scatter2, points))
        self.scatter3.sigClicked.connect(lambda plot, points: self.on_click(self.scatter3, points))
        self.scatter4.sigClicked.connect(lambda plot, points: self.on_click(self.scatter4, points))




    def change_points_properties(self, indices, new_size, new_color):
        # Restore original properties of last modified points
        for p, original_size, original_brush in self.main_window.last_modified_points:
            p.setSize(original_size)
            p.setBrush(original_brush)

        self.main_window.last_modified_points = []  # Clear the list

        # 如果old scatter map有点被点击，new scatter map中的点也会被点击，如果old scatter map未就绪，new scatter map中的点不会被点击
        # Change properties of new points
        
        # if self.main_window.old_scatter_map_widget.scatter1 is not None:
        # 检查 new_scatter_map_widget 是否已初始化
        if hasattr(self.main_window, 'new_scatter_map_widget') and \
           hasattr(self.main_window.new_scatter_map_widget, 'scatter1') and \
           self.main_window.new_scatter_map_widget.scatter1 is not None:
            for scatter in [self.scatter1, self.scatter2, self.scatter3, self.scatter4, self.main_window.old_scatter_map_widget.scatter1, self.main_window.old_scatter_map_widget.scatter2]:
                points = scatter.points()
                for p in points:
                    if p.data() in indices:
                        # Save current properties
                        original_size = p.size()
                        original_brush = p.brush()
                        self.main_window.last_modified_points.append((p, original_size, original_brush))

                        # Change to new properties
                        p.setSize(new_size)
                        p.setBrush(pg.mkBrush(new_color))
        else:
            # 如果控件未初始化，显示消息框
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText("All scatter plots have not yet been initialized.")
            msg_box.setInformativeText("Please initialize the scatter plots before continuing.")
            msg_box.setWindowTitle("Plotting error.")
            msg_box.exec_()
            # for scatter in [self.scatter1, self.scatter2, self.scatter3, self.scatter4]:
            #     points = scatter.points()
            #     for p in points:
            #         if p.data() in indices:
            #             # Save current properties
            #             original_size = p.size()
            #             original_brush = p.brush()
            #             self.main_window.last_modified_points.append((p, original_size, original_brush))

            #             # Change to new properties
            #             p.setSize(new_size)
            #             p.setBrush(pg.mkBrush(new_color))

    def on_click(self, scatter, points):
        idxs = []  # Initialize idxs list
        for p in points:
            idx = p.data()
            idxs.append(idx)  # Add idx to idxs list
        self.change_points_properties(idxs, new_size=14, new_color=(255, 0, 0, 255))