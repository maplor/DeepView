import os
from pathlib import Path
import pickle
import pandas as pd
import pyqtgraph as pg
import numpy as np
import argparse

import torch


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
from PySide6.QtCore import QCoreApplication, QThread, Signal, QObject
from deepview.gui.supervised_cl.train.utils import (
    get_window_data_scl,
    load_model_parameters,
    train,
    evaluate,
    get_scl_criterion_opt,
    adjust_learning_rate,
)
from deepview.clustering_pytorch.datasets.factory import (
    generate_dataloader
)
from deepview.clustering_pytorch.nnet.common_config import (
    get_model,
)
from deepview.utils.auxiliaryfunctions import (
    read_config,
    get_unsup_model_folder,
)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
p_setup = 'simclr'
if p_setup == 'simclr':
    AUGMENT = True  # 也存到yaml文件或者opt里
else:
    AUGMENT = False  # 是否用data augment，作为参数存储


class Scl2ClWorker(QObject):
    finished = Signal(tuple)  # Signal to indicate the task is finished
    progress = Signal(int)    # Signal to indicate progress
    stopped = Signal()

    def __init__(self, full_model_path, net_type, data, data_length, column_names, batch_size, aug1, aug2):
        super().__init__()
        self.full_model_path = full_model_path
        self.net_type = net_type
        self.data = data
        self.data_length = data_length
        self.column_names = column_names
        self.batch_size = batch_size
        self.aug1 = aug1
        self.aug2 = aug2
        self._is_running = True

    def run(self):

        selected_data, label_flag, timestamp, label = get_window_data_scl(
            self.data, self.column_names, self.data_length
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader = generate_dataloader(
            selected_data, label, timestamp, self.batch_size,
            True, device, label_flag, self.aug1, self.aug2
        )

        num_channel = selected_data.shape[-1]

        model = get_model(
            p_backbone=self.net_type,
            p_setup='simclr',
            num_channel=num_channel,
            data_len=self.data_length
        ).to(device)

        loaded_model = load_model_parameters(model, self.full_model_path, device)
        loaded_model, criterion, optimizer = get_scl_criterion_opt(loaded_model, device)

        method = 'SimCLR'
        nepochs = 5

        for epoch in range(1, nepochs + 1):
            if not self._is_running:
                self.stopped.emit()
                return
            adjust_learning_rate(optimizer, epoch, nepochs)
            loss = train(train_loader, model, method, criterion, optimizer, epoch, nepochs, device)
            print(f'SimCLR loss of the {epoch}-th training epoch is : {loss}')
            self.progress.emit(int((epoch / nepochs) * 100))
            QCoreApplication.processEvents()

        repre_tsne_SimCLR, flag_concat_SimCLR, label_concat_SimCLR = evaluate(model, train_loader, device)

        method = 'Supervised_SimCLR'
        for epoch in range(1, nepochs + 1):
            if not self._is_running:
                self.stopped.emit()
                return
            loss = train(train_loader, model, method, criterion, optimizer, epoch, nepochs, device)
            print(f'Supervised_SimCLR loss of the {epoch}-th training epoch is : {loss}')
            self.progress.emit(int((epoch / nepochs) * 100))
            QCoreApplication.processEvents()

        repre_tsne_CLR, _, _ = evaluate(model, train_loader, device)

        self.finished.emit((repre_tsne_SimCLR, flag_concat_SimCLR, label_concat_SimCLR, repre_tsne_CLR))

    def stop(self):
        self._is_running = False


class NewScatterMapWidget(QWidget):
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

    def display_data(self, data, model_name, data_length, column_names):
        aug1 = self.main_window.select_parameters_widget.augmentationComboBox_CLR.currentText()
        aug2 = self.main_window.select_parameters_widget.augmentationComboBox_SimCLR.currentText()

        # show existing labels
        existing_labels = self.main_window.select_parameters_widget.existing_labels_checkbox.isChecked()

        # show manual labels
        manual_labels = self.main_window.select_parameters_widget.manual_labels_checkbox.isChecked()

        batch_size = 1024
        full_model_path = self.get_unsup_model_path()
        self.Scl2ClThread = QThread()
        self.Scl2CLworker = Scl2ClWorker(full_model_path, model_name, data, data_length, column_names, batch_size, aug1, aug2)
        # Move worker to the thread
        self.Scl2CLworker.moveToThread(self.Scl2ClThread)
        # Connect signals and slots
        self.Scl2ClThread.started.connect(self.Scl2CLworker.run)
        self.Scl2CLworker.finished.connect(self.on_finished)
        self.Scl2CLworker.progress.connect(self.on_progress)
        self.Scl2CLworker.stopped.connect(self.on_stop)
        self.Scl2CLworker.finished.connect(self.clean_up)
        # self.Scl2CLworker.finished.connect(self.Scl2ClThread.quit)
        # self.Scl2CLworker.finished.connect(self.Scl2CLworker.deleteLater)
        # self.Scl2ClThread.finished.connect(self.Scl2ClThread.deleteLater)
        # Start the thread
        self.Scl2ClThread.start()

    def clean_up(self):
        if self.Scl2CLworker:
            self.Scl2CLworker.stop()
            self.Scl2ClThread.quit()
            self.Scl2ClThread.wait()
            self.Scl2CLworker.deleteLater()
            self.Scl2ClThread.deleteLater()
            


    def stop_thread(self):
        if self.Scl2CLworker:
            self.Scl2CLworker.stop()
        
        # TODO 需要绑定到stop按钮
        # Optionally disable stop button
        # self.stop_button.setEnabled(False)

    def on_finished(self, data):
        (repre_tsne_SimCLR, flag_concat_CLR,
         label_concat_CLR, repre_tsne_CLR) = data
        self.add_data_to_plot(repre_tsne_CLR, flag_concat_CLR, label_concat_CLR,
                              repre_tsne_SimCLR, flag_concat_CLR, label_concat_CLR)
        
    def on_stop(self):
        self.clean_up()
        print("Thread stopped")
        
    def on_progress(self, value):
        # print(f"Progress: {value}%")
        pass



    def update_existing_labels_status(self, status):
        self.existing_labels_status = status

    def update_manual_labels_status(self, status):
        self.manual_labels_status = status

    def get_unsup_model_path(self):
        config = self.main_window.root.config
        model_name = self.main_window.select_model_widget.modelComboBox.currentText()
        cfg = read_config(config)
        unsup_model_path = get_unsup_model_folder(cfg)
        full_model_path = os.path.join(cfg["project_path"], unsup_model_path, model_name)
        return full_model_path

    def generate_scl2cl_data(self, net_type, data, data_length, column_names, batch_size, aug1, aug2):
        # net_type=model_name: AE_CNN, 从文件名中读取
        # 1 calculate contrastive learning
        # 2 calculate supervised contrastive learning
        # 目前设定必须先cl再scl, 返回model

        full_model_path = self.get_unsup_model_path()

        # get parameters for data loading and training
        selected_data, label_flag, timestamp, label = get_window_data_scl(data,
                                                                          column_names,
                                                                          data_length)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader = generate_dataloader(selected_data, label, timestamp, batch_size,
                                           True, device, label_flag, aug1, aug2)

        # get model input channel
        num_channel = selected_data.shape[-1]

        # get model structure
        model = get_model(p_backbone=net_type,
                          p_setup='simclr',
                          num_channel=num_channel,
                          data_len=data_length)
        model = model.to(device)

        # load existing model parameters
        loaded_model = load_model_parameters(model, full_model_path, device)
        loaded_model, criterion, optimizer = get_scl_criterion_opt(loaded_model, device)

        

        # training routine
        '''
        opt.epochs参数从Parameter选框中读取
        当点击 Apply SCL按钮后运行下面程序
        '''
        method = 'SimCLR'
        nepochs = 5
        epoch = 0
        for epoch in range(1, nepochs + 1):
            # adjust_learning_rate(opt, optimizer, epoch)
            adjust_learning_rate(optimizer, epoch, nepochs)
            # loss = train(train_loader, model, criterion, optimizer, epoch, nepochs, opt)
            loss = train(train_loader, model, method, criterion, optimizer, epoch, nepochs, device)
            print('SimCLR loss of the ' + str(epoch) + '-th training epoch is :' + loss.__str__())
            QCoreApplication.processEvents()
        # evaluate and plot
        # train_loader, _ = set_loader(augment=AUGMENT, labeled_flag=False)
        '''
        第一张New scatter map生成方式
        '''
        # evaluate(model, epoch, nepochs, train_loader, fig_name=method)
        repre_tsne_SimCLR, flag_concat_SimCLR, label_concat_SimCLR = evaluate(model, train_loader, device)

        # supervised contrastive learning
        method = 'Supervised_SimCLR'
        for epoch in range(1, nepochs + 1):
            loss = train(train_loader, model, method, criterion, optimizer, epoch, nepochs, device)
            print('Supervised_SimCLR loss of the ' + str(epoch) + '-th training epoch is :' + loss.__str__())
            QCoreApplication.processEvents()
        # evaluate and plot
        '''
        第2张New scatter map生成方式
        '''
        # evaluate(model, epoch, train_loader, fig_name=method)
        repre_tsne_CLR, _, _ = evaluate(model, train_loader, device)


        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.method = method

        # # save the last model
        # ## 将新模型保存在旧的模型所在目录，后面加上opt.method标志
        # full_model_path_new = r'C:\Users\dell\Desktop\ss-cc-2024-08-05\unsup-models\iteration-0\ssAug5\AE_CNN_epoch29_datalen180_gps-acceleration_%s.pth' % method
        # state = {
        #     # 'opt': opt,
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'epoch': epoch,
        # }
        # torch.save(state, full_model_path_new)

        return repre_tsne_SimCLR, flag_concat_SimCLR, label_concat_SimCLR, repre_tsne_CLR
    

    def save_model(self):
        try:
            # 获取根对象配置
            config = self.main_window.root.config
            # 读取配置
            cfg = read_config(config)
            # 获取无监督模型文件夹路径
            unsup_model_path = get_unsup_model_folder(cfg)
            full_path = os.path.join(self.cfg["project_path"], unsup_model_path)
            model_name = 'AE_CNN_epoch29_datalen180_gps-acceleration_%s.pth' % self.method
            full_model_path_new = os.path.join(full_path, model_name)

            # save the last model
            ## 将新模型保存在旧的模型所在目录，后面加上opt.method标志
            # full_model_path_new = r'C:\Users\dell\Desktop\ss-cc-2024-08-05\unsup-models\iteration-0\ssAug5\AE_CNN_epoch29_datalen180_gps-acceleration_%s.pth' % method
            state = {
                # 'opt': opt,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': self.epoch,
            }
            torch.save(state, full_model_path_new)
        except Exception as e:
            # print(e)
            pass




    def generate_test_data(self, data, model_name, data_length, column_names):
        # 首先生成模型训representation，再生成tsne结果
        # num_points = 100
        # repre_tsne_CLR = np.random.rand(num_points, 2)
        # flag_concat_CLR = np.random.randint(0, 2, num_points)  # 生成一维数组
        # label_concat_CLR = np.random.randint(0, 4, num_points)  # 生成一维数组
        #
        # repre_tsne_SimCLR = np.random.rand(num_points, 2)
        # flag_concat_SimCLR = np.random.randint(0, 2, num_points)
        # label_concat_SimCLR = np.random.randint(0, 4, num_points)
        # aug1, aug2 = 't_warp', 't_warp'
        aug1 = self.main_window.select_parameters_widget.augmentationComboBox_CLR.currentText()
        aug2 = self.main_window.select_parameters_widget.augmentationComboBox_SimCLR.currentText()
        # batch_size = 1024
        batch_size = self.main_window.select_parameters_widget.batch_size

        (repre_tsne_SimCLR, flag_concat_CLR,
         label_concat_CLR, repre_tsne_CLR) = self.generate_scl2cl_data(
                                                                 model_name,
                                                                 data,
                                                                 data_length,
                                                                 column_names,
                                                                 batch_size,
                                                                 aug1,
                                                                 aug2)

        # 两张图同时化
        self.add_data_to_plot(repre_tsne_CLR, flag_concat_CLR, label_concat_CLR,
                              repre_tsne_SimCLR, flag_concat_CLR, label_concat_CLR)
        

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