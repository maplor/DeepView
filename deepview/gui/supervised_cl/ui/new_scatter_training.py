import os

import torch
from PySide6.QtCore import QCoreApplication, QThread
from PySide6.QtWidgets import QMessageBox

from deepview.clustering_pytorch.datasets.factory import generate_dataloader
from deepview.clustering_pytorch.nnet.common_config import get_model
from deepview.gui.supervised_cl.train.utils import (
    adjust_learning_rate,
    evaluate,
    get_scl_criterion_opt,
    get_window_data_scl,
    load_model_parameters,
    train,
)
from deepview.gui.supervised_cl.ui.new_scatter_worker import Scl2ClWorker
from deepview.utils.auxiliaryfunctions import read_config, get_unsup_model_folder


class NewScatterTrainingMixin:
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