import os

import pandas as pd
import torch

from deepview.gui.label_with_interactive_plot.utils import get_data_from_pkl
from deepview.gui.tabs.train_network_worker import transfer_sensor2columns
from deepview.utils.auxiliaryfunctions import (
    get_param_from_path,
    get_unsup_model_folder,
    get_unsupervised_set_folder,
    read_config,
)


class SupervisedClDataMixin:
    def save_model(self):
        try:
            # 获取根对象配置
            config = self.root.config
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
                'model': self.new_scatter_map_widget.model.state_dict(),
                'optimizer': self.new_scatter_map_widget.optimizer.state_dict(),
                'epoch': self.new_scatter_map_widget.epoch,
            }
            torch.save(state, full_model_path_new)
        except Exception as e:
            # print(e)
            pass

    def display_old_scatter_data(self):
        # read sensor data
        # 特征提取：找到数据帧中的列名
        model_filename = self.select_model_widget.modelComboBox.currentText()
        # preprocessing: find column names in dataframe
        model_name, data_length, column_names = \
            get_param_from_path(model_filename)  # 从路径获取模型参数
        # data, _ = get_data_from_pkl(self.select_model_widget.RawDatacomboBox.currentText(),
        #                             self.cfg)

        selected_items = [checkbox.text() for checkbox in self.select_model_widget.display_dataset_cb_list if checkbox.isChecked()]

        all_data = pd.DataFrame()
        for item in selected_items:
            data, _ = get_data_from_pkl(item,
                                        self.cfg)

            # data是pd.DataFrame，将所有data合并到一个DataFrame中
            # all_data = pd.concat([all_data, data])
            all_data = pd.concat([all_data, data], ignore_index=True)

        # transfer sensor name to columns
        data_columns = transfer_sensor2columns(column_names, self.sensor_dict)
        self.old_scatter_map_widget.display_data(all_data, model_filename, data_length, data_columns, model_name)

    def display_new_scatter_data(self):
        # read sensor data
        # 特征提取：找到数据帧中的列名
        model_filename = self.select_model_widget.modelComboBox.currentText()
        # preprocessing: find column names in dataframe
        model_name, data_length, column_names = \
            get_param_from_path(model_filename)  # 从路径获取模型参数

        # data, _ = get_data_from_pkl(self.select_model_widget.RawDatacomboBox.currentText(),
        #                             self.cfg)
        selected_items = [checkbox.text() for checkbox in self.select_model_widget.display_dataset_cb_list if checkbox.isChecked()]
        all_data = pd.DataFrame()
        for item in selected_items:
            data, _ = get_data_from_pkl(item,
                                        self.cfg)

            # data是pd.DataFrame，将所有data合并到一个DataFrame中
            all_data = pd.concat([all_data, data])

        # transfer sensor name to columns
        data_columns = transfer_sensor2columns(column_names, self.sensor_dict)

        self.new_scatter_map_widget.display_data(all_data, model_filename, data_length, data_columns)

    # 从.pkl文件获取数据的方法
    def get_data_from_pkl(self, filename):
        self.data_name = filename
        # 获取无监督数据集文件夹路径
        unsup_data_path = get_unsupervised_set_folder()
        # 构建文件路径
        self.data_path = os.path.join(self.cfg["project_path"], unsup_data_path, filename)
        return

    def get_model_param_from_path(self, model_path):
        # 从路径获取模型参数的方法
        if model_path:
            model_name, data_length, column_names = get_param_from_path(model_path)
            # 保存到主窗口的属性
            self.model_path = model_path
            self.model_name = model_name
            self.data_length = data_length
            self.column_names = column_names