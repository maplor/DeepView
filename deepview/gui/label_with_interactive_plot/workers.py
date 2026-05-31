import json
import logging
import os

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from deepview.gui.label_with_interactive_plot.utils import (
    featureExtraction,
    find_data_columns,
)


logger = logging.getLogger(__name__)


# 定义一个QObject来保存各种后台线程信号
class TaskSignals(QObject):
    # 后台保存csv文件完成信号，SaveCsvTask
    save_csv_finished = Signal(str)


# 后台保存csv文件类
class SaveCsvTask(QRunnable):
    def __init__(self, area_data, data, cfg, combo_box_text, is_timer):
        super().__init__()
        self.signals = TaskSignals()
        self.area_data = area_data
        self.data = data
        self.cfg = cfg
        self.combo_box_text = combo_box_text
        self.is_timer = is_timer

    @Slot()
    def run(self):
        # Parse the area data
        try:
            area_data = json.loads(self.area_data)
        except json.JSONDecodeError:
            logger.exception("Failed to decode JSON")
            return

        for reg in area_data:
            name = reg[0].get("name")
            first_timestamp = reg[0].get("timestamp", {}).get("start")
            second_timestamp = reg[0].get("timestamp", {}).get("end")
            self.data.loc[
                (self.data['unixtime'] >= int(first_timestamp)) &
                (self.data['unixtime'] <= int(second_timestamp)), 'label'] = name

        # Handle saving the data
        os.makedirs(os.path.join(self.cfg["project_path"], "edit-data"), exist_ok=True)
        edit_data_path = os.path.join(self.cfg["project_path"], "edit-data", self.combo_box_text)

        try:
            if os.path.exists(edit_data_path):
                for num in range(1, 100):
                    firstname = edit_data_path.split('Hz')[0]
                    new_path = firstname + '_' + str(num) + '.pkl'
                    if not os.path.exists(new_path):
                        self.data.to_csv(new_path)
                        break
            else:
                new_path = edit_data_path
                self.data.to_csv(edit_data_path)
        except Exception:
            logger.exception("Save data error")
        else:
            logger.info("File saved at %s", new_path)
            if self.is_timer == 0:
                self.signals.save_csv_finished.emit(new_path)


# 后台handleComputeData类
class HandleComputeWorker(QObject):
    dataChangedSignal = Signal(pd.DataFrame)
    finished = Signal(object)
    stopped = Signal()

    def __init__(self, root, data, RawDataName, cfg, dataChanged, sensor_dict, column_names, data_length, model_path, model_name):
        super().__init__()
        self.root = root
        self.data = data
        self.RawDataName = RawDataName
        self.cfg = cfg
        self.dataChanged = dataChanged
        self.sensor_dict = sensor_dict
        self.column_names = column_names
        self.model_path = model_path
        self.data_length = data_length
        self.model_name = model_name
        self._is_running = True

    @Slot()
    def run(self):

        # 获取combobox的内容
        # self.data, self.dataChanged = get_data_from_pkl(self.RawDataName, self.cfg, self.dataChanged)
        # self.dataChangedSignal.emit(self.data)

        new_column_names = find_data_columns(self.sensor_dict, self.column_names)

        self.data['datetime'] = pd.to_datetime(self.data['datetime'])

        # 将数据切割成片段以获取潜在特征和索引
        start_indice, end_indice, pos = featureExtraction(self.root,
                                                          self.data,
                                                          self.data_length,
                                                          new_column_names,
                                                          self.model_path,
                                                          self.model_name)

        # 保存数据到scatterItem的属性中
        n = len(start_indice)
        spots = [{'pos': pos[i, :],
                  'data': (i, start_indice[i], end_indice[i]),
                  'brush': self.checkColor(self.data.loc[i * self.data_length, 'label'], first=True)}
                 for i in range(n)]

        if not self._is_running:
            self.stopped.emit()
            return

        # Emit the result
        self.finished.emit((spots, start_indice, end_indice))

    def stop(self):
        self._is_running = False

    def checkColor(self, label, first=False):
        if first:
            # 如果是第一次调用，返回默认的白色笔刷
            # return pg.mkBrush(255, 255, 255, 120)
            # 改成灰色
            return pg.mkBrush(72, 72, 96, 120)

        if label not in list(self.label_dict.keys()):
            # # 如果标签不在标签字典中，返回默认的白色笔刷
            # return pg.mkBrush(255, 255, 255, 120)
            # 改成灰色
            return pg.mkBrush(72, 72, 96, 120)

        # 定义一组颜色
        list_color = [pg.mkBrush(0, 0, 255, 120),
                      pg.mkBrush(255, 0, 0, 120),
                      pg.mkBrush(0, 255, 0, 120),
                      pg.mkBrush(255, 255, 255, 120),
                      pg.mkBrush(255, 0, 255, 120),
                      pg.mkBrush(0, 255, 255, 120),
                      pg.mkBrush(255, 255, 0, 120),
                      pg.mkBrush(5, 5, 5, 120)]
        count = 0
        for lstr, _ in self.label_dict.items():
            if label == lstr:
                # 根据标签返回相应的颜色
                return list_color[count % len(list_color)]
            count += 1


# 创建一个函数来找到最近的有效索引
def find_nearest_index(target_index, valid_indices):
    if len(valid_indices) == 0:
        return None
    nearest_index = valid_indices[np.abs(valid_indices - target_index).argmin()]
    return nearest_index