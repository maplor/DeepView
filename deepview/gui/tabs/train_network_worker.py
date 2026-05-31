from PySide6 import QtCore

import deepview


class TrainWorker(QtCore.QObject):
    progress = QtCore.Signal(int)
    finished = QtCore.Signal()
    stopped = QtCore.Signal()

    def __init__(self, config, net_type, sensor_dict, select_filenames, learning_rate, batch_size, max_iter, data_length, data_columns):
        super().__init__()
        self.config = config
        self.net_type = net_type
        self.sensor_dict = sensor_dict
        self.select_filenames = select_filenames
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.data_length = data_length
        self.data_columns = data_columns
        self._is_running = True

    def train_network(self):
        self.progress.emit(0)

        # Call the training function
        deepview.train_network(
            self.sensor_dict,
            self.progress,
            self.config,
            self.select_filenames,
            net_type=self.net_type,
            lr=self.learning_rate,
            batch_size=self.batch_size,
            num_epochs=self.max_iter,
            data_len=self.data_length,
            data_column=self.data_columns,
            stop_callback=self.check_running
        )
        # if not self._is_running:
        #     self.stopped.emit()
        #     return

        # 根据运行状态发出信号
        if self._is_running:
            self.finished.emit()
        else:
            self.stopped.emit()
        # self.finished.emit()

    def stop(self):
        self._is_running = False

    def check_running(self):
        return self._is_running


def get_sensor_columns(strings):
    '''
    其实放在config文件里更好，直接定义sensors，然后在代码中处理
    define a list of sensors, find sensors used in target data
    '''
    combined_strings = []
    current_combined = strings[0]

    for i in range(1, len(strings)):
        prefix_length = min(len(current_combined), len(strings[i]))
        prefix_length = next((k for k in range(prefix_length, 0, -1) if current_combined[:k] == strings[i][:k]), 0)

        if prefix_length > 0:
            current_combined = current_combined[:prefix_length]
        else:
            combined_strings.append(current_combined)
            current_combined = strings[i]

    combined_strings.append(current_combined)
    return combined_strings


def transfer_sensor2columns(newSelectColumn, sensor_dict):
    data_columns = []
    for sensor in newSelectColumn:
        # replace columns of GPS sensor
        if sensor.upper() == "GPS":
            data_columns.extend(['GPS_velocity', 'GPS_bearing'])
        else:
            data_columns.extend(sensor_dict[sensor])
    return data_columns