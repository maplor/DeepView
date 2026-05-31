import os

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QIcon

import deepview
from deepview.gui.tabs.train_network_worker import TrainWorker, transfer_sensor2columns


class TrainNetworkTrainingMixin:
    def start_training(self):
        # TODO: check if training is already in progress
        if hasattr(self, 'training_in_progress') and self.training_in_progress:
            print("Training is already in progress.")
            return

        config = self.root.config

        net_type = str(self.net_type.upper())
        learning_rate = float(self.save_iters_spin.text())
        batch_size = int(self.batchsize_spin.text())
        max_iter = int(self.display_iters_spin.text())

        data_length = int(self.display_datalen_spin.text())

        newSelectFilename = []
        for cb in self.display_dataset_cb_list:
            if cb.isChecked():
                newSelectFilename.append(cb.text())
        select_filenames = newSelectFilename

        newSelectColumn = []
        for i, cb in enumerate(self.display_column_cb_list):
            if cb.isChecked():
                newSelectColumn.append(cb.text())
        if len(newSelectColumn) == 0:
            self.show_message("please select sensor data to train.")
            return
        data_columns = transfer_sensor2columns(newSelectColumn, self.sensor_dict)

        self.train_thread = QtCore.QThread()
        self.train_worker = TrainWorker(config, net_type, self.sensor_dict, select_filenames, learning_rate, batch_size, max_iter, data_length, data_columns)
        self.train_worker.moveToThread(self.train_thread)

        self.train_worker.progress.connect(self.progress_bar.setValue)
        self.train_worker.finished.connect(self.on_training_finished)
        self.train_worker.stopped.connect(self.on_training_stopped)
        self.train_thread.started.connect(self.train_worker.train_network)
        self.train_worker.finished.connect(self.clean_up)

        self.ok_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.training_in_progress = True  # 设置标志
        self.train_thread.start()

    def clean_up(self):
        self.training_in_progress = False  # 重置标志
        self.train_thread.quit()
        self.train_thread.wait()
        self.train_worker.deleteLater()
        self.train_thread.deleteLater()
        self.ok_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def stop_training(self):
        self.train_worker.stop()

    def on_training_finished(self):
        self.stop_button.setEnabled(False)
        self.ok_button.setEnabled(True)
        self.show_message("The network is now trained and ready to use.")

    def on_training_stopped(self):
        self.stop_button.setEnabled(False)
        self.ok_button.setEnabled(True)
        self.clean_up()
        self.show_message("Training was stopped.")

    def train_network(self):
        self.progress_bar.setValue(0)

        config = self.root.config

        net_type = str(self.net_type.upper())
        learning_rate = float(self.save_iters_spin.text())
        batch_size = int(self.batchsize_spin.text())
        max_iter = int(self.display_iters_spin.text())

        data_length = int(self.display_datalen_spin.text())

        newSelectFilename = []
        for cb in self.display_dataset_cb_list:
            if cb.isChecked():
                newSelectFilename.append(cb.text())
        select_filenames = newSelectFilename

        newSelectColumn = []
        for i, cb in enumerate(self.display_column_cb_list):
            if cb.isChecked():
                newSelectColumn.append(cb.text())
        # data_columns = []
        # for sensor in newSelectColumn:
        #     # replace columns of GPS sensor
        #     if sensor.upper() == "GPS":
        #         data_columns.extend(['GPS_velocity', 'GPS_bearing'])
        #     else:
        #         data_columns.extend(self.sensor_dict[sensor])
        data_columns = transfer_sensor2columns(newSelectColumn, self.sensor_dict)

        deepview.train_network(
            self.sensor_dict,
            self.progress_update,
            config,
            select_filenames,
            net_type=net_type,
            lr=learning_rate,
            batch_size=batch_size,
            num_epochs=max_iter,
            data_len=data_length,
            data_column=data_columns
        )
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("The network is now trained and ready to use.")
        msg.setInformativeText(
            "Use the function 'Label with Interaction Plot' to visualize the data."
        )

        msg.setWindowTitle("Info")
        msg.setMinimumWidth(900)
        self.logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
        self.logo = self.logo_dir + "/assets/logo.png"
        msg.setWindowIcon(QIcon(self.logo))
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()