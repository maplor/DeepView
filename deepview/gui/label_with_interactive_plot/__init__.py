# 导入数学模块
# 从typing模块导入List类型
# 导入torch模块
import datetime
import json
import logging
import os
import pickle
from functools import partial
from pathlib import Path
from ruamel.yaml import YAML
from PySide6 import QtGui
import cv2
import time

import matplotlib
import numpy as np
import pandas as pd
import pyqtgraph as pg
import torch
from PySide6.QtCore import (
    QObject, Signal, Slot, QTime, QTimer, Qt
)
# 从PySide6.QtCore导入QTimer, QRectF, Qt
from PySide6.QtCore import QRectF
from PySide6.QtCore import QRunnable, QThreadPool, Slot, QThread, QObject, Signal, QFileSystemWatcher
# 从PySide6.QtWidgets导入多个类
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QRadioButton,
    QSplitter,
    QFrame,
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QComboBox, QPushButton, QSpacerItem, QSizePolicy, QLineEdit,
    QMessageBox, QDoubleSpinBox, QFileDialog, QCalendarWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent, QStandardItemModel, QStandardItem, QColor, QPainter
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QHBoxLayout, QPushButton, QMessageBox, QInputDialog

from PySide6.QtGui import QImage, QPixmap
from datetime import datetime, timedelta
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QCalendarWidget, QLabel
from PySide6.QtCore import QTime, Qt, QRectF
from PySide6.QtGui import QMouseEvent, QPainter, QColor
import sqlite3
from PySide6.QtCore import QDate
from PySide6.QtGui import QTextCharFormat
from PySide6.QtWidgets import QTextEdit, QTimeEdit, QPushButton


# 从deepview.utils.auxiliaryfunctions导入多个函数
from deepview.utils.auxiliaryfunctions import (
    read_config,
    get_param_from_path,
    get_unsupervised_set_folder,
    get_raw_data_folder,
    get_unsup_model_folder,
    grab_files_in_folder_deep,
    get_db_folder
)

from deepview.gui.label_with_interactive_plot.utils import (
    get_data_from_pkl,
    featureExtraction,
    find_data_columns,
    generate_filename
)

from deepview.gui.label_with_interactive_plot.styles import combobox_style_light, combobox_style_dark


# 创建一个蓝色的pg.mkPen对象，宽度为2
clickedPen = pg.mkPen('b', width=2)



class ClickableLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()





class TimeSelectorWidget(QLabel):
    def __init__(self, begin_time_edit, end_time_edit, video_time_list):
        super().__init__()
        self.setText("Select a time range")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(200, 100)  # Adjust height to accommodate two rows
        self.start_time_edit = begin_time_edit
        self.end_time_edit = end_time_edit
        self.video_time_list = video_time_list
        self.hourly_data = {}
        self.start_time = None
        self.end_time = None
        self.setStyleSheet("background-color: lightgray;")
        self.selected_rects = []

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(Qt.black)
        width = self.size().width()
        height = self.size().height()

        # Draw vertical lines
        for i in range(1, 12):
            x = i * (width / 12)
            painter.drawLine(x, 0, x, height / 2)
            painter.drawLine(x, height / 2, x, height)

        # Draw middle line
        painter.drawLine(0, height / 2, width, height / 2)

        # Draw hourly data with different colors
        total_minutes = 24 * 60
        for hour in range(24):
            if hour in self.hourly_data:
                start_minute = hour * 60
                end_minute = start_minute + 60

                # Determine presence of data types
                acc_present = self.hourly_data[hour]['acc']
                gyro_present = self.hourly_data[hour]['gyro']
                mag_present = self.hourly_data[hour]['mag']

                for minute in range(start_minute, end_minute):
                    x = (minute % (total_minutes / 2)) / (total_minutes / 2) * width
                    y = 0 if minute < (total_minutes / 2) else height / 2
                    rect = QRectF(x, y, width / (total_minutes / 2), height / 2)

                    # Draw rectangles with different colors
                    if acc_present:
                        painter.setBrush(QColor(255, 255, 0, 128))  # Semi-transparent yellow
                        painter.setPen(Qt.NoPen)
                        acc_rect = QRectF(rect.x(), rect.y(), rect.width(), rect.height() * 0.3)
                        painter.drawRect(acc_rect)
                    if gyro_present:
                        painter.setBrush(QColor(255, 0, 0, 128))  # Semi-transparent red
                        painter.setPen(Qt.NoPen)
                        gyro_rect = QRectF(rect.x(), rect.y() + rect.height() * 0.35, rect.width(), rect.height() * 0.3)
                        painter.drawRect(gyro_rect)
                    if mag_present:
                        painter.setBrush(QColor(0, 0, 255, 128))  # Semi-transparent blue
                        painter.setPen(Qt.NoPen)
                        mag_rect = QRectF(rect.x(), rect.y() + rect.height() * 0.7, rect.width(), rect.height() * 0.3)
                        painter.drawRect(mag_rect)

        # Draw predefined time segments
        segments = self.parse_time_segments()
        painter.setBrush(QColor(0, 255, 0, 128))  # Semi-transparent green
        painter.setPen(Qt.NoPen)
        for start, end in segments:
            start_minute = start.hour() * 60 + start.minute()
            end_minute = end.hour() * 60 + end.minute()
            for minute in range(start_minute, end_minute):
                x = (minute % (1440 / 2)) / (1440 / 2) * width
                # 计算矩形的 y 坐标和高度
                rect_height = height * 0.7 / 2
                y_offset = (height / 2 - rect_height) / 2
                y = y_offset if minute < (1440 / 2) else height / 2 + y_offset
                rect = QRectF(x, y, width / (1440 / 2), rect_height)
                painter.drawRect(rect)

        # Draw selected rectangles in semi-transparent blue
        painter.setBrush(QColor(0, 0, 255, 128))
        painter.setPen(Qt.NoPen)
        for rect in self.selected_rects:
            painter.drawRect(rect)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_time = self.map_time(event.position())
            # self.setText(f"Start: {self.start_time.toString()}")

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.start_time:
            self.end_time = self.map_time(event.position())
            self.start_time_edit.setTime(self.start_time)
            self.end_time_edit.setTime(self.end_time)

            self.update_selected_rects()
            self.update()

    def map_time(self, pos):
        width = self.size().width()
        height = self.size().height()
        total_minutes = 24 * 60
        if pos.y() < height / 2:
            minute = (pos.x() / width) * (total_minutes / 2)
        else:
            minute = (pos.x() / width) * (total_minutes / 2) + (total_minutes / 2)
        return QTime(int(minute // 60), int(minute % 60))

    def update_selected_rects(self):
        width = self.size().width()
        height = self.size().height()
        total_minutes = 24 * 60
        start_minute = self.start_time.hour() * 60 + self.start_time.minute()
        end_minute = self.end_time.hour() * 60 + self.end_time.minute()
        self.selected_rects.clear()

        for minute in range(start_minute, end_minute + 1):
            x = (minute % (total_minutes / 2)) / (total_minutes / 2) * width
            y = 0 if minute < (total_minutes / 2) else height / 2
            rect = QRectF(x, y, width / (total_minutes / 2), height / 2)
            self.selected_rects.append(rect)

    def reset_green_blocks(self, video_time_list, hourly_data):
        self.video_time_list = video_time_list #[('2024-05-28 06:20:29', '2024-05-28 06:21:29'), ('2024-05-28 06:27:33', '2024-05-28 06:28:33'), ('2024-05-28 07:41:39', '2024-05-28 07:42:39'), ('2024-05-28 07:55:24', '2024-05-28 07:56:24')]
        self.hourly_data = hourly_data
        # print(self.hourly_data)
        # 重新设置预定义的时间段
        self.update()

    def parse_time_segments(self):
        segments = []
        for start_str, end_str in self.video_time_list: # PySide6.QtCore.QTime(7, 55, 24, 0) PySide6.QtCore.QTime(7, 56, 24, 0)
            start_str = start_str.split()[1] 
            end_str = end_str.split()[1]
            start_time = QTime.fromString(start_str, "HH:mm:ss")
            end_time = QTime.fromString(end_str, "HH:mm:ss")
            # print(start_time, end_time)
            if start_time.isValid() and end_time.isValid():
                segments.append((start_time, end_time))
        return segments


class DateTimeSelector(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("Date and Time Selector")
        self.setGeometry(100, 100, 400, 300)

        # conn = sqlite3.connect('database.db')
        conn = sqlite3.connect(self.main_window.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT DATE(datetime) FROM raw_data")
        dates = cursor.fetchall()
        conn.close()

        date_list = [QDate.fromString(date[0], 'yyyy-MM-dd') for date in dates]

        self.video_time_list = []

        layout = QVBoxLayout(self)

        self.calendar = QCalendarWidget(self)
        self.calendar.selectionChanged.connect(self.date_changed)

        format = QTextCharFormat()
        format.setBackground(QColor('yellow'))

        for single_date in date_list:
            self.calendar.setDateTextFormat(single_date, format)

        if date_list:
            self.center_calendar_on_dates(date_list)

        self.date_label = QLabel("Selected Date: None", self)

        self.text_edit = QTextEdit(self)
        self.text_edit.setPlaceholderText("Enter text here...")

        # self.text_edit.append("04:37:27-05:37:34 others")
        # self.text_edit.append("05:37:34-06:37:35 others")
        # self.text_edit.append("06:37:35-07:37:36 others")
        # self.text_edit.append("08:37:36-15:37:37 others")

        self.begin_label = QLabel("Begin Time:", self)
        self.begin_input = QTimeEdit(self)
        self.begin_input.setDisplayFormat("HH:mm:ss")
        self.begin_input.setTime(QTime.currentTime())

        self.end_label = QLabel("End Time:", self)
        self.end_input = QTimeEdit(self)
        self.end_input.setDisplayFormat("HH:mm:ss")
        self.end_input.setTime(QTime.currentTime())

        self.ok_button = QPushButton("OK", self)

        self.time_selector = TimeSelectorWidget(self.begin_input, self.end_input, self.video_time_list)

        layout.addWidget(self.calendar)
        layout.addWidget(self.date_label)
        layout.addWidget(self.time_selector)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.begin_label)
        layout.addWidget(self.begin_input)
        layout.addWidget(self.end_label)
        layout.addWidget(self.end_input)
        layout.addWidget(self.ok_button)
        self.ok_button.clicked.connect(self.ok_button_clicked)

    def ok_button_clicked(self):
        # 使用选择的日期和时间来查询数据库
        start_time = self.begin_input.time()
        end_time = self.end_input.time()
        date = self.calendar.selectedDate().toString('yyyy-MM-dd')
        start_datetime = datetime.strptime(f"{date} {start_time.toString()}", "%Y-%m-%d %H:%M:%S")
        end_datetime = datetime.strptime(f"{date} {end_time.toString()}", "%Y-%m-%d %H:%M:%S")

        # conn = sqlite3.connect('database.db')
        conn = sqlite3.connect(self.main_window.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT *
        FROM raw_data
        WHERE datetime >= ? AND datetime <= ?
        ''', (start_datetime, end_datetime))

        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
        df['index'] = df.index

        self.main_window.handel_calendar_data(df)



    def date_changed(self):

        date = self.calendar.selectedDate()
        # conn = sqlite3.connect('database.db')
        conn = sqlite3.connect(self.main_window.db_path)
        cursor = conn.cursor()
        start_of_day = datetime.strptime(date.toString('yyyy-MM-dd'), '%Y-%m-%d')
        end_of_day = start_of_day + timedelta(days=1)

        # print(start_of_day) # 2024-05-28 00:00:00
        # print(end_of_day) # 2024-05-29 00:00:00

        # 查询特定日期的数据
        cursor.execute('''
        SELECT video_stt, video_stp FROM videos
        WHERE video_stt >= ? AND video_stt < ?
        ''', (start_of_day, end_of_day))

        # 获取查询结果
        results = cursor.fetchall()
        # print(results)

        # 将结果转换为包含开始和结束时间的列表
        self.video_time_list = [(row[0], row[1]) for row in results]
        # 创建一个字典来存储每小时的标记
        hourly_data = {hour: {'acc': False, 'gyro': False, 'mag': False} for hour in range(24)}
        date_str = date.toString('yyyy-MM-dd')
        # 查询特定日期的数据
        cursor.execute('''
            SELECT datetime, acc_x, gyro_x, mag_x
            FROM raw_data 
            WHERE date(datetime) = ?
        ''', (date_str,))

        rows = cursor.fetchall()

        for row in rows:
            dt, acc_x, gyro_x, mag_x = row
            hour = datetime.fromisoformat(dt).hour

            # 标记对应的传感器数据存在
            if acc_x is not None:
                hourly_data[hour]['acc'] = True
            if gyro_x is not None:
                hourly_data[hour]['gyro'] = True
            if mag_x is not None:
                hourly_data[hour]['mag'] = True




        self.time_selector.reset_green_blocks(self.video_time_list, hourly_data)
        # 查询指定日期的labels表数据，数据有开始结束时间，标签名(stt_timestamp TEXT, stp_timestamp TEXT,label_name TEXT,)
        cursor.execute('''
        SELECT stt_timestamp, stp_timestamp, label_name FROM labels
        WHERE stt_timestamp >= ? AND stt_timestamp < ?
        ''', (start_of_day, end_of_day))
        results = cursor.fetchall()

        # 将结果转换为包含开始和结束时间的列表
        time_list = [(row[0], row[1], row[2]) for row in results]

        # 将结果添加到text_edit里
        self.text_edit.clear()
        # print(time_list)
        for start_time, end_time, label in time_list:
            start_time_str = start_time.split()[1]  # Extract time part
            end_time_str = end_time.split()[1]  # Extract time part
            self.text_edit.append(f"{start_time_str}-{end_time_str} {label}")

        # 关闭连接
        conn.close()
        self.date_label.setText(f"Selected Date: {date.toString()}")

    def center_calendar_on_dates(self, dates):
        min_date = min(dates)
        max_date = max(dates)

        mid_year = (min_date.year() + max_date.year()) // 2
        mid_month = (min_date.month() + max_date.month()) // 2

        self.calendar.setCurrentPage(mid_year, mid_month)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = DateTimeSelector()
#     window.show()
#     sys.exit(app.exec())

# # TODO 时间选择窗口
# class TimeSelectorWidget(QLabel):
#     def __init__(self):
#         super().__init__()
#         self.setText("Select a time range")
#         self.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.setFixedSize(200, 100)  # Adjust height to accommodate two rows
#         self.start_time = None
#         self.end_time = None
#         self.setStyleSheet("background-color: lightgray;")  # Set background color
#         self.selected_rects = []

#     def paintEvent(self, event):
#         super().paintEvent(event)
#         painter = QPainter(self)
#         painter.setPen(Qt.black)
#         width = self.size().width()
#         height = self.size().height()
        
#         # Draw vertical lines
#         for i in range(1, 12):
#             x = i * (width / 12)
#             painter.drawLine(x, 0, x, height / 2)
#             painter.drawLine(x, height / 2, x, height)
        
#         # Draw middle line
#         painter.drawLine(0, height / 2, width, height / 2)

#         # Draw selected rectangles in semi-transparent blue
#         painter.setBrush(QColor(0, 0, 255, 128))  # RGBA for semi-transparent blue
#         painter.setPen(Qt.NoPen)  # No outline for rectangles
#         for rect in self.selected_rects:
#             painter.drawRect(rect)

#     def mousePressEvent(self, event: QMouseEvent):
#         if event.button() == Qt.MouseButton.LeftButton:
#             self.start_time = self.map_time(event.pos())
#             self.setText(f"Start: {self.start_time.toString()}")

#     def mouseReleaseEvent(self, event: QMouseEvent):
#         if event.button() == Qt.MouseButton.LeftButton and self.start_time:
#             self.end_time = self.map_time(event.pos())
#             self.setText(f"From {self.start_time.toString()} to {self.end_time.toString()}")
#             self.update_selected_rects()
#             self.update()

#     def map_time(self, pos):
#         width = self.size().width()
#         height = self.size().height()
#         hours = 24
#         total_minutes = hours * 60
#         if pos.y() < height / 2:
#             minute = (pos.x() / width) * (total_minutes / 2)
#         else:
#             minute = (pos.x() / width) * (total_minutes / 2) + (total_minutes / 2)
#         return QTime(int(minute // 60), int(minute % 60))

#     def update_selected_rects(self):
#         width = self.size().width()
#         height = self.size().height()
#         hours = 24
#         total_minutes = hours * 60
#         start_minute = self.start_time.hour() * 60 + self.start_time.minute()
#         end_minute = self.end_time.hour() * 60 + self.end_time.minute()
#         self.selected_rects.clear()

#         for minute in range(start_minute, end_minute + 1):
#             x = (minute % (total_minutes / 2)) / (total_minutes / 2) * width
#             y = 0 if minute < (total_minutes / 2) else height / 2
#             rect = QRectF(x, y, width / (total_minutes / 2), height / 2)
#             self.selected_rects.append(rect)


# class DateTimeSelector(QWidget):
#     def __init__(self, main_widget):
#         super().__init__()

#         self.setWindowTitle("Date and Time Selector")
#         self.setGeometry(100, 100, 400, 300)

#         layout = QVBoxLayout(self)

#         self.calendar = QCalendarWidget(self)
#         self.calendar.selectionChanged.connect(self.date_changed)

#         self.date_label = QLabel("Selected Date: None", self)

#         self.time_selector = TimeSelectorWidget()

#         layout.addWidget(self.calendar)
#         layout.addWidget(self.date_label)
#         layout.addWidget(self.time_selector)

#     def date_changed(self):
#         date = self.calendar.selectedDate()
#         self.date_label.setText(f"Selected Date: {date.toString()}")


class VideoProcessor(QThread):
    finished = Signal(list, str)

    def __init__(self, video_paths, start_times, output_folder):
        super().__init__()
        self.video_paths = video_paths
        self.start_times = self.parse_times(start_times)
        # self.start_times = start_times
        self.output_folder = output_folder

    def parse_times(self, time_strings):
        times_in_seconds = []
        for time_str in time_strings:
            parts = list(map(int, time_str.split(':')))
            if len(parts) == 2:  # MM:SS
                seconds = parts[0] * 60 + parts[1]
            elif len(parts) == 3:  # HH:MM:SS
                seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
            else:
                raise ValueError("时间格式必须为 MM:SS 或 HH:MM:SS")
            times_in_seconds.append(seconds)
        return times_in_seconds
    
    def run(self):
        output_frames = []
        output_fps = None
        time_series = []

        for i, video_path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if output_fps is None:
                output_fps = video_fps

            frames_to_fill = int((self.start_times[i] - (len(output_frames) / output_fps)) * output_fps)

            if frames_to_fill > 0:
                ret, frame = cap.read()
                if not ret:
                    print(f"Cannot read video {video_path}")
                    return
                blank_frame = np.zeros_like(frame)
                output_frames.extend([blank_frame] * frames_to_fill)
                time_series.append((len(output_frames) / output_fps, frames_to_fill / output_fps, 'Blank'))
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            video_start = len(output_frames) / output_fps
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                output_frames.append(frame)
            video_end = len(output_frames) / output_fps
            time_series.append((video_start, video_end - video_start, 'Video'))
            cap.release()

        if not output_frames:
            print("No available output frames")
            return

        height, width, _ = output_frames[0].shape
        output_path = os.path.join(self.output_folder, 'output.mp4')
        output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (width, height))
        # output_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (width, height))

        for frame in output_frames:
            output_video.write(frame)

        output_video.release()
        # print("视频处理完成，已保存为 output.mp4")

        # self.finished.emit(time_series)
        self.finished.emit(time_series, output_path)

class VideoEditor(QDialog):
    def __init__(self, main_widget):
        super().__init__()
        self.setWindowTitle("Select Video Folder")
        self.setGeometry(100, 100, 600, 400)
        self.main_widget = main_widget

        self.layout = QVBoxLayout()

        self.video_paths = []
        self.start_times = []
        self.output_folder = None

        self.add_folder_button = QPushButton("Select Folder")
        self.add_folder_button.clicked.connect(self.add_folder)

        self.start_time_input = QLineEdit()
        self.start_time_input.setPlaceholderText("Enter start times, separated by commas (HH:MM or HH:MM:SS)")

        self.process_button = QPushButton("Start Processing")
        self.process_button.clicked.connect(self.process_videos)

        self.layout.addWidget(self.add_folder_button)
        self.layout.addWidget(QLabel("Start Times:"))
        self.layout.addWidget(self.start_time_input)
        self.layout.addWidget(self.process_button)

        self.setLayout(self.layout)
        self.plot_window = None

    def add_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.video_paths = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.avi')
            ]
            self.output_folder = folder_path
            for path in self.video_paths:
                file_name = os.path.basename(path)
                self.layout.addWidget(QLabel(file_name))
                # self.layout.addWidget(QLabel(path))

    # def process_videos(self):
    #     # 弹出窗口提示，正在处理视频
    #     self.plot_window = QMessageBox(self)
    #     self.plot_window.setWindowTitle("Processing Videos")
    #     self.plot_window.setText("Please wait while the videos are being processed...")
    #     self.plot_window.show()
        
        # start_times = list(map(int, self.start_time_input.text().split(',')))
        # if len(start_times) != len(self.video_paths):
        #     print("Number of start times does not match the number of videos")
        #     return

    def process_videos(self):



        start_times = self.start_time_input.text().split(',')
        self.test_times(start_times)
        if len(start_times) != len(self.video_paths):
            print("Number of start times does not match the number of videos")
            return
        
        if not self.output_folder:
            return
        
        # 弹出窗口提示，正在处理视频
        self.plot_window = QMessageBox(self)
        self.plot_window.setWindowTitle("Processing Videos")
        self.plot_window.setText("Please wait while the videos are being processed...")
        self.plot_window.show()

        self.processor_thread = VideoProcessor(self.video_paths, start_times, self.output_folder)
        self.processor_thread.finished.connect(self.on_processing_finished)
        self.processor_thread.start()

    def test_times(self, time_strings):
        times_in_seconds = []
        for time_str in time_strings:
            parts = list(map(int, time_str.split(':')))
            if len(parts) == 2:  # MM:SS
                seconds = parts[0] * 60 + parts[1]
            elif len(parts) == 3:  # HH:MM:SS
                seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
            else:
                error_message_box = QMessageBox()
                error_message_box.setWindowTitle("Error")
                error_message_box.setText("Time format must be MM:SS or HH:MM:SS")
                error_message_box.exec()
                raise ValueError("时间格式必须为 MM:SS 或 HH:MM:SS")
        

    def on_processing_finished(self,time_series, video_path):
        self.main_widget.handle_finished(time_series, video_path)
        self.close()



# 带删除的选择框
class ReComboBox:
    def __init__(self, _comboBox, label_dict):
        self.comboBox = _comboBox
        self.comboBox.setModel(QStandardItemModel(self.comboBox))
        self.label_dict = label_dict

    def addItem(self, itemTxt):
        QS_item = QStandardItem(itemTxt)
        # 设置文字颜色
        QS_item.setBackground(QColor('#19232d'))
        QS_item.setForeground(QColor('#ffffff'))
        QS_item.setText(itemTxt)
        self.comboBox.model().appendRow(QS_item)
        index = self.comboBox.count()-1
        self.comboBox.view().repaint()
        self.add_btn(index, itemTxt)

    def add_btn(self, _index, _itemTxt):
        # 创建一个水平布局，并将标签和删除按钮添加到其中
        layout = QHBoxLayout()
        layout.setContentsMargins(75, 0, 0, 0)
        layout.setAlignment(Qt.AlignRight)  # Align the button to the right
        button = QPushButton('x')
        button.setStyleSheet("QPushButton { border: none; color:#6D6D6D ; font-size: 15px}")
        button.setFixedSize(20, 20)
        layout.addWidget(button)
        # 将水平布局添加到下拉菜单项的QWidget中
        widget = QWidget()
        widget.setLayout(layout)
        item = self.comboBox.model().item(_index)
        item.setSizeHint(widget.sizeHint())
        self.comboBox.view().setIndexWidget(item.index(), widget)
        # 将按钮连接到槽函数，用于从下拉列表中删除相应的项目
        button.clicked.connect(lambda: self.remove_Row(_itemTxt))

    def remove_Row(self, i):
        reply = QMessageBox.question(
            None,
            "Confirm Delete",
            f"Are you sure you want to remove '{i}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            index = self.comboBox.findText(i)
            if index != -1:
                self.comboBox.model().removeRow(index)
                # Remove the item from the dictionary
                if i in self.label_dict:
                    del self.label_dict[i]
                    # print(self.label_dict)
                    # print(f"Removed {i} from dictionary")


# 定义LabelOption类，继承自QDialog
class LabelOption(QDialog):
    def __init__(self, label_dict):
        super().__init__()

        # 创建一个垂直布局
        layout = QVBoxLayout()
        # 保存标签字典
        self.label_dict = label_dict
        # 创建一个空字典来保存单选按钮
        self.radio_buttons = {}
        # 遍历标签字典
        for label, lid in label_dict.items():
            # 为每个标签创建一个单选按钮
            self.radio_buttons[label] = QRadioButton(label)
            # 将单选按钮添加到布局中
            layout.addWidget(self.radio_buttons[label])

        # 创建确认按钮
        self.confirm_button = QPushButton("Confirm")
        # 连接确认按钮的点击事件到confirm_selection方法
        self.confirm_button.clicked.connect(self.confirm_selection)
        # 将确认按钮添加到布局中
        layout.addWidget(self.confirm_button)
        # 设置布局
        self.setLayout(layout)

    # 确认选择的方法
    def confirm_selection(self):
        # 遍历标签字典
        for label, lid in self.label_dict.items():
            # 如果单选按钮被选中
            if self.radio_buttons[label].isChecked():
                # 设置选中的选项为当前标签
                selected_option = label
                # 接受对话框，关闭对话框
                self.accept()
                # 返回选中的选项
                return selected_option
            else:
                # 如果没有选中任何选项，设置为None
                selected_option = None
        # 接受对话框，关闭对话框
        self.accept()
        # 返回选中的选项
        return selected_option

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
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
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
        except Exception as e:
            print('Save data error:', e)
        else:
            print(f'File saved at {new_path}')
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


class Backend(QObject):
    highlightDotByindex = Signal(int, float, float)
    # TODO 参数待定
    highlightScatterDotByindexSign = Signal(int)
    getSelectedAreaByHtml = Signal(str)
    setStartEndTime = Signal(str, str)
    setStartAndEndDataSign = Signal(str, str, str, str, str, str)
    getSelectedAreaToSaveSign = Signal(str)
    getSelectedAreaToSaveTimerSign = Signal(str)

    def __init__(self):
        super().__init__()
        self.data = None
        self.select_option = None

    # 创建一个函数来找到最近的有效索引
    @Slot(pd.DataFrame)
    def handle_data_changed(self, data):
        self.data = data  # Update the data attribute
        # print("Backend's DataFrame has been updated:")
        # print(self.data)

    @Slot()
    def handle_label_change(self, option):
        self.select_option = option

    @Slot(result='QString')
    def get_label_option(self):
        result = self.select_option if self.select_option is not None else ""
        # print(f"Returning: {result}")
        return result

    # 通过索引高亮散点，点击地图散点高亮折线图散点
    @Slot()
    def triggeLineChartHighlightDotByIndex(self, index):
        print(f"Triggering highlight dot({index})...")
        self.view.page().runJavaScript(f"highlightLineChartDotByIndex('{index}')")

    # 设置开始结束时间到标签
    @Slot(str, str)
    def setStartEndTimeToLabel(self, start_time, end_time):
        print(f"Setting start time: {start_time}, end time: {end_time}")
        self.setStartEndTime.emit(start_time, end_time)

    @Slot(str, str, str, str, str, str)
    def setStartAndEndData(self, id1, lon1, lat1, id2, lon2, lat2):
        # print("Setting start and end data...")
        self.setStartAndEndDataSign.emit(id1, lon1, lat1, id2, lon2, lat2)

    # 通过索引高亮散点，点击折线图散点高亮散点图散点
    @Slot(int)
    def handleHighlightScatterDotByIndex(self, index):
        print(f"Triggering highlight dot({index})...")
        self.highlightScatterDotByindexSign.emit(index)
        

    # 通过索引高亮散点，点击折线图散点高亮地图散点
    @Slot(int)
    def handleHighlightDotByIndex(self, index):
        lat, lon = self.data.loc[index, 'latitude'], self.data.loc[index, 'longitude']

        # # 获取所有非空纬度和经度的索引
        # valid_lat_indices = self.data.index[~self.data['latitude'].isna()].to_numpy()
        # valid_lon_indices = self.data.index[~self.data['longitude'].isna()].to_numpy()

        # # 如果纬度或经度为空，找到最近的有效值
        # if pd.isna(lat) or pd.isna(lon):

        #     if pd.isna(lat):
        #         nearest_lat_index = find_nearest_index(index, valid_lat_indices)
        #         if nearest_lat_index is not None:
        #             lat = self.data.loc[nearest_lat_index, 'latitude']
        #     if pd.isna(lon):
        #         nearest_lon_index = find_nearest_index(index, valid_lon_indices)
        #         if nearest_lon_index is not None:
        #             lon = self.data.loc[nearest_lon_index, 'longitude']

        if pd.isna(lat) or pd.isna(lon):
            print("Latitude or longitude is missing.")
            return

        print("handleing highlight dot...")
        self.highlightDotByindex.emit(index, lat, lon)

    # add label按钮点击事件
    @Slot()
    def handleAddLabel(self, selected_option):
        print("Adding label...")
        self.view.page().runJavaScript(f"addLabel('{selected_option}')")

    # delete label按钮点击事件
    @Slot(int)
    def handleDeleteLabel(self, status):
        print("Deleting label...")
        self.view.page().runJavaScript(f"deleteLabel('{status}')")

    # 从框选的散点图设置折线图markData
    @Slot(str)
    def setMarkData(self, data):
        print("Setting mark data...")
        self.view.page().runJavaScript(f"setMarkData('{data}')")

    # 清空折线图markData
    @Slot()
    def clearMarkData(self):
        print("Clearing mark data...")
        self.view.page().runJavaScript("clearMarkData()")

    # 从html获取折线图框选区域
    # @Slot(result='QVariant')
    @Slot()
    def getSelectedArea(self):
        print("etSelectedArea..")
        return self.view.page().runJavaScript("getSelectedArea()", 0, self.test_callback)

    def test_callback(self, result):
        self.getSelectedAreaByHtml.emit(result)

    # @Slot()
    # def getSelectedAreaToSave(self):
    #     print("getSelectedAreaToSave..")
    #     return self.view.page().runJavaScript("getSelectedArea()", 0, self.save_callback)

    # def save_callback(self, result):
    #     # print(result)
    #     self.getSelectedAreaToSaveSign.emit(result)

    @Slot()
    def getSelectedAreaToSave(self, is_timer):
        print("getSelectedAreaToSave..")
        # 使用lambda传递参数给save_callback
        self.view.page().runJavaScript("getSelectedArea()", 0, lambda result: self.save_callback(result, is_timer))

    def save_callback(self, result, is_timer):
        # 根据传入的参数选择要发射的信号
        if is_timer == 1:
            self.getSelectedAreaToSaveTimerSign.emit(result)
        elif is_timer == 0:
            self.getSelectedAreaToSaveSign.emit(result)

    # 添加label弹出确认提示框
    @Slot(result='QVariant')
    def confirmOverlap(self):
        # 显示确认对话框
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setText("Labels overlapped, Over write?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        result = msg_box.exec() == QMessageBox.Yes
        # print(result)
        # 返回布尔值
        return result

    # 删除标签弹出确认提示框
    @Slot(result='QVariant')
    def confirmDelete(self):
        # 显示确认对话框
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setText("Do you want to delete the label?")
        # msg_box.setText("是否删除选中的标记区域？")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        result = msg_box.exec() == QMessageBox.Yes
        # print(result)
        # 返回布尔值
        return result

    @Slot()
    def displayData(self, data, metadata=None, label_colors=None):
        if isinstance(data, pd.DataFrame):
            # 指定要排除的列,Python 的 json 模块不能直接序列化某些自定义对象或非基本数据类型（如 datetime、Timestamp 等
            # columns_to_drop = ['datetime', 'logger_id', 'animal_tag', 'gps_status',
            #                    'activity_class', 'label']
            #
            # # 删除指定列
            # data = data.drop(columns=columns_to_drop, errors='ignore')

            # 根据传入信号选择需要的列
            series_combined = ["timestamp", "unixtime", "index", "latitude", "longitude"] + [item for data in metadata
                                                                                             for item in
                                                                                             data["series"]]
            data = data[series_combined]

            # 将空字符串替换为 None
            data = data.replace('', None)

            # 将 NaN 值替换为 None,避免转换为json出错
            data = data.where(pd.notnull(data), None)

            # 将 DataFrame 转换为字典列表
            data_records = data.to_dict(orient='records')

            if metadata is None:
                # 创建元数据信息
                metadata = [
                    {
                        "name": "acceleration",
                        "xAxisName": "timestamp",
                        "yAxisName": "Y Axis 1",
                        "series": ["acc_x", "acc_y", "acc_z"]
                    }
                    # {
                    #     "name": "gyroscope",
                    #     "xAxisName": "timestamp",
                    #     "yAxisName": "Y Axis 2",
                    #     "series": ["gyro_x", "gyro_y", "gyro_z"]
                    # }
                ]

            # 将元数据和数据打包到一个字典中
            result = {
                "metadata": metadata,
                "data": data_records,
                "labelColors": label_colors
            }
        else:
            # 如果数据不是 DataFrame，则直接使用传入的数据
            result = data
        # 将结果转换为 JSON 格式
        json_data = json.dumps(result)
        # print(json_data)
        self.view.page().runJavaScript(f"displayData('{json_data}')")

    # @Slot()
    # def displayData(self, data, metadata=None, label_colors=None):
    #     start_time = time.time()
    #     if isinstance(data, pd.DataFrame):
    #         # 指定要排除的列,Python 的 json 模块不能直接序列化某些自定义对象或非基本数据类型（如 datetime、Timestamp 等
    #         # columns_to_drop = ['datetime', 'logger_id', 'animal_tag', 'gps_status',
    #         #                    'activity_class', 'label']
    #         #
    #         # # 删除指定列
    #         # data = data.drop(columns=columns_to_drop, errors='ignore')
    #         step_time = time.time()

    #         # 根据传入信号选择需要的列
    #         series_combined = ["timestamp", "unixtime", "index", "latitude", "longitude"] + [item for data in metadata
    #                                                                                          for item in
    #                                                                                          data["series"]]
    #         print(f"Time for selecting columns: {time.time() - step_time:.6f} seconds")

    #         step_time = time.time()
    #         data = data[series_combined]

    #         print(f"Time for da: {time.time() - step_time:.6f} seconds")

    #         step_time = time.time()

    #         # 将空字符串替换为 None
    #         data = data.replace('', None)
    #         print(f"Time for replace: {time.time() - step_time:.6f} seconds")

    #         step_time = time.time()

    #         # 将 NaN 值替换为 None,避免转换为json出错
    #         data = data.where(pd.notnull(data), None)

    #         print(f"Time for where: {time.time() - step_time:.6f} seconds")

    #         step_time = time.time()

    #         # 将 DataFrame 转换为字典列表
    #         data_records = data.to_dict(orient='records')

    #         print(f"Time for dict: {time.time() - step_time:.6f} seconds")

    #         step_time = time.time()

    #         if metadata is None:
    #             # 创建元数据信息
    #             metadata = [
    #                 {
    #                     "name": "acceleration",
    #                     "xAxisName": "timestamp",
    #                     "yAxisName": "Y Axis 1",
    #                     "series": ["acc_x", "acc_y", "acc_z"]
    #                 }
    #                 # {
    #                 #     "name": "gyroscope",
    #                 #     "xAxisName": "timestamp",
    #                 #     "yAxisName": "Y Axis 2",
    #                 #     "series": ["gyro_x", "gyro_y", "gyro_z"]
    #                 # }
    #             ]

    #         # 将元数据和数据打包到一个字典中
    #         result = {
    #             "metadata": metadata,
    #             "data": data_records,
    #             "labelColors": label_colors
    #         }
    #     else:
    #         # 如果数据不是 DataFrame，则直接使用传入的数据
    #         result = data
    #     # 将结果转换为 JSON 格式
    #     json_data = json.dumps(result)
    #     print(f"Time for converting to JSON: {time.time() - step_time:.6f} seconds")

    #     # print(json_data)
    #     self.view.page().runJavaScript(f"displayData('{json_data}')")

    # def displayData(self, data, metadata=None, label_colors=None):
    #     timings = {}
    #     start_time = time.time()

    #     if isinstance(data, pd.DataFrame):
    #         step_time = time.time()

    #         # 根据传入信号选择需要的列
    #         series_combined = ["timestamp", "unixtime", "index", "latitude", "longitude"] + [
    #             item for data in metadata for item in data["series"]
    #         ]
    #         data = data[series_combined]
    #         timings['select_columns'] = time.time() - step_time

    #         step_time = time.time()
    #         # 将空字符串替换为 None
    #         data = data.replace('', None)
    #         timings['replace_empty_strings'] = time.time() - step_time

    #         step_time = time.time()
    #         # 将 NaN 值替换为 None,避免转换为json出错
    #         data = data.where(pd.notnull(data), None)
    #         timings['replace_nan'] = time.time() - step_time

    #         step_time = time.time()
    #         # 将 DataFrame 转换为字典列表
    #         data_records = data.to_dict(orient='records')
    #         timings['convert_to_dict'] = time.time() - step_time

    #         step_time = time.time()
    #         if metadata is None:
    #             # 创建元数据信息
    #             metadata = [
    #                 {
    #                     "name": "acceleration",
    #                     "xAxisName": "timestamp",
    #                     "yAxisName": "Y Axis 1",
    #                     "series": ["acc_x", "acc_y", "acc_z"]
    #                 }
    #             ]
    #         timings['handle_metadata'] = time.time() - step_time

    #         step_time = time.time()
    #         # 将元数据和数据打包到一个字典中
    #         result = {
    #             "metadata": metadata,
    #             "data": data_records,
    #             "labelColors": label_colors
    #         }
    #         timings['create_result'] = time.time() - step_time
    #     else:
    #         step_time = time.time()
    #         # 如果数据不是 DataFrame，则直接使用传入的数据
    #         result = data
    #         timings['handle_non_dataframe'] = time.time() - step_time

    #     step_time = time.time()
    #     # 将结果转换为 JSON 格式
    #     json_data = json.dumps(result)
    #     timings['convert_to_json'] = time.time() - step_time

    #     step_time = time.time()
    #     # 执行 JavaScript
    #     self.view.page().runJavaScript(f"displayData('{json_data}')")
    #     timings['run_javascript'] = time.time() - step_time

    #     timings['total'] = time.time() - start_time

    #     print_str = set()
    #     # 打印所有耗时信息
    #     for step, duration in timings.items():
    #         print_str.add(f"{step}: {duration:.6f} seconds")
    #         print("\n".join(print_str))
    #         # print(f"{step}: {duration:.6f} seconds")

    # 更新labelColors
    @Slot()
    def updateLabelColors(self, label_colors):
        label_colors = json.dumps(label_colors)
        self.view.page().runJavaScript(f"handleLabelColorChange('{label_colors}')")

    # combox选择事件
    @Slot()
    def handleComboxSelection(self, charts_data):
        charts_data = json.dumps(charts_data)
        print("Combox selection...")
        self.view.page().runJavaScript(f"handleComboxChange('{charts_data}')")

    def handleJavaScriptLog(self, result):
        print(f"JavaScript log: {result}")

    @Slot(str)
    def receiveData(self, data):
        print("Received data from frontend:", data)

    @Slot()
    def triggerUpdate(self):
        self.view.page().runJavaScript("getInputValue()")  # 调用前端的getInputValue函数


class BackendMap(QObject):
    highlightLineChartDotByindex = Signal(int)

    def __init__(self):
        super().__init__()

    # 点击折线图高亮地图散点，没有则添加新点
    @Slot(str, float, float)
    def triggeLineMapHighlightDotByIndex(self, index, lat, lon):
        print(f"Triggering highlight dot({index})...")
        # self.view.page().runJavaScript(f"highlightByIndexAndLatLng('{index}, {lat}, {lon}')")
        self.view.page().runJavaScript(f"highlightByIndexAndLatLng('{index}', {lat}, {lon})")

    # 点击地图高亮折线图散点
    @Slot(int)
    def handleHighlightLineDotByIndex(self, index):
        print("handleing highlight dot...")
        self.highlightLineChartDotByindex.emit(index)

    @Slot()
    def highlightLineChartTwoDots(self, id1, lon1, lat1, id2, lon2, lat2):
        print("highlightLineChartTwoDots...")
        self.view.page().runJavaScript(f"highlightTwoMarkers('{id1}', {lat1}, {lon1}, '{id2}', {lat2}, {lon2})")

    # 在data display之后读取gps边界，然后作为初始地图
    @Slot()
    def displayMapData(self, data):
        if isinstance(data, pd.DataFrame):
            # 将列名转换为列表
            columns_list = data.columns.tolist()
            logging.debug(f"columns_list: {columns_list}")
            # 选择需要的列 index、latitude 和 longitude，并去除 latitude 和 longitude 中的缺失值。
            data = data[['index', 'latitude', 'longitude']].dropna(subset=['latitude', 'longitude'])
            # 使用 iloc 按索引进行降采样, 一万个条目取一个。
            interval = 10 * 60 * 25  # 25 is sampling rate, 10 is minutes
            data = data.iloc[::interval]

            data = data.to_dict(orient='records')
        data = json.dumps(data)
        self.view.page().runJavaScript(f"displayMapData('{data}')")

    def handleJavaScriptLog(self, result):
        print(f"JavaScript log: {result}")


# 定义LabelWithInteractivePlot类，继承自QWidget
class LabelWithInteractivePlot(QWidget):
    dataChanged = Signal(pd.DataFrame)

    def __init__(self, root, cfg) -> None:
        super().__init__()
        # 创建一个日志记录器
        self.end_indice = []
        self.start_indice = []
        self.logger = logging.getLogger("GUI")
        self.backend = Backend()
        self.backend_map = BackendMap()
        self.yaml = YAML()

        self.select_video_widget = None
        self.plot_window = None
        self.time_series = None

        self.save_csv_thread_pool = QThreadPool()

        # self.data改变时同步数据到backend
        self.dataChanged.connect(self.backend.handle_data_changed)

        # 将折线图backend的点击折线图事件连接到backend_map的高亮地图散点方法
        self.backend.highlightDotByindex.connect(self.backend_map.triggeLineMapHighlightDotByIndex)
        # 点击折线散点高亮散点图散点
        self.backend.highlightScatterDotByindexSign.connect(self.handle_highlight_scatter_dot_by_index)
        # 点击地图散点高亮折线图散点
        self.backend_map.highlightLineChartDotByindex.connect(self.backend.triggeLineChartHighlightDotByIndex)
        # 点击地图散点高亮散点图散点
        self.backend_map.highlightLineChartDotByindex.connect(self.handle_highlight_scatter_dot_by_index)
        self.backend.getSelectedAreaByHtml.connect(self.handleReflectToLatent)
        self.backend.setStartEndTime.connect(self.setStartEndTime)
        self.backend.setStartAndEndDataSign.connect(self.backend_map.highlightLineChartTwoDots)
        self.backend.getSelectedAreaToSaveSign.connect(self.getSelectedAreaToSave)
        self.backend.getSelectedAreaToSaveTimerSign.connect(self.getSelectedAreaToSaveTimer)


        self.button_style = """QPushButton {
            background-color: #1ea123; 
            border: none;
            color: white;
            padding: 5px 10px;
            text-align: center;
            text-decoration: none;
            font-size: 12px;
            margin: 4px 2px;
            border-radius: 10px; 
        }

        QPushButton:pressed {
            background-color: #148f1d; 
        }"""

        self.remove_button_style = """QPushButton { 
            border: none;
            color:#6D6D6D; 
            font-size: 15px; 
            }
        """

        # 初始化特征提取按钮为None
        self.featureExtractBtn = None
        # 初始化当前高亮散点为None
        self.current_highlight_map_scatter = None

        # 保存根对象
        self.root = root
        # 读取根对象的配置
        root_cfg = read_config(root.config)
        # 保存标签字典
        self.label_dict = root_cfg['label_dict']


        # 初始化最后修改的点
        self.last_modified_points = []
        # 预定义颜色数组
        self.colorPalette = ['#91cc75', '#5470c6', '#fac858', '#ee6666',
                             '#73c0de', '#3ba272', '#fc8452', '#9a60b4',
                             '#ea7ccc', '#fff018', '#6800ff', '#4bb0ff',
                             '#1bff00', '#09ffdb']
        # 保存标签颜色字典 
        self.label_colors = {}
        self.init_label_colors()

        # 保存传感器字典
        self.sensor_dict = root_cfg['sensor_dict']
        self.all_sensor = list(self.sensor_dict.keys())

        # 初始化主布局、顶部布局和底部布局
        self.initLayout()

        # 创建一个QTimer对象
        self.computeTimer = QTimer()

        # 创建一个空的DataFrame
        self.data = pd.DataFrame()
        # 配置
        self.cfg = cfg

        self.db_path = os.path.join(self.cfg["project_path"], get_db_folder(), "database.db")
        self.video_path = os.path.join(self.cfg["project_path"], "videos")

        # 模型参数
        self.model_path_list = None
        # 初始化模型路径列表
        self.model_path = []
        # 初始化模型名称
        self.model_name = ''
        # 初始化数据长度
        self.data_length = 180
        # 初始化列名列表
        self.column_names = []

        # 手动校准视频时间
        self.offset = 0.0

        # 初始化最小时间
        self.min_time = 0

        # 初始化视频路径
        self.cap = None

        # 状态
        # 初始化训练状态为False
        self.isTraining = False
        # 初始化模式为空字符串
        self.mode = ''

        # 创建右上模型数据选择区域
        self.createModelSelectLabelArea()

        # 创建右中按钮
        self.createSettingArea()

        # 创建左中按钮区域
        self.createLeftBotton()

        # 创建左上视频区域
        self.createVideoArea()

        # 初始化定时器，保存到csv
        self.init_timer()

        # 更新按钮状态
        self.updateBtn()

        self.model_watcher = QFileSystemWatcher()
        self.data_watcher = QFileSystemWatcher()

        self.model_watcher.directoryChanged.connect(self.update_model_combobox)
        self.data_watcher.directoryChanged.connect(self.update_data_combobox)

        self.update_model_combobox()
        self.update_data_combobox()

    def init_label_colors(self):
        self.label_colors = {}
        for i, label in enumerate(self.label_dict.keys()):
            # 使用取余运算符来循环使用颜色
            color_index = i % len(self.colorPalette)
            self.label_colors[label] = self.colorPalette[color_index]

        # self.logger.debug(
        #     "Attempting..."
        # )


    def initLayout(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)


        # 创建第一行三个按钮布局
        self.first_row_layout = QHBoxLayout()


        # 创建第二行选择框和颜色布局
        self.second_row_layout = QVBoxLayout()


        # 创建第三行（包含三个图和按钮）布局
        self.third_row_layout = QHBoxLayout()
        # 地图布局
        self.left_row1_layout = QHBoxLayout()
        # 视频布局
        self.left_row1_video_layout = QVBoxLayout()
        # 散点图布局
        self.row3_layout = QHBoxLayout()
        # 按钮布局,放入垂直的多个按钮
        self.charts_show_button_layout = QVBoxLayout()

        self.third_row_layout.addLayout(self.left_row1_layout, 1)
        self.third_row_layout.addLayout(self.left_row1_video_layout, 1)
        self.third_row_layout.addLayout(self.row3_layout, 1)
        self.third_row_layout.addLayout(self.charts_show_button_layout, 0)


        # 创建第四行布局，一个折线图，一个聚合按钮输入框图表占最大，按钮占最小
        self.fourth_row_layout = QHBoxLayout()

        # 折线图布局
        self.left_row3_layout = QHBoxLayout()

        # 按键聚合布局
        self.nestend_button_layout = QVBoxLayout()
        self.label_edit_button_layout = QVBoxLayout()
        # 放两个时间输入框
        self.st_end_time_layout = QVBoxLayout()
        self.first_edit_button_layout = QHBoxLayout()
        self.second_edit_button_layout = QHBoxLayout()
        self.third_edit_button_layout = QHBoxLayout()

        self.label_edit_button_layout.addLayout(self.first_edit_button_layout)
        self.label_edit_button_layout.addLayout(self.second_edit_button_layout)
        self.label_edit_button_layout.addLayout(self.third_edit_button_layout)

        self.nestend_button_layout.addStretch()
        self.nestend_button_layout.addLayout(self.st_end_time_layout)
        self.nestend_button_layout.addLayout(self.label_edit_button_layout)
        self.nestend_button_layout.addStretch()

        self.fourth_row_layout.addLayout(self.left_row3_layout,1)
        self.fourth_row_layout.addLayout(self.nestend_button_layout,0)


        self.main_layout.addLayout(self.first_row_layout,0)
        self.main_layout.addLayout(self.second_row_layout,0)
        self.main_layout.addLayout(self.third_row_layout,1)
        self.main_layout.addLayout(self.fourth_row_layout,1)
        # 创建中心图表
        self.createCenterPlot()






        

    # 初始化布局的方法
    def initLayout_old(self):

        # 创建主水平布局
        self.main_layout = QHBoxLayout()

        # 设置主布局
        self.setLayout(self.main_layout)

        # 创建左侧布局
        self.left_layout = QVBoxLayout()



        # 创建左侧row1布局
        self.left_row1_layout_all = QHBoxLayout()
        self.left_row1_layout = QHBoxLayout()
        self.left_row1_layout = QHBoxLayout()
        self.left_row1_layout_all.addLayout(self.left_row1_layout)
        self.left_row1_layout_all.addLayout(self.left_row1_video_layout)
        self.left_layout.addLayout(self.left_row1_layout_all)
        # self.left_layout.addLayout(self.left_row1_layout)

        # 创建左侧row2布局
        self.left_row2_layout = QVBoxLayout()
        self.left_layout.addLayout(self.left_row2_layout)

        # 创建左侧row3布局
        self.left_row3_layout = QHBoxLayout()
        self.left_layout.addLayout(self.left_row3_layout)

        # 创建右侧布局
        self.right_layout = QVBoxLayout()

        # 创建一个 QVBoxLayout 用于 row1_layout 中的多行布局
        self.nestend_layout = QVBoxLayout()

        # 创建 row1_layout 并将嵌套布局添加到其中
        self.row1_layout = QHBoxLayout()
        self.row1_layout.addLayout(self.nestend_layout)

        self.right_layout.addLayout(self.row1_layout)

        self.row2_layout = QVBoxLayout()
        self.right_layout.addLayout(self.row2_layout)

        self.row3_layout = QHBoxLayout()
        self.right_layout.addLayout(self.row3_layout)

        # # 创建一个弹簧 (QSpacerItem)
        # self.spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        # self.right_layout.addItem(self.spacer)

        # 将左侧和右侧布局添加到主布局中，并设置相同的 stretch 因子，使它们宽度相同
        left_column = QWidget()
        left_column.setLayout(self.left_layout)
        self.main_layout.addWidget(left_column, 1)

        right_column = QWidget()
        right_column.setLayout(self.right_layout)
        self.main_layout.addWidget(right_column, 1)

        # 创建中心图表
        self.createCenterPlot()

    def init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.backend.getSelectedAreaToSave(1))
        # 设置定时器每隔五分钟（300000 毫秒）触发一次
        self.timer.start(600000)

    '''
    ==================================================
    左中按键区域
    ==================================================
    '''

    def createLeftBotton(self):
        # 第一行start time显示框
        self.left_start_time_layout = QHBoxLayout()
        self.start_input_box = QLineEdit(self)
        self.start_input_box.setPlaceholderText("start time")
        self.left_start_time_layout.addWidget(QLabel("Start time:"))
        self.left_start_time_layout.addWidget(self.start_input_box)
        self.left_start_time_layout.addStretch()
        self.st_end_time_layout.addLayout(self.left_start_time_layout)

        # self.left_row2_layout.addLayout(self.left_start_time_layout)

        # 第二行end time显示框
        self.left_end_time_layout = QHBoxLayout()
        self.end_input_box = QLineEdit(self)
        self.end_input_box.setPlaceholderText("end time")
        self.left_end_time_layout.addWidget(QLabel("End time: "))
        self.left_end_time_layout.addWidget(self.end_input_box)
        self.left_end_time_layout.addStretch()
        self.st_end_time_layout.addLayout(self.left_end_time_layout)
        # self.left_row2_layout.addLayout(self.left_end_time_layout)

        # 第三行label选项框
        # self.left_label_layout = QHBoxLayout()

        self.label_combobox = QComboBox()
        self.label_combobox.setStyleSheet(combobox_style_light)
        self.label_combobox.setModel(QStandardItemModel(self.label_combobox))

        # self.comboBoxHandler = ReComboBox(self.label_combobox, self.label_dict)
        # self.label_dict
        # self.label_combobox = ReComboBox()


        # for label in self.label_dict.keys():
        #     self.comboBoxHandler.addItem(label)

        for item in self.label_dict.keys():
            self.addItem(item)
        
        self.label_combobox.currentTextChanged.connect(
            self.backend.handle_label_change
        )
        
        # for label in self.label_dict.keys():
        #     self.label_combobox.addItem(label)
        # self.backend.handle_label_change(self.label_combobox.currentText())
        # self.label_combobox.currentTextChanged.connect(
        #     self.backend.handle_label_change
        # )

        # self.left_label_layout.addWidget(QLabel("Label:     "))
        # self.left_label_layout.addWidget(self.label_combobox, alignment=Qt.AlignLeft)
        self.first_edit_button_layout.addWidget(QLabel("Label:     "))
        self.first_edit_button_layout.addWidget(self.label_combobox, alignment=Qt.AlignLeft)

        # 创建label按钮
        add_label_btn = QPushButton('Create label')
        add_label_btn.clicked.connect(self.add_item)
        add_label_btn.setStyleSheet(self.button_style)
        # self.left_label_layout.addWidget(add_label_btn)
        self.first_edit_button_layout.addWidget(add_label_btn)

        # 暂时不用
        self.save_label_btn = QPushButton('Save label')
        self.save_label_btn.clicked.connect(self.save_label)
        self.save_label_btn.setStyleSheet(self.button_style)
        # self.left_label_layout.addWidget(self.save_label_btn)


        # 保存csv按钮 TODO 问一下这个还要不
        self.save_csv_btn = QPushButton('Save csv')
        self.save_csv_btn.clicked.connect(lambda: self.backend.getSelectedAreaToSave(0))
        self.save_csv_btn.setStyleSheet(self.button_style)
        # self.left_label_layout.addWidget(self.save_csv_btn)
        # self.left_label_layout.addStretch()
        # self.left_row2_layout.addLayout(self.left_label_layout)


    

        # 第四行三个按钮
        # self.left_button_layout = QHBoxLayout()
        # add label按钮
        add_label_btn = QPushButton('Add label')
        add_label_btn.clicked.connect(lambda: self.backend.handleAddLabel(self.label_combobox.currentText()))
        # 设置按钮样式
        add_label_btn.setStyleSheet(self.button_style)

        # delete label按钮
        delete_label_btn = QPushButton('Delete label')
        delete_label_btn.setCheckable(True)  # Make the button checkable
        delete_label_btn.clicked.connect(lambda: self.backend.handleDeleteLabel(int(delete_label_btn.isChecked())))
        # Set the button style based on the checked state
        delete_label_btn.setStyleSheet(
            self.button_style + "background-color: red;" if delete_label_btn.isChecked() else self.button_style + "background-color: green;")

        # Reflect to latent space按钮
        reflect_to_latent_btn = QPushButton('View on latent space')
        reflect_to_latent_btn.clicked.connect(lambda: self.backend.getSelectedArea())
        # 设置按钮样式
        reflect_to_latent_btn.setStyleSheet(self.button_style)


        self.second_edit_button_layout.addWidget(add_label_btn)
        self.second_edit_button_layout.addWidget(delete_label_btn)
        self.second_edit_button_layout.addWidget(self.save_csv_btn)

        self.third_edit_button_layout.addWidget(reflect_to_latent_btn)
        

        # # 将按钮添加到布局中
        # self.left_button_layout.addWidget(add_label_btn)
        # self.left_button_layout.addStretch(1)
        # self.left_button_layout.addWidget(delete_label_btn)
        # self.left_button_layout.addStretch(1)
        # self.left_button_layout.addWidget(reflect_to_latent_btn)
        # self.left_button_layout.addStretch(10)

        # self.left_row2_layout.addLayout(self.left_button_layout)

    def addItem(self, itemTxt):
        QS_item = QStandardItem(itemTxt)
        # QS_item.setBackground(QColor('#19232d'))
        # QS_item.setForeground(QColor('#ffffff'))
        QS_item.setBackground(QColor('#ffffff'))
        QS_item.setForeground(QColor('#19232d'))
        QS_item.setText(itemTxt)
        self.label_combobox.model().appendRow(QS_item)
        index = self.label_combobox.count() - 1
        self.add_btn(index, itemTxt)

    def add_btn(self, _index, _itemTxt):
        layout = QHBoxLayout()
        layout.setContentsMargins(75, 0, 0, 0)
        layout.setAlignment(Qt.AlignRight)
        button = QPushButton('x')
        button.setFixedSize(20, 20)
        button.setStyleSheet(self.remove_button_style)
        layout.addWidget(button)
        widget = QWidget()
        widget.setLayout(layout)
        item = self.label_combobox.model().item(_index)
        item.setSizeHint(widget.sizeHint())
        self.label_combobox.view().setIndexWidget(item.index(), widget)
        button.clicked.connect(lambda: self.remove_Row(_itemTxt))

    def remove_Row(self, i):
        reply = QMessageBox.question(
            None,
            "Confirm Delete",
            f"Are you sure you want to remove '{i}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            index = self.label_combobox.findText(i)
            if index != -1:
                self.label_combobox.model().removeRow(index)
                # Remove the item from the dictionary
                if i in self.label_dict:
                    del self.label_dict[i]
                # Update the label colors
                self.init_label_colors()
                self.backend.updateLabelColors(self.label_colors)
                # self.clear_color_layout()
                # self.display_colors(self.label_colors)
                self.clear_color_layout_and_display(self.label_colors)


    def add_item(self):
        key, ok = QInputDialog.getText(self, 'Add Label', 'Enter the Label:')
        if ok and key:
            if key not in self.label_dict:
                # value = f"{self.label_combobox.count() + 1}"
                value = self.label_combobox.count() + 1
                self.label_dict[key] = value
                self.addItem(key)
                # self.comboBoxHandler.addItem(key) updateLabelColors
                print(self.label_dict)
                self.init_label_colors()
                self.backend.updateLabelColors(self.label_colors)
                # self.clear_color_layout()
                # self.display_colors(self.label_colors)
                self.clear_color_layout_and_display(self.label_colors)

            else:
                QMessageBox.warning(self, 'Error', 'Label already exists.')




    def save_label(self):
        # 保存标签字典
        # save_label_dict(self.root, self.label_dict)
        config_path = os.path.join(self.cfg["project_path"], "config.yaml")
        with open(config_path, 'r') as f:
            config = self.yaml.load(f)
        config['label_dict'] = self.label_dict
        with open(config_path, 'w') as f:
            self.yaml.dump(config, f)
        print("Saving label Successfully")

    


    def setStartEndTime(self, start_time, end_time):
        self.start_input_box.setText(start_time)
        self.end_input_box.setText(end_time)

    # 定义一个函数将ISO格式的时间字符串转换为Unix时间戳
    def iso_to_timestamp(self, iso_str):
        dt = datetime.datetime.fromisoformat(iso_str.rstrip('Z'))
        timestamp = dt.timestamp()

        # return dt.timestamp()
        return round(timestamp, 5)

    # 更新右侧散点图的颜色
    def handleReflectToLatent(self, areaData):
        # print(areaData)
        try:
            areaData = json.loads(areaData)  # 解析 JSON 字符串
            # print("Parsed data:", areaData)
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
        spots = []
        for spot in self.scatterItem.points():
            pos = spot.pos()
            i, start, end = spot.data()
            # 如果first=False，使用已有的标签
            # 如果first=True，使用手动标签
            color = self.checkColor(self.data.loc[start, 'label'], first=True)
            # original_brush = spot.brush()  # 读取原有颜色
            # color = original_brush.color()  # 默认使用原有颜色
            # spots.append({
            # 'pos': (pos.x(), pos.y()),
            # 'data': (i, start, end),
            # 'brush': pg.mkBrush(color)
            # })
            spot = {'pos': (pos.x(), pos.y()), 'data': (i, start, end),
                    'brush': pg.mkBrush(color)}
            spots.append(spot)

        for reg in areaData:
            # 获取区域的起始和结束索引
            name = reg[0].get("name")
            first_timestamp = reg[0].get("timestamp", {}).get("start")
            second_timestamp = reg[0].get("timestamp", {}).get("end")
            new_color = reg[0].get("itemStyle", {}).get("color")

            idx_begin, idx_end = self._to_idx(int(first_timestamp), int(second_timestamp))
            for spot in spots:
                if idx_begin < spot['data'][1] and idx_end > spot['data'][2]:
                    spot['brush'] = pg.mkBrush(new_color)

        self.scatterItem.setData(spots=spots)

    '''
    ==================================================
    左区域折线图
    ==================================================
    '''
    # 在外层定义



    '''
    ==================================================
    左上视频
    ==================================================
    '''

    def createVideoArea(self):
        # 手动校准视频时间
        self.video_time_layout = QHBoxLayout()
        # self.video_time_label = QLabel("当前时间 / 总时间", self)
        # self.video_time_label = QLabel("Current Time / Total Time", self)
        self.video_time_label = ClickableLabel("Current Time / Total Time",self)
        
        self.video_time_layout.addWidget(self.video_time_label, alignment=Qt.AlignLeft)
        self.video_time_layout.addWidget(QLabel("Offset(s):"), alignment=Qt.AlignRight)
        self.timestamp_input = QDoubleSpinBox()
        self.timestamp_input.setRange(-10000.0, 10000.0)  # Set desired range
        self.timestamp_input.setSingleStep(0.1)  # Set step size for increment/decrement
        self.timestamp_input.setValue(0.0)  # Default value
        self.video_time_layout.addWidget(self.timestamp_input, alignment=Qt.AlignRight)
        
        # 视频标签
        # self.video_label = QLabel(self)
        self.video_label = ClickableLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.left_row1_video_layout.addWidget(self.video_label)
        self.left_row1_video_layout.addLayout(self.video_time_layout)

        self.video_label.clicked.connect(self.open_file_dialog)
        self.video_time_label.clicked.connect(self.open_detial_dialog)

        # self.update_video('')
        self.init_video()

    def init_video(self):
        # cv显示一个白色的图片
        self.qt_image = QImage(600, 400, QImage.Format_RGB888)
        self.qt_image.fill(Qt.white)
        self.video_label.setPixmap(QPixmap.fromImage(self.qt_image))
        self.video_label.setScaledContents(True)

    def open_detial_dialog(self):
        if self.plot_window is not None:
            self.plot_window.close()
        self.plot_window = QWidget()
        self.plot_window.setWindowTitle("Time Series Plot")
        plot_layout = QVBoxLayout()
        plot_widget = pg.PlotWidget()
        plot_layout.addWidget(plot_widget)
        self.plot_window.setLayout(plot_layout)

        if self.time_series is None:
            return

        for start, duration, label in self.time_series:
            color = QtGui.QColor(0, 0, 255) if label == 'Video' else QtGui.QColor(128, 128, 128)
            bar_graph = pg.BarGraphItem(x=[start + duration / 2],
                                        height=[1],
                                        width=[duration],
                                        brush=color)
            plot_widget.addItem(bar_graph)

        plot_widget.setYRange(-0.5, 1.5)
        # plot_widget.setLabel('bottom', 'Time (s)')
        plot_widget.setLabel('bottom', 'Time (HH:MM:SS)')
        plot_widget.setTitle('Video and Blank Periods')

        # Convert time in seconds to HH:MM:SS format for x-axis
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02}:{minutes:02}:{seconds:02}"

        tick_values = [start for start, _, _ in self.time_series]
        tick_strings = [format_time(tick) for tick in tick_values]
        plot_widget.getAxis('bottom').setTicks([list(zip(tick_values, tick_strings))])
        
        self.plot_window.show()

    # def open_file_dialog(self):
    #     file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
    #     if file_path:
    #         self.update_video(file_path)

    def plot_time_series(self, time_series):
        self.time_series = time_series
        if self.plot_window is not None:
            self.plot_window.close()
        self.plot_window = QWidget()
        self.plot_window.setWindowTitle("Time Series Plot")
        plot_layout = QVBoxLayout()
        plot_widget = pg.PlotWidget()
        plot_layout.addWidget(plot_widget)
        self.plot_window.setLayout(plot_layout)

        for start, duration, label in time_series:
            color = QtGui.QColor(0, 0, 255) if label == 'Video' else QtGui.QColor(128, 128, 128)
            bar_graph = pg.BarGraphItem(x=[start + duration / 2],
                                        height=[1],
                                        width=[duration],
                                        brush=color)
            plot_widget.addItem(bar_graph)

        plot_widget.setYRange(-0.5, 1.5)
        # plot_widget.setLabel('bottom', 'Time (s)')
        plot_widget.setLabel('bottom', 'Time (HH:MM:SS)')
        plot_widget.setTitle('Video and Blank Periods')

        # Convert time in seconds to HH:MM:SS format for x-axis
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02}:{minutes:02}:{seconds:02}"

        tick_values = [start for start, _, _ in time_series]
        tick_strings = [format_time(tick) for tick in tick_values]
        plot_widget.getAxis('bottom').setTicks([list(zip(tick_values, tick_strings))])
        
        self.plot_window.show()

    def open_video_editor(self):
        self.video_editor = VideoEditor(self)
        self.video_editor.exec()

    def handle_finished(self,time_series, video_path):
        self.plot_time_series(time_series)
        self.update_video(video_path)
        # print("处理完成:", video_path)


    # 新建一个窗口，选择视频文件夹
    def open_file_dialog(self):
        self.open_video_editor()



    def update_video(self, video_path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_number = 0

        # Get total frame count
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate total duration
        total_duration = total_frames / fps

        # Get current time in seconds
        current_time = frame_number / fps

        # Format the times
        current_time_str = self.format_time(current_time)
        total_duration_str = self.format_time(total_duration)

        self.video_time_label.setText(f"当前时间: {current_time_str} / 总时间: {total_duration_str}")
        self.display_frame(0)
    # def update_video(self, video_path):
    #     # self.cap = cv2.VideoCapture(r'C:\Users\user\Videos\test_hardware_encoder.mp4')
    #     # self.cap = cv2.VideoCapture(r'C:\Users\user\Documents\WeChat Files\wxid_mi05poeuk7a022\FileStorage\File\2024-09\xia-san-video-sample\umineko\LB11\PBOT0001.avi')
    #     self.cap = cv2.VideoCapture(r'G:\素材\9月30日.mp4')
    #     self.display_frame(0)


    def format_time(self, seconds):
        if seconds >= 3600:
            # Format as HH:MM:SS
            return f"{int(seconds // 3600):02}:{int((seconds % 3600) // 60):02}:{int(seconds % 60):02}"
        else:
            # Format as MM:SS
            return f"{int(seconds // 60):02}:{int(seconds % 60):02}"
        
    def display_frame(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            self.qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 缩放图像以适应 QLabel
            scaled_pixmap = QPixmap.fromImage(self.qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            # self.video_label.setPixmap(QPixmap.fromImage(self.qt_image))
            self.video_label.setPixmap(scaled_pixmap)
            self.video_label.setScaledContents(True)



    def jump_to_timestamp(self, index):
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.offset = self.timestamp_input.value()
            datetime_org = self.data.loc[index, 'datetime'] # 2018-08-27 21:19:23.880000
            # print(type(unixtime))
            datetime_str = datetime_org.strftime('%Y-%m-%d %H:%M:%S.%f')
            
            
            # Connect to the database
            # conn = sqlite3.connect('database.db')
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Query the database for the video information
            cursor.execute('''
                SELECT animal_tag, video_stt, video_stp, framerate, frame_count, video_id
                FROM videos
                WHERE ? BETWEEN video_stt AND video_stp
            ''', (datetime_str,))
            video_info = cursor.fetchone()

            if video_info is None:
                print("No video found for the specified timestamp.")
                return

            animal_tag, video_stt, video_stp, framerate, frame_count, video_id = video_info

            video_name = generate_filename(video_id)
            # Construct the video file path
            # video_path = f"{animal_tag}/{video_name}"
            animal_tag = animal_tag.replace('.csv', '')

            video_path = os.path.join(self.video_path, animal_tag, video_name)


            # Open the video
            self.cap = cv2.VideoCapture(video_path)

            # Convert video_stt to datetime object
            video_stt = datetime.strptime(video_stt, '%Y-%m-%d %H:%M:%S')
            
            # Calculate the timestamp offset from the video start time
            timestamp_offset = (datetime_org - video_stt).total_seconds()

            # Calculate the frame number to jump to
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(fps * (timestamp_offset + self.offset))

            # Get total frame count
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate total duration
            total_duration = total_frames / fps

            # Get current time in seconds
            current_time = frame_number / fps

            # Format the times
            current_time_str = self.format_time(current_time)
            total_duration_str = self.format_time(total_duration)

            self.video_time_label.setText(f"{current_time_str} / {total_duration_str}")

            # Display the frame
            self.display_frame(frame_number)

            conn.close()
        except ValueError:
            print("Please enter a valid timestamp.")
        except Exception as e:
            print(f"An error occurred: {e}")

    # def jump_to_timestamp(self, index):
    #     try:
    #         if self.cap is None:
    #             return
    #         # self.offset = float(self.timestamp_input.text())
    #         self.offset = self.timestamp_input.value()
    #         unixtime = self.data.loc[index, 'unixtime']
            
    #         # TODO 用指定时间戳去查数据库，得到对应视频名，然后打开视频根据时间戳减去视频开始时间，然后跳转到对应帧

    #         timestamp = unixtime - self.min_time
    #         # timestamp = float(self.timestamp_input.text())
    #         fps = self.cap.get(cv2.CAP_PROP_FPS)
    #         frame_number = int(fps * (timestamp + self.offset))

    #         # Get total frame count
    #         total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #         # Calculate total duration
    #         total_duration = total_frames / fps

    #         # Get current time in seconds
    #         current_time = frame_number / fps

    #         # Format the times
    #         current_time_str = self.format_time(current_time)
    #         total_duration_str = self.format_time(total_duration)

    #         self.video_time_label.setText(f"{current_time_str} / {total_duration_str}")
    #         # # Print current and total time
    #         # print(f"当前时间: {current_time:.2f} 秒 / 总时间: {total_duration:.2f} 秒")
            
    #         self.display_frame(frame_number)
    #     except ValueError:
    #         print("Please enter a valid timestamp.")       


    # def update_video(self, video_path):
        # print(video_path)
        # self.video_label.setPixmap(QPixmap(video_path))
        # self.video_label.setScaledContents(True)
        # self.video_label.setFixedSize(600, 400)



    '''
    ==================================================
    右上区域复选框: 列表
    - self.checkboxList 列表(QCheckBox)
    ==================================================
    '''

    # 创建右上角的选择模型和数据复选框
    def createModelSelectLabelArea(self):

        # 日历按钮
        self.calendar_btn = QPushButton('Calendar')
        self.calendar_btn.setStyleSheet(self.button_style)
        self.calendar_btn.clicked.connect(self.open_calendar)
        self.first_row_layout.addWidget(self.calendar_btn, alignment=Qt.AlignLeft)
        
        # 第一行布局,包含Select model标签和选择框，, alignment=Qt.AlignLeft
        # 创建模型组合框和标签
        modelComboBoxLabel, modelComboBox = self.createModelComboBox()
        # self.first_row1_layout = QHBoxLayout()
        # self.first_row1_layout.addWidget(modelComboBoxLabel, alignment=Qt.AlignLeft)
        # self.first_row1_layout.addWidget(modelComboBox, alignment=Qt.AlignLeft)
        self.first_row_layout.addWidget(modelComboBoxLabel, alignment=Qt.AlignLeft)
        self.first_row_layout.addWidget(modelComboBox, alignment=Qt.AlignLeft)
        self.refresh_btn = QPushButton('Refresh')
        self.refresh_btn.setStyleSheet(self.button_style)
        self.refresh_btn.clicked.connect(self.handleRefresh)

        # self.first_row1_layout.addWidget(self.refresh_btn, alignment=Qt.AlignLeft)
        # self.first_row1_layout.addStretch()  # 添加一个伸缩因子来填充剩余空间

        # 第二行布局
        # 创建原始数据组合框和标签
        RawDataComboBoxLabel, RawDatacomboBox = self.createRawDataComboBox()
        # self.first_row_layout.addLayout(RawDataComboBoxLabel, alignment=Qt.AlignLeft)
        # self.first_row_layout.addWidget(RawDataComboBoxLabel, alignment=Qt.AlignLeft)
        # self.first_row_layout.addWidget(RawDatacomboBox, alignment=Qt.AlignLeft)

        featureExtractBtn = self.createFeatureExtractButton()
        # self.first_row_layout.addWidget(featureExtractBtn, alignment=Qt.AlignRight)
        # self.second_row1_layout = QHBoxLayout()
        # self.first_row_layout.addLayout(self.second_row1_layout)
        # self.second_row1_layout.addWidget(RawDataComboBoxLabel, alignment=Qt.AlignLeft)
        # self.second_row1_layout.addWidget(RawDatacomboBox, alignment=Qt.AlignLeft)
        # self.second_row1_layout.addStretch()  # 添加一个伸缩因子来填充剩余空间

        # check_box布局
        self.checkbox_layout = QHBoxLayout()

        # 颜色展示
        self.color_layout = QHBoxLayout()

        self.second_row_layout.addLayout(self.checkbox_layout)
        self.second_row_layout.addLayout(self.color_layout)


        # TODO: 将一部分功能改到日历中
        # 第三行布局 Display data 按钮
        # self.third_row1_layout = QHBoxLayout()
        # featureExtractBtn = self.createFeatureExtractButton()
        # self.labelColorBtn = self.createToggleLabelColor()  # 单击可以让右下散点图显示已有标签

        # self.third_row1_layout.addWidget(featureExtractBtn, alignment=Qt.AlignRight)
        # self.third_row1_layout.addWidget(labelColorBtn, alignment=Qt.AlignRight)

        # self.nestend_layout.addLayout(self.first_row1_layout)
        # self.nestend_layout.addLayout(self.second_row1_layout)
        # self.nestend_layout.addLayout(self.checkbox_layout)
        # self.nestend_layout.addLayout(self.color_layout)
        # self.nestend_layout.addLayout(self.third_row1_layout)

        self.renderColumnList()
        # self.clear_color_layout()
        self.display_colors(self.label_colors)
    
    def handleRefresh(self):
        self.update_model_combobox()
        self.update_data_combobox()

    def open_calendar(self):
        self.calendar = DateTimeSelector(self)
        self.calendar.show()


    def display_colors(self, colors):
        # 创建水平布局并添加标签和颜色框
        for color_name, color_value in colors.items():
            layout = QHBoxLayout()

            label = QLabel(color_name + ":")
            layout.addWidget(label)

            color_frame = QFrame()
            color_frame.setFixedSize(20, 20)
            color_frame.setStyleSheet(f"background-color: {color_value};")
            layout.addWidget(color_frame)

            self.color_layout.addLayout(layout)
        self.color_layout.addStretch()

    def clear_color_layout(self):
        # 移除并删除所有布局项
        while self.color_layout.count() > 0:  # 改为0以清除所有
            item = self.color_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    widget = item.layout().takeAt(0).widget()
                    if widget:
                        widget.deleteLater()
                item.layout().deleteLater()
    
    def clear_color_layout_and_display(self, colors):
        # 移除并删除所有布局项
        while self.color_layout.count() > 0:  # 改为0以清除所有
            item = self.color_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    widget = item.layout().takeAt(0).widget()
                    if widget:
                        widget.deleteLater()
                item.layout().deleteLater()
                # 创建水平布局并添加标签和颜色框

        for color_name, color_value in colors.items():
            layout = QHBoxLayout()

            label = QLabel(color_name + ":")
            layout.addWidget(label)

            color_frame = QFrame()
            color_frame.setFixedSize(20, 20)
            color_frame.setStyleSheet(f"background-color: {color_value};")
            layout.addWidget(color_frame)

            self.color_layout.addLayout(layout)
        self.color_layout.addStretch()



    '''
    ==================================================
    右中区域复选框: 列表
    ==================================================
    '''

    def createSettingArea(self):
        self.charts_show_button_layout.addStretch()
        self.labelColorBtn = self.createToggleLabelColor()
        self.charts_show_button_layout.addWidget(self.labelColorBtn)
        # 第一行生成选框按钮
        # self.first_row2_layout = QHBoxLayout()
        addRegionBtn = QPushButton('Generate area')
        # 设置按钮样式
        addRegionBtn.setStyleSheet(self.button_style)
        # 设置按钮最小宽度
        # addRegionBtn.setFixedWidth(160)
        addRegionBtn.clicked.connect(self.handleAddRegion)
        self.charts_show_button_layout.addWidget(addRegionBtn)
        # self.first_row2_layout.addWidget(addRegionBtn, alignment=Qt.AlignLeft)
        # self.first_row2_layout.addStretch()

        # 第二行Threshold输入框
        # self.second_row2_layout = QHBoxLayout()

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("Input threshold")
        self.charts_show_button_layout.addWidget(self.input_box)

        # self.second_row2_layout.addWidget(QLabel("Threshold:"))
        # self.second_row2_layout.addWidget(self.input_box)

        # cache select region 缓存选定区域
        self.rightRegionRect = QRectF(0, 0, 1, 1)

        # 第三行两个按钮
        # self.third_row2_layout = QHBoxLayout()
        toLabelBtn = QPushButton('Find data')  # Save to label
        # 设置按钮样式
        toLabelBtn.setStyleSheet(self.button_style)
        toLabelBtn.clicked.connect(self.handleToLabel)

        clearEmptyRegionBtn = QPushButton('Clear data')
        # 设置按钮样式
        clearEmptyRegionBtn.setStyleSheet(self.button_style)
        clearEmptyRegionBtn.clicked.connect(self.handleClearEmptyRegion)

        self.charts_show_button_layout.addWidget(toLabelBtn)
        self.charts_show_button_layout.addWidget(clearEmptyRegionBtn)
        self.charts_show_button_layout.addStretch()


        # # 添加一个伸缩因子来创建间距
        # # self.third_row2_layout.addStretch(1)
        # self.third_row2_layout.addWidget(toLabelBtn, alignment=Qt.AlignLeft)
        # self.third_row2_layout.addStretch(1)  # Increase the stretch factor to create more space
        # self.third_row2_layout.addWidget(clearEmptyRegionBtn, alignment=Qt.AlignLeft)
        # self.third_row2_layout.addStretch(10)

        # self.row2_layout.addLayout(self.first_row2_layout)
        # self.row2_layout.addLayout(self.second_row2_layout)
        # self.row2_layout.addLayout(self.third_row2_layout)

    '''
    ==================================================
    顶部区域复选框: 列表
    - self.checkboxList 列表(QCheckBox)
    ==================================================
    '''

    # 创建顶部区域的方法
    def createTopArea(self):
        # 创建原始数据组合框和标签
        RawDataComboBoxLabel, RawDatacomboBox = self.createRawDataComboBox()
        # 将标签添加到顶部布局
        self.top_layout.addWidget(RawDataComboBoxLabel, alignment=Qt.AlignLeft)
        # 将组合框添加到顶部布局
        self.top_layout.addWidget(RawDatacomboBox, alignment=Qt.AlignLeft)

        # 创建模型组合框和标签
        modelComboBoxLabel, modelComboBox = self.createModelComboBox()
        # 将标签添加到顶部布局
        self.top_layout.addWidget(modelComboBoxLabel, alignment=Qt.AlignLeft)
        # 将组合框添加到顶部布局
        self.top_layout.addWidget(modelComboBox, alignment=Qt.AlignLeft)

        # 创建特征提取按钮
        featureExtractBtn = self.createFeatureExtractButton()
        # 将按钮添加到顶部布局
        self.top_layout.addWidget(featureExtractBtn, alignment=Qt.AlignLeft)

        featureColor = self.createToggleLabelColor()
        # 将按钮添加到顶部布局
        self.top_layout.addWidget(featureColor, alignment=Qt.AlignLeft)

        # createToggleLabelColor

        # 添加一个伸缩项以填充其余空间并保持左对齐
        self.top_layout.addStretch()

    # # 从row_data的csv创建原始数据组合框的方法
    # def createRawDataComboBox(self):
    #     # 创建标签
    #     RawDataComboBoxLabel = QLabel('Select data:')

    #     # 创建组合框
    #     RawDatacomboBox = QComboBox()
    #     # 获取原始数据文件夹路径
    #     raw_data_path = get_raw_data_folder()
    #     # 获取所有.csv文件路径
    #     rawdata_file_path_list = list(
    #         Path(os.path.join(self.cfg["project_path"], raw_data_path)).glob('*.csv'),
    #     )
    #     # 遍历路径列表
    #     for path in rawdata_file_path_list:
    #         # 将文件名添加到组合框
    #         RawDatacomboBox.addItem(str(path.name))
    #     # 保存组合框
    #     self.RawDatacomboBox = RawDatacomboBox

    #     # combbox change组合框改变时的处理
    #     # 打开第一个.csv文件
    #     self.get_data_from_csv(rawdata_file_path_list[0].name)
    #     self.RawDatacomboBox.currentTextChanged.connect(
    #         # 连接组合框文本改变事件到get_data_from_csv方法
    #         self.get_data_from_csv
    #     )
    #     # 返回标签和组合框
    #     return RawDataComboBoxLabel, RawDatacomboBox

    # 创建原始数据组合框的方法
    def createRawDataComboBox(self):
        # find data at here:C:\Users\dell\Desktop\aa-bbb-2024-04-28\unsupervised-datasets\allDataSet
        # 创建标签
        RawDataComboBoxLabel = QLabel('Select data:')

        # 创建组合框
        RawDatacomboBox = QComboBox()
        # 获取无监督数据集文件夹路径
        unsup_data_path = get_unsupervised_set_folder()
        # 获取所有.pkl文件路径
        rawdata_file_path_list = list(
            Path(os.path.join(self.cfg["project_path"], unsup_data_path)).glob('*.pkl'),
        )
        # 遍历路径列表
        for path in rawdata_file_path_list:
            # 将文件名添加到组合框
            RawDatacomboBox.addItem(str(path.name))
        # 保存组合框
        self.RawDatacomboBox = RawDatacomboBox

        # combbox change组合框改变时的处理
        # 改变不再打开pkl，在点击dataplay再加载数据
        # 打开第一个.pkl文件
        # self.get_data_from_pkl(rawdata_file_path_list[0].name)
        # self.RawDatacomboBox.currentTextChanged.connect(
        #     # 连接组合框文本改变事件到get_data_from_pkl方法
        #     self.get_data_from_pkl
        # )
        # 返回标签和组合框
        return RawDataComboBoxLabel, RawDatacomboBox

    def get_data_from_csv(self, filename):
        raw_data_path = get_unsupervised_set_folder()
        datapath = os.path.join(self.cfg["project_path"], raw_data_path, filename)
        with open(datapath, 'rb') as f:
            self.data = pickle.load(f)
        # self.data = pd.read_csv(datapath, low_memory=False)
        #
        # # 将 timestamp 列转换为 datetime 对象
        # self.data['datetime'] = pd.to_datetime(self.data['timestamp'])
        #
        # # 生成 unixtime 列（秒级时间戳）
        # self.data['unixtime'] = self.data['datetime'].astype('int64') // 10 ** 9

        # 添加时间戳列
        self.data['_timestamp'] = pd.to_datetime(self.data['datetime']).apply(lambda x: x.timestamp())

        # 复制经纬度并进行线性插值
        self.data['_latitude'] = self.data['latitude'].interpolate()
        self.data['_longitude'] = self.data['longitude'].interpolate()

        # 保留经纬度非空值
        # self.data = self.data.dropna(subset=['acc_x', 'acc_y', 'acc_z'])
        self.data['index'] = self.data.index  # Add an index column
        self.dataChanged.emit(self.data)
        return

    def update_model_combobox(self):
        # 清空原有的选项
        self.modelComboBox.clear()

        # 获取根对象配置
        config = self.root.config
        # 读取配置
        cfg = read_config(config)
        # 获取无监督模型文件夹路径
        unsup_model_path = get_unsup_model_folder(cfg)

        full_path = os.path.join(self.cfg["project_path"], unsup_model_path)
        model_path_list = grab_files_in_folder_deep(full_path, ext='*.pth')

        # # 获取所有.pth文件路径
        # model_path_list = grab_files_in_folder_deep(
        #     os.path.join(self.cfg["project_path"], unsup_model_path),
        #     ext='*.pth')
        # 保存模型路径列表
        self.model_path_list = model_path_list
        if model_path_list:
            # 遍历路径列表
            for path in model_path_list:
                self.modelComboBox.addItem(str(Path(path).name))

        # 更新监控的目录
        self.model_watcher.removePaths(self.model_watcher.directories())
        self.model_watcher.addPath(full_path)
    

    def update_data_combobox(self):
        # 清空原有的选项
        self.RawDatacomboBox.clear()

        # 获取无监督数据集文件夹路径
        unsup_data_path = get_unsupervised_set_folder()

        full_path = os.path.join(self.cfg["project_path"], unsup_data_path)
        rawdata_file_path_list = list(Path(full_path).glob('*.pkl'))

        # 获取所有.pkl文件路径
        # rawdata_file_path_list = list(
        #     Path(os.path.join(self.cfg["project_path"], unsup_data_path)).glob('*.pkl'),
        # )
        # 遍历路径列表
        for path in rawdata_file_path_list:
            self.RawDatacomboBox.addItem(str(Path(path).name))

         # 更新监控的目录
        self.data_watcher.removePaths(self.data_watcher.directories())
        self.data_watcher.addPath(full_path)

    # 创建模型组合框的方法
    def createModelComboBox(self):
        # 创建标签
        modelComboBoxLabel = QLabel('Select model:')

        # 创建组合框
        modelComboBox = QComboBox()
        # 从deepview.utils导入辅助函数
        # from deepview.utils import auxiliaryfunctions
        # Read file path for pose_config file. >> pass it on
        # 获取根对象配置
        config = self.root.config
        # 读取配置
        cfg = read_config(config)
        # 获取无监督模型文件夹路径
        unsup_model_path = get_unsup_model_folder(cfg)

        # 获取所有.pth文件路径
        model_path_list = grab_files_in_folder_deep(
            os.path.join(self.cfg["project_path"], unsup_model_path),
            ext='*.pth')
        # 保存模型路径列表
        self.model_path_list = model_path_list
        if model_path_list:
            # 遍历路径列表
            for path in model_path_list:
                # 将文件名添加到组合框
                modelComboBox.addItem(str(Path(path).name))
            # modelComboBox.currentIndexChanged.connect(self.handleModelComboBoxChange)

            self.modelComboBox = modelComboBox

            # if selection changed, run this code
            # 如果选择改变，运行这段代码
            model_name, data_length, column_names = \
                get_param_from_path(modelComboBox.currentText())  # 从路径获取模型参数
            # 保存模型路径
            self.model_path = modelComboBox.currentText()
            # 保存模型名称
            self.model_name = model_name
            # 保存数据长度
            self.data_length = data_length
            # 保存列名列表
            self.column_names = column_names
        modelComboBox.currentTextChanged.connect(
            # 连接组合框文本改变事件到get_model_param_from_path方法
            self.get_model_param_from_path
        )
        # 返回标签和组合框
        return modelComboBoxLabel, modelComboBox

    # 从路径获取模型参数的方法
    def get_model_param_from_path(self, model_path):
        # set model information according to model name
        # 根据模型名称设置模型信息
        if model_path:
            model_name, data_length, column_names = \
                get_param_from_path(model_path)
            # 保存模型路径
            self.model_path = model_path
            # 保存模型名称
            self.model_name = model_name
            # 保存数据长度
            self.data_length = data_length
            # 保存列名列表
            self.column_names = column_names
        return

    # 创建特征提取按钮的方法
    def createFeatureExtractButton(self):
        # 创建按钮
        featureExtractBtn = QPushButton('Data display')
        # 设置按钮样式
        featureExtractBtn.setStyleSheet(self.button_style)
        # 保存按钮
        self.featureExtractBtn = featureExtractBtn
        # 设置按钮宽度
        featureExtractBtn.setFixedWidth(160)
        # 设置按钮不可用
        featureExtractBtn.setEnabled(False)
        # 连接按钮点击事件到handleCompute方法
        # featureExtractBtn.clicked.connect(self.handleCompute)
        featureExtractBtn.clicked.connect(self.start_handle_compute)
        # 返回按钮
        return featureExtractBtn

    def createToggleLabelColor(self):
        # 创建按钮
        featureExtractBtn = QPushButton('Data Coloring')
        # 设置按钮样式
        featureExtractBtn.setStyleSheet(self.button_style)
        self.is_toggled = True
        # 设置按钮宽度
        featureExtractBtn.setFixedWidth(160)
        # 设置按钮不可用
        # featureExtractBtn.setEnabled(False)
        # 连接按钮点击事件到handleCompute方法
        featureExtractBtn.clicked.connect(self.toggleLabelColor)
        return featureExtractBtn

    # 处理计算的方法
    def handleCompute(self):
        # 打印开始训练
        print('start training...')
        # 设置训练状态为True
        self.isTraining = True
        # 更新按钮状态
        self.updateBtn()

        # 重新设置选框
        self.renderColumnList()

        # 获取combobox的内容
        self.data, self.dataChanged = get_data_from_pkl(self.RawDatacomboBox.currentText(), self.cfg, self.dataChanged)

        # 延时100毫秒调用handleComputeAsyn方法
        self.computeTimer.singleShot(100, self.handleComputeAsyn)

    # 异步处理计算的方法
    def handleComputeAsyn(self):
        metadatas = find_charts_data_columns(self.sensor_dict, self.column_names)
        self.backend.displayData(self.data, metadatas, self.label_colors)
        self.backend_map.displayMapData(self.data)

        # 初始化图表之后不用添加spacer了
        # self.right_layout.removeItem(self.spacer)

        # 渲染右侧图表（特征提取功能）
        self.renderRightPlot()  # feature extraction function here

        # 设置训练状态为False
        self.isTraining = False

        self.updateBtn()


    def handel_calendar_data(self, data):
        self.data = data
        self.dataChanged.emit(self.data)
        metadatas = find_charts_data_columns(self.sensor_dict, self.column_names)
        self.backend.displayData(self.data, metadatas, self.label_colors)
        self.backend_map.displayMapData(self.data)
        self.start_handle_compute()

    def start_handle_compute(self):
        # 打印开始训练
        print('start training...')
        # 设置训练状态为True
        self.isTraining = True
        # 更新按钮状态
        self.updateBtn()

        # 重新设置选框
        self.renderColumnList()

        self.handle_compute_thread = QThread()
        self.handle_compute_worker = HandleComputeWorker(self.root, self.data, self.RawDatacomboBox.currentText(), self.cfg, self.dataChanged, self.sensor_dict, self.column_names, self.data_length, self.model_path, self.model_name)
        self.handle_compute_worker.moveToThread(self.handle_compute_thread)
        self.handle_compute_thread.started.connect(self.handle_compute_worker.run)
        self.handle_compute_worker.finished.connect(self.handle_compute_finished)
        self.handle_compute_worker.stopped.connect(self.on_training_stopped)
        self.handle_compute_worker.dataChangedSignal.connect(self.compute_data_changed)
        self.handle_compute_worker.finished.connect(self.handle_compute_thread.quit)
        self.handle_compute_worker.finished.connect(self.handle_compute_worker.deleteLater)
        self.handle_compute_thread.finished.connect(self.handle_compute_thread.deleteLater)
        self.handle_compute_thread.start()

    def stop_training(self):
        self.handle_compute_worker.stop()


    def on_training_stopped(self):
        # TODO 绑定停止训练的方法
        # self.stop_button.setEnabled(False)
        self.isTraining = False
        self.updateBtn()
        print("Training was stopped.")

    def compute_data_changed(self, data):
        self.data = data
        self.min_time = self.data['unixtime'].min()
        metadatas = find_charts_data_columns(self.sensor_dict, self.column_names)
        self.backend.displayData(self.data, metadatas, self.label_colors)
        self.backend_map.displayMapData(self.data)

    def handle_compute_finished(self, data):
        (spots, start_indice, end_indice) = data
        # 设置训练状态为False
        self.isTraining = False
        self.updateBtn()
        # 清除中央绘图区域
        self.viewC.clear()
        # 创建一个散点图项
        scatterItem = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None))
        self.start_indice = start_indice
        self.end_indice = end_indice
        # 在散点图中绘制点
        scatterItem.addPoints(spots)
        self.scatterItem = scatterItem
        scatterItem.sigClicked.connect(self.handleScatterItemClick)
        self.viewC.addItem(scatterItem)
        return

    def handleScatterItemClick(self, scatterItem, points):
        if len(points) >= 1:
            index, start, end = points[0].data()

            # start为latent space的切片索引，index为原始索引（对应切片的开始索引）
            lat, lon = self.data.loc[start, 'latitude'], self.data.loc[start, 'longitude']

            if pd.isna(lat) or pd.isna(lon):
                print("Latitude or longitude is missing.")
                return

            # 点击散点图高亮地图散点
            self.backend_map.triggeLineMapHighlightDotByIndex(start, lat, lon)
            # 点击散点图高亮折线图散点
            self.backend.triggeLineChartHighlightDotByIndex(start)
            # 点击散点图高亮自己
            self.handle_highlight_scatter_dot_by_index(index, True)

        return

    # 更新按钮状态的方法
    def updateBtn(self):
        # enabled 启用按钮
        if self.isTraining:
            # 如果在训练，设置按钮不可用
            self.featureExtractBtn.setEnabled(False)
        else:
            # 如果不在训练，设置按钮可用
            self.featureExtractBtn.setEnabled(True)

    # 渲染列列表的方法
    def renderColumnList(self):
        # 清空 layout
        while self.checkbox_layout.count():
            item = self.checkbox_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        # 初始化复选框列表
        self.checkboxList = []
        # 遍历列名列表
        for column in self.all_sensor:
            # 创建复选框
            cb = QCheckBox(column)
            # 设置复选框为选中状态，如果在列名列表中
            cb.setChecked(column in self.column_names)
            # 将复选框添加到布局中
            self.checkbox_layout.addWidget(cb)
            # 将复选框添加到列表中
            self.checkboxList.append(cb)
            # 连接复选框状态改变事件到handleCheckBoxStateChange方法
            cb.stateChanged.connect(self.handleCheckBoxStateChange)

        # 添加一个伸缩项以填充剩余区域并保持复选框左对齐
        self.checkbox_layout.addStretch()

    # 处理复选框状态改变的方法
    def handleCheckBoxStateChange(self):
        # 创建新选择列列表
        newSelectColumn = []
        # 遍历列名列表
        for i, column in enumerate(self.all_sensor):
            # 如果复选框被选中
            if self.checkboxList[i].isChecked():
                # 添加列到新选择列列表
                newSelectColumn.append(column)
        # 打印选择列
        # self.selectColumn = newSelectColumn
        print('selectColumn: %s' % (newSelectColumn))
        # self.current_select_sensor_column = newSelectColumn

        metadata = find_charts_data_columns(self.sensor_dict, newSelectColumn)

        # 更新左下图表
        self.backend.handleComboxSelection(metadata)

    '''
    ==================================================
    bottom left area: plot左下区域: 图表
    - self.viewL GraphicsLayoutWidget
    - self.leftPlotList list(PlotItem)
    ==================================================
    '''

    # 创建左侧图表的方法
    def createLeftPlot(self):  # 创建左侧图表的方法
        viewL = QVBoxLayout()  # 创建一个垂直布局
        self.viewL = viewL  # 保存布局
        self.splitter = QSplitter(Qt.Vertical)  # 创建一个垂直分割器
        self.plot_widgets = [None] * len(self.column_names)  # 初始化图表小部件列表
        self.click_begin = True  # 初始化点击开始状态为True
        self.start_line = [None] * len(self.column_names)  # 初始化开始线列表
        self.end_line = [None] * len(self.column_names)  # 初始化结束线列表
        self.regions = []  # 初始化区域列表
        for _ in range(len(self.column_names)):  # 遍历列名列表
            self.regions.append([])  # 为每个列创建一个空的区域列表
        self.bottom_layout.addLayout(viewL, 2)  # 将布局添加到底部布局中

        self.resetLeftPlot()  # 重置左侧图表
        self.splitter.setSizes([100] * len(self.column_names))  # 设置分割器大小
        self.viewL.addWidget(self.splitter)  # 将分割器添加到布局中

    def resetLeftPlot(self):  # 重置左侧图表的方法
        # 重置小部件
        for i in range(len(self.column_names)):  # 遍历列名列表
            if self.plot_widgets[i] is not None:  # 如果图表小部件不为None
                self.splitter.removeWidget(self.plot_widgets[i])  # 从分割器中移除小部件
                self.plot_widgets[i].close()  # 关闭小部件
                self.plot_widgets[i] = None  # 设置小部件为None
        # 添加小部件
        for i, columns in enumerate(self.column_names):  # 遍历列名列表
            real_columns = self.sensor_dict[columns]  # 获取真实列名列表
            plot = pg.PlotWidget(title=columns, name=columns, axisItems={'bottom': pg.DateAxisItem()})  # 创建图表小部件
            for j, c in enumerate(real_columns):  # 遍历真实列名列表
                plot.plot(self.data['_timestamp'], self.data[c], pen=pg.mkPen(j))  # 绘制数据
            # plot.plot(self.data['datetime'], self.data[columns[0]], pen=pg.mkPen(i))
            plot.scene().sigMouseClicked.connect(self.mouse_clicked)  # 连接鼠标点击事件到mouse_clicked方法
            plot.scene().sigMouseMoved.connect(self.mouse_moved)  # 连接鼠标移动事件到mouse_moved方法
            self.plot_widgets[i] = plot  # 保存图表小部件
            self.splitter.addWidget(plot)  # 将图表小部件添加到分割器中

    def updateLeftPlotList(self):
        # 遍历每一列的名称
        for i, column in enumerate(self.column_names):
            # 显示每个绘图窗口
            self.plot_widgets[i].show()

    # def _to_idx(self, start_ts, end_ts):
    #     # 根据给定的时间戳范围筛选数据，并获取对应的索引
    #     selected_indices = self.data[(self.data['_timestamp'] >= start_ts)
    #                                  & (self.data['_timestamp'] <= end_ts)].index
    #     # 返回起始和结束索引
    #     return selected_indices.values[0], selected_indices.values[-1]

    # def _to_time(self, start_idx, end_idx):
    #     # 根据给定的索引范围获取起始和结束时间戳
    #     start_ts = self.data.loc[start_idx, '_timestamp']
    #     end_ts = self.data.loc[end_idx, '_timestamp']
    #     # 返回起始和结束时间戳
    #     return start_ts, end_ts

    # _timestamp于unixtime相同，改用unixtime
    def _to_idx(self, start_ts, end_ts):
        # 根据给定的时间戳范围筛选数据，并获取对应的索引
        selected_indices = self.data[(self.data['unixtime'] >= start_ts)
                                     & (self.data['unixtime'] <= end_ts)].index
        # 返回起始和结束索引
        return selected_indices.values[0], selected_indices.values[-1]

    def _to_time(self, start_idx, end_idx):
        # 根据给定的索引范围获取起始和结束时间戳
        start_ts = self.data.loc[start_idx, 'unixtime']
        end_ts = self.data.loc[end_idx, 'unixtime']
        # 返回起始和结束时间戳
        return start_ts, end_ts

    def _add_region(self, pos):
        if self.click_begin:
            # 如果是第一次点击，记录开始位置
            self.click_begin = False
            for i, plot in enumerate(self.plot_widgets):
                # 创建并添加起始和结束的垂直线
                self.start_line[i] = pg.InfiniteLine(pos.x(), angle=90, movable=False)
                self.end_line[i] = pg.InfiniteLine(pos.x(), angle=90, movable=False)
                plot.addItem(self.start_line[i])
                plot.addItem(self.end_line[i])
        else:
            # 如果是第二次点击，记录结束位置并创建区域
            self.click_begin = True
            for i, plot in enumerate(self.plot_widgets):
                # 移除起始和结束的垂直线
                plot.removeItem(self.start_line[i])
                plot.removeItem(self.end_line[i])

                # 创建一个线性区域并添加到绘图窗口
                region = pg.LinearRegionItem([self.start_line[i].value(), self.end_line[i].value()],
                                             brush=(0, 0, 255, 100))
                region.sigRegionChanged.connect(self._region_changed)

                self.start_line[i] = None
                self.end_line[i] = None

                plot.addItem(region)
                self.regions[i].append(region)
                # 获取选中的索引范围
                start_idx, end_idx = self._to_idx(int(region.getRegion()[0]), int(region.getRegion()[1]))
                print(f'Selected range: from index {start_idx} to index {end_idx}')

    def _region_changed(self, region):
        idx = 0
        # 找到当前改变的区域索引
        for reg_lst in self.regions:
            for i, reg in enumerate(reg_lst):
                if reg == region:
                    idx = i
                    break
        # 同步更新所有绘图窗口中的相应区域
        for reg_lst in self.regions:
            reg_lst[idx].setRegion(region.getRegion())

    def _del_region(self, pos):
        # 删除点击位置对应的区域
        for i, pwidget in enumerate(self.plot_widgets):
            for reg in self.regions[i]:
                if reg.getRegion()[0] < pos.x() and reg.getRegion()[1] > pos.x():
                    pwidget.removeItem(reg)
                    self.regions[i].remove(reg)

                    start_idx, end_idx = self._to_idx(int(reg.getRegion()[0]), int(reg.getRegion()[1]))
                    print(f'Delete region({start_idx}, {int(end_idx)})')
                    break

    def _edit_region(self, pos):
        set_val = None
        # 编辑点击位置对应的区域
        for i, _ in enumerate(self.regions):
            for reg in self.regions[i]:
                if reg.getRegion()[0] < pos.x() and reg.getRegion()[1] > pos.x():
                    if set_val is None:
                        # 弹出对话框选择标签
                        dialog = LabelOption(self.label_dict)
                        if dialog.exec() == QDialog.Accepted:
                            set_val = dialog.confirm_selection()
                        else:
                            set_val = None

                    # 设置区域的颜色和标签
                    reg.setBrush(self.checkColor(set_val))
                    reg.label = set_val

                    start_idx, end_idx = self._to_idx(int(reg.getRegion()[0]), int(reg.getRegion()[1]))
                    print(f'Edit region({start_idx}, {end_idx}) label: {set_val}')

    def mouse_clicked(self, event):
        if event.button() == Qt.LeftButton and hasattr(self, 'scatterItem'):
            pos = self.plot_widgets[0].plotItem.vb.mapToView(event.pos())
            # print(f'Clicked at {event.pos()} mapSceneToView {pos.x()},{pos.y()} mapToView {pos2.x()},{pos2.y()}')

            if self.mode == 'add':
                self._add_region(pos)
            elif self.mode == 'edit':
                self._edit_region(pos)
            elif self.mode == 'del':
                self._del_region(pos)

    def mouse_moved(self, event):
        pos = self.plot_widgets[0].plotItem.vb.mapSceneToView(event)
        if not self.click_begin:
            # 动态更新结束线的位置
            for line in self.end_line:
                line.setPos(pos.x())

    '''
    ==================================================
    bottom center area: result plot底部中心区域:结果图
    - self.viewC PlotWidget
    - self.selectRect QRect
    - self.lastChangePoint list(SpotItem)
    - self.lastMarkList list(LinearRegionItem)
    ==================================================
    '''

    def createCenterPlot(self):
        # 创建一个用于显示中央绘图区域的PlotWidget
        viewC = pg.PlotWidget()
        self.viewC = viewC
        # 禁用右键菜单
        self.viewC.setMenuEnabled(False)
        self.viewC.setBackground('w')
        # 将该PlotWidget添加到底部布局中
        self.row3_layout.addWidget(viewC, 2)

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

    # 更新右侧散点图的颜色
    def updateRightPlotColor(self):
        spots = []
        for spot in self.scatterItem.points():
            pos = spot.pos()
            i, start, end = spot.data()
            # 如果first=False，使用已有的标签
            # 如果first=True，使用手动标签
            color = self.checkColor(self.data.loc[start, 'label'], first=True)
            spot = {'pos': (pos.x(), pos.y()), 'data': (i, start, end),
                    'brush': pg.mkBrush(color)}
            spots.append(spot)

        # 更新散点数据
        for reg in self.regions[0]:
            # 获取区域的起始和结束索引
            idx_begin, idx_end = self._to_idx(int(reg.getRegion()[0]), int(reg.getRegion()[1]))
            for spot in spots:
                if idx_begin < spot['data'][1] and idx_end > spot['data'][2]:
                    spot['brush'] = reg.brush

        self.scatterItem.setData(spots=spots)

    def renderRightPlot(self):
        # 清除中央绘图区域
        self.viewC.clear()

        # 创建一个散点图项
        scatterItem = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None))

        new_column_names = find_data_columns(self.sensor_dict, self.column_names)

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

        self.start_indice = start_indice
        self.end_indice = end_indice
        # 在散点图中绘制点
        scatterItem.addPoints(spots)
        self.scatterItem = scatterItem

        self.viewC.addItem(scatterItem)
        return

    def find_i_by_indice(self, indice, start_indice, end_indice):
        '''
        map和sensor data都用原始索引，latent space用切片索引，这里从原始索引查找切片
        indice为原始索引
        start indice为切片索引
        '''
        for i in range(len(start_indice)):
            if start_indice[i] <= indice < end_indice[i]:
                return i
        return None  # 如果没有找到合适的范围

    # TODO 点击过快可能报错    self._plot.updateSpots(self._data.reshape(1)) AttributeError: 'NoneType' object has no attribute 'updateSpots'
    def handle_highlight_scatter_dot_by_index(self, index, useRawIndex = False):
        self.jump_to_timestamp(index)
        indice = index
        if not useRawIndex:
            indice = self.find_i_by_indice(index, self.start_indice, self.end_indice)
        if self.last_modified_points:
            for p, original_size, original_brush in self.last_modified_points:
                p.setSize(original_size)
                p.setBrush(original_brush)
            
        self.last_modified_points = []  # Clear the list
        if indice is not None:
            for spot in self.scatterItem.points():
                i, start, end = spot.data()
                if i == indice:
                    # Save current properties
                    original_size = spot.size()
                    original_brush = spot.brush()
                    self.last_modified_points.append((spot, original_size, original_brush))
                    spot.setSize(15)
                    spot.setBrush(pg.mkBrush(255, 0, 0, 255))

        



    # 显示原始标签在右下散点图上
    def toggleLabelColor(self):

        spots = []
        for spot in self.scatterItem.points():
            pos = spot.pos()
            i, start, end = spot.data()
            # spots.append({
            # 'pos': (pos.x(), pos.y()),
            # 'data': (i, start, end),
            # 'brush': pg.mkBrush(color)
            # })
            if self.data.loc[start, 'label_flag'] == 0:
                original_brush = spot.brush()  # 读取原有颜色
                color = original_brush.color()  # 默认使用原有颜色
                # color = self.checkColor(self.data.loc[start, 'label'], first=True)  # 相同背景色
            else:
                if self.is_toggled:
                    color = self.checkColor(self.data.loc[start, 'label'], first=False)
                else:
                    original_brush = spot.brush()  # 读取原有颜色
                    color = original_brush.color()  # 默认使用原有颜色
                    # color = self.checkColor(self.data.loc[start, 'label'], first=True)  # 相同背景色
            spot = {'pos': (pos.x(), pos.y()), 'data': (i, start, end),
                    'brush': pg.mkBrush(color)}
            spots.append(spot)
        # Toggle the flag
        self.is_toggled = not self.is_toggled

        # # 更新散点数据
        # for reg in self.regions[0]:
        #     # 获取区域的起始和结束索引
        #     idx_begin, idx_end = self._to_idx(int(reg.getRegion()[0]),
        #                                       int(reg.getRegion()[1]))
        #     for spot in spots:
        #         if idx_begin < spot['data'][1] and idx_end > spot['data'][2]:
        #             spot['brush'] = reg.brush

        self.scatterItem.setData(spots=spots)

        return

    def select_random_continuous_seconds(self, num_samples=100, points_per_second=90):
        # 随机选择连续的秒数数据段
        selected_dfs = []
        start_indice = []
        end_indice = []

        while len(selected_dfs) < num_samples:
            start_idx = np.random.randint(0, len(self.data) - points_per_second)
            end_idx = start_idx + points_per_second - 1
            selected_range = self.data.iloc[start_idx:end_idx + 1]

            if not selected_range[['acc_x', 'acc_y', 'acc_z']].isna().any().any():
                selected_dfs.append(selected_range)  # 从start_idx到end_idx的数据段
                start_indice.append(start_idx)
                end_indice.append(end_idx)

        return start_indice, end_indice, selected_dfs

    '''
    ==================================================
    bottom right area: setting panel
    - self.settingPannel QVBoxLayout
    - self.currentLabel str
    - self.maxColumn int
    - self.maxRow int
    ==================================================
    '''

    # 创建右侧设置面板
    def createRightSettingPannel(self):
        settingPannel = QVBoxLayout()
        self.settingPannel = settingPannel
        self.bottom_layout.addLayout(self.settingPannel)

        self.settingPannel.setAlignment(Qt.AlignTop)

        self.createLabelButton()
        self.createRegionBtn()
        self.createSaveButton()

    # 创建保存按钮
    def createSaveButton(self):
        saveButton = QPushButton('Save')
        saveButton.clicked.connect(self.handleSaveButton)
        self.settingPannel.addWidget(saveButton)



    # def getSelectedAreaToSave(self, area_data):
    #     # print(areaData)
    #     try:
    #         area_data = json.loads(area_data)  # 解析 JSON 字符串
    #         # print("Parsed data:", areaData)
    #     except json.JSONDecodeError as e:
    #         print("Failed to decode JSON:", e)
    #         return
    #     for reg in area_data:
    #         name = reg[0].get("name")
    #         first_timestamp = reg[0].get("timestamp", {}).get("start")
    #         second_timestamp = reg[0].get("timestamp", {}).get("end")
    #         self.data.loc[(self.data['unixtime'] >= int(first_timestamp)) & (
    #                 self.data['unixtime'] <= int(second_timestamp)), 'label'] = name
    #     self.handleSaveButton()

    def getSelectedAreaToSave(self, area_data):
        print("Saving CSV in the background.")
        combo_box_text = self.RawDatacomboBox.currentText()
        save_task = SaveCsvTask(area_data, self.data, self.cfg, combo_box_text, 0)
        save_task.signals.save_csv_finished.connect(self.on_save_finished)
        self.save_csv_thread_pool.start(save_task)

    def on_save_finished(self, new_path):
        QMessageBox.information(None, "保存CSV", f"文件已保存于 {new_path}", QMessageBox.Ok)
    
    def getSelectedAreaToSaveTimer(self, area_data):
        print("Saving CSV in the background.")
        combo_box_text = self.RawDatacomboBox.currentText()
        save_task = SaveCsvTask(area_data, self.data, self.cfg, combo_box_text, 1)
        self.save_csv_thread_pool.start(save_task)

    # 处理保存按钮点击事件
    def handleSaveButton(self):
        # for reg in self.regions[0]:
        #     if hasattr(reg, 'label') and reg.label:
        #         regionRange = reg.getRegion()
        #         self.data.loc[(self.data['_timestamp'] >= int(regionRange[0])) & (self.data['_timestamp'] <= int(regionRange[1])), 'label'] = reg.label

        os.makedirs(os.path.join(self.cfg["project_path"], "edit-data", ), exist_ok=True)
        edit_data_path = os.path.join(self.cfg["project_path"], "edit-data", self.RawDatacomboBox.currentText())
        # edit_data_path = os.path.join(self.cfg["project_path"], "edit-data", self.RawDatacomboBox.currentText().replace(".pkl", ".csv"))
        try:  # 如果文件存在就新建
            if os.path.exists(edit_data_path):
                for num in range(1, 100, 1):
                    firstname = edit_data_path.split('Hz')[0]
                    new_path = firstname + '_' + str(num) + '.pkl'
                    if not os.path.exists(new_path):
                        self.data.to_csv(new_path)
                        break
            else:
                new_path = edit_data_path
                self.data.to_csv(edit_data_path)
        except:
            print('save data error!')
        else:
            print(f'文件已经保存在{new_path}')

    # 创建标签按钮
    def createLabelButton(self):
        self.add_mode = QPushButton("Label Add Mode", self)
        self.add_mode.clicked.connect(partial(self._change_mode, "add"))
        self.edit_mode = QPushButton("Label Edit Mode", self)
        self.edit_mode.clicked.connect(partial(self._change_mode, "edit"))
        self.del_mode = QPushButton("Label Delete Mode", self)
        self.del_mode.clicked.connect(partial(self._change_mode, "del"))
        self.refresh = QPushButton("Refresh Spots", self)
        self.refresh.clicked.connect(self.updateRightPlotColor)
        self.settingPannel.addWidget(self.add_mode)
        self.settingPannel.addWidget(self.edit_mode)
        self.settingPannel.addWidget(self.del_mode)
        self.settingPannel.addWidget(self.refresh)

        # Add horizontal line 添加水平线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.settingPannel.addWidget(line)

    # 改变模式
    def _change_mode(self, mode: str):
        print(f'Change mode to "{mode}"')
        self.mode = mode

    # 创建区域按钮
    def createRegionBtn(self):
        addRegionBtn = QPushButton('Add region')
        addRegionBtn.clicked.connect(self.handleAddRegion)

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("Enter threshold")

        toLabelBtn = QPushButton('Reflect to Data')  # Save to label
        toLabelBtn.clicked.connect(self.handleToLabel)
        self.settingPannel.addWidget(addRegionBtn)
        self.settingPannel.addWidget(self.input_box)
        self.settingPannel.addWidget(toLabelBtn)

        # cache select region 缓存选定区域
        self.rightRegionRect = QRectF(0, 0, 1, 1)

        # Add horizontal line 添加水平线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.settingPannel.addWidget(line)

        # Clear empty region 清除空区域
        clearEmptyRegionBtn = QPushButton('Clear Empty Region')
        clearEmptyRegionBtn.clicked.connect(self.handleClearEmptyRegion)
        self.settingPannel.addWidget(clearEmptyRegionBtn)

    def handleAddRegion(self):
        if hasattr(self, 'rightRegionRoi'):
            return

        rect = self.viewC.viewRect()
        w = rect.width()
        h = rect.height()
        x = rect.x()
        y = rect.y()

        # create ROI
        roi = pg.ROI([x + w * 0.45, y + h * 0.45], [w * 0.1, h * 0.1])
        # 上
        roi.addScaleHandle([0.5, 1], [0.5, 0])
        # 右
        roi.addScaleHandle([1, 0.5], [0, 0.5])
        # 下
        roi.addScaleHandle([0.5, 0], [0.5, 1])
        # 左
        roi.addScaleHandle([0, 0.5], [1, 0.5])
        # 右下
        roi.addScaleHandle([1, 0], [0, 1])

        self.viewC.addItem(roi)

        self.rightRegionRoi = roi

        # roi.sigRegionChanged.connect(self.handleROIChange)
        # roi.sigRegionChangeFinished.connect(self.handleROIChangeFinished)
        # self.handleROIChange(roi)
        # self.handleROIChangeFinished(roi)

    # 处理反射到标签的方法
    def handleToLabel(self):
        if not hasattr(self, 'rightRegionRoi'):  # 如果没有右侧区域ROI，提示用户先添加区域
            print('Add region first.')
            return

        pos: pg.Point = self.rightRegionRoi.pos()
        size: pg.Point = self.rightRegionRoi.size()

        self.rightRegionRect.setRect(pos.x(), pos.y(), size.x(), size.y())
        points = self.scatterItem.pointsAt(self.rightRegionRect)

        # 是否需要合并区间
        rectangles = []
        for p in points:
            index, start, end = p.data()
            startT, endT = self._to_time(start, end)
            rectangles.append((startT, endT))
        # combine rectangles 合并矩形，数据为开始结束时间
        if self.input_box.text() == "":
            combined_rectangles = combine_rectangles(rectangles, float(30))  # set default value
        else:
            combined_rectangles = combine_rectangles(rectangles, float(self.input_box.text()))

        # 传递combined_rectangles到backend
        markData = []
        for startT, endT in combined_rectangles:
            # print(startT, endT)
            start_id, end_id = self._to_idx(startT, endT)
            start_timestamp = self.data.loc[start_id, 'timestamp']
            end_timestamp = self.data.loc[end_id, 'timestamp']

            # 创建markData
            start_Area = {
                'name': 'data',
                'xAxis': start_timestamp,
                'itemStyle': {
                    'color': 'rgba(0, 0, 255, 0.39)'
                }
            }
            end_Area = {
                'xAxis': end_timestamp,
            }
            newArray = [start_Area, end_Area]

            markData.append(newArray)
        # print(markData)
        # 将 markData 转换为 JSON 字符串
        mark_data = json.dumps(markData)
        # 传递markData到backend
        self.backend.setMarkData(mark_data)

    def handleClearEmptyRegion(self):
        # 绑定html的Clear
        self.backend.clearMarkData()


def combine_rectangles(rectangles, threshold_seconds=100):
    if not rectangles:
        return []

    # 将矩形按开始时间排序
    rectangles.sort(key=lambda x: x[0])

    combined_rectangles = []
    current_start, current_end = rectangles[0]

    for start, end in rectangles[1:]:
        # 如果当前时间段与下一个时间段间隔小于阈值
        if (start - current_end) <= threshold_seconds:
            # 合并时间段
            current_end = max(current_end, end)
        else:
            # 否则，将当前时间段加入结果列表，并更新当前时间段
            combined_rectangles.append((current_start, current_end))
            current_start, current_end = start, end

    # 添加最后一个时间段
    combined_rectangles.append((current_start, current_end))

    return combined_rectangles


def find_charts_data_columns(sensor_dict, column_names):
    # new_column_names = []
    metadatas = []
    for column_name in column_names:
        # real_names = sensor_dict[column_name]  # 获取每个列名对应的实际列名
        # new_column_names.extend(real_names) # 将实际列名添加到新的列名列表中
        if column_name.upper() == "GPS":
            real_names = ['GPS_velocity', 'GPS_bearing']
        else:
            real_names = sensor_dict[column_name]  # 获取每个列名对应的实际列名
        # 创建元数据信息
        metadata = {
            "name": column_name,
            "xAxisName": "timestamp",
            "yAxisName": "Y Axis 1",
            "series": real_names
        }
        metadatas.append(metadata)
    return metadatas
