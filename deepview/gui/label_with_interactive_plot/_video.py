import os
import sqlite3
from datetime import datetime

import cv2
import pyqtgraph as pg
from PySide6 import QtGui
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDoubleSpinBox, QLabel, QHBoxLayout, QVBoxLayout, QWidget

from deepview.gui.label_with_interactive_plot.utils import generate_filename
from deepview.gui.label_with_interactive_plot.video import VideoEditor
from deepview.gui.label_with_interactive_plot.widgets.time_selector import ClickableLabel


class VideoControlsMixin:
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