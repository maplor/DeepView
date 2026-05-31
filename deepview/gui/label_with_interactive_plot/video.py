import os
import logging

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


logger = logging.getLogger(__name__)


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
                    logger.warning("Cannot read video %s", video_path)
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
            logger.warning("No available output frames")
            return

        height, width, _ = output_frames[0].shape
        output_path = os.path.join(self.output_folder, 'output.mp4')
        output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (width, height))
        # output_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (width, height))

        for frame in output_frames:
            output_video.write(frame)

        output_video.release()

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
        #     return

    def process_videos(self):



        start_times = self.start_time_input.text().split(',')
        self.test_times(start_times)
        if len(start_times) != len(self.video_paths):
            logger.warning("Number of start times does not match the number of videos")
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