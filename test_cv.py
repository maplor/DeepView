import sys
import cv2
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QLineEdit, QHBoxLayout
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 800, 600)

        # Video capture
        # self.cap = cv2.VideoCapture('video.mp4')
        self.cap = cv2.VideoCapture(r'C:\Users\user\Documents\WeChat Files\wxid_mi05poeuk7a022\FileStorage\File\2024-09\xia-san-video-sample\umineko\LB11\PBOT0001.avi')

        # Offset for manual alignment
        self.offset = 0.0

        # UI elements
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.timestamp_input = QLineEdit(self)
        self.timestamp_input.setPlaceholderText("Enter timestamp in seconds")

        self.jump_button = QPushButton("Jump to Timestamp", self)
        self.jump_button.clicked.connect(self.jump_to_timestamp)

        self.increase_button = QPushButton("Increase +0.1s", self)
        self.increase_button.clicked.connect(self.increase_offset)

        self.decrease_button = QPushButton("Decrease -0.1s", self)
        self.decrease_button.clicked.connect(self.decrease_offset)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.timestamp_input)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.decrease_button)
        button_layout.addWidget(self.increase_button)
        button_layout.addWidget(self.jump_button)

        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Show first frame
        self.display_frame(0)

    def display_frame(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def jump_to_timestamp(self):
        try:
            timestamp = float(self.timestamp_input.text())
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(fps * (timestamp + self.offset))
            self.display_frame(frame_number)
        except ValueError:
            print("Please enter a valid timestamp.")

    def increase_offset(self):
        self.offset += 0.1

    def decrease_offset(self):
        self.offset = max(0, self.offset - 0.1)

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec())