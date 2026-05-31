import sqlite3
from datetime import datetime, timedelta

import pandas as pd
from PySide6.QtCore import QDate, QRectF, Qt, QTime, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QTextCharFormat
from PySide6.QtWidgets import (
    QCalendarWidget,
    QLabel,
    QPushButton,
    QTextEdit,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)


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
        # 重新设置预定义的时间段
        self.update()

    def parse_time_segments(self):
        segments = []
        for start_str, end_str in self.video_time_list: # PySide6.QtCore.QTime(7, 55, 24, 0) PySide6.QtCore.QTime(7, 56, 24, 0)
            start_str = start_str.split()[1]
            end_str = end_str.split()[1]
            start_time = QTime.fromString(start_str, "HH:mm:ss")
            end_time = QTime.fromString(end_str, "HH:mm:ss")
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
        # Convert label_flag to integer if it exists
        if 'label_flag' in df.columns:
            df['label_flag'] = df['label_flag'].fillna(0).astype(int)
        df['index'] = df.index

        self.main_window.handel_calendar_data(df)



    def date_changed(self):

        date = self.calendar.selectedDate()
        # conn = sqlite3.connect('database.db')
        conn = sqlite3.connect(self.main_window.db_path)
        cursor = conn.cursor()
        start_of_day = datetime.strptime(date.toString('yyyy-MM-dd'), '%Y-%m-%d')
        end_of_day = start_of_day + timedelta(days=1)

        # 查询特定日期的数据
        cursor.execute('''
        SELECT video_stt, video_stp FROM videos
        WHERE video_stt >= ? AND video_stt < ?
        ''', (start_of_day, end_of_day))

        # 获取查询结果
        results = cursor.fetchall()

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

    def closeEvent(self, event):
        # 在关闭前执行任何清理操作
        super().closeEvent(event)