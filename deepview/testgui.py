import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QPushButton, QWidget, QFileDialog, QApplication, QScrollArea, \
    QGridLayout
import folium



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPS Trajectories Plotter")
        self.setGeometry(100, 100, 1200, 800)  # Set window size

        self.layout = QVBoxLayout()

        self.select_files_button = QPushButton("Select CSV Files")
        self.select_files_button.clicked.connect(self.select_files)
        self.layout.addWidget(self.select_files_button)

        self.scroll_area = QScrollArea()
        self.scroll_area_widget = QWidget()
        self.scroll_area_layout = QGridLayout()
        self.scroll_area_widget.setLayout(self.scroll_area_layout)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.scroll_area.setWidgetResizable(True)

        self.layout.addWidget(self.scroll_area)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def select_files(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("CSV Files (*.csv)")
        if file_dialog.exec():
            self.selected_files = file_dialog.selectedFiles()
            self.plot_trajectories()

    def plot_trajectories(self):
        row, col = 0, 0
        for file in self.selected_files:
            try:
                tmpdf = pd.read_csv(file)
                df = tmpdf[['timestamp', 'latitude', 'longitude']].copy()
                df = df.dropna()
                latitudes = df['latitude']
                longitudes = df['longitude']
                timestamps = pd.to_datetime(df['timestamp'])

                # Validate the data
                if latitudes.isnull().any() or longitudes.isnull().any() or timestamps.isnull().any():
                    print(f"Data in {file} contains null values. Please check the CSV file.")
                    continue

                # Create a folium map centered around the average latitude and longitude
                folium_map = folium.Map(location=[latitudes.mean(), longitudes.mean()], zoom_start=13)

                # Add the GPS trajectory to the map
                coordinates = list(zip(latitudes, longitudes))
                folium.PolyLine(locations=coordinates, color='blue', weight=2.5).add_to(folium_map)

                # Save the map to an HTML file
                output_file = os.path.splitext(file)[0] + '_trajectory.html'
                folium_map.save(output_file)

                # Load the HTML file into a QWebEngineView
                web_view = QWebEngineView()
                web_view.setFixedSize(400, 300)
                web_view.setUrl(f'file:///{os.path.abspath(output_file)}')

                # Add the QWebEngineView to the grid layout
                self.scroll_area_layout.addWidget(web_view, row, col)
                col += 1
                if col >= 3:
                    col = 0
                    row += 1

                print(f"Trajectory map saved to {output_file}")

            except Exception as e:
                print(f"An error occurred while processing {file}: {e}")

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
