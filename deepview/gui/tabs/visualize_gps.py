from PySide6.QtWebEngineWidgets import QWebEngineView

from deepview.gui.components import (
    DefaultTab,
    _create_grid_layout,
    _create_label_widget,
)

# from deepview.utils.auxiliaryfunctions import (
#     read_config,
#     get_unsupervised_set_folder
# )
from PySide6.QtWidgets import (
    # QVBoxLayout,
    QPushButton,
    QWidget,
    QFileDialog,
    QScrollArea,
    QGridLayout
)

import folium
import os
import pickle
# import pandas as pd

class GPSDisplayer(DefaultTab):
    # # 定义进度信号
    # progress_update = Signal(int)

    def __init__(self, root, parent, h1_description):
        super(GPSDisplayer, self).__init__(root, parent, h1_description)

        self.root = root

        self.dataset_attributes_dataset = _create_grid_layout(margins=(20, 0, 0, 0))

        self.main_layout.addWidget(_create_label_widget("Select sensor data files",
                                                        "font:bold"))
        self.layout_attributes = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.main_layout.addWidget(_create_label_widget(""))  # dummy label

        self.main_layout.addWidget(_create_label_widget("Display maps",
                                                        "font:bold"))
        self.dataset_attributes_dataset = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes_dataset(self.dataset_attributes_dataset)
        self.main_layout.addLayout(self.dataset_attributes_dataset)

        # #------------------------
        # self.select_files_button = QPushButton("Select CSV Files")
        # self.select_files_button.clicked.connect(self.select_files)
        # self.main_layout.addWidget(self.select_files_button)
        #
        # self.scroll_area = QScrollArea()
        # self.scroll_area_widget = QWidget()
        # self.scroll_area_layout = QGridLayout()
        # self.scroll_area_widget.setLayout(self.scroll_area_layout)
        # self.scroll_area.setWidget(self.scroll_area_widget)
        # self.scroll_area.setWidgetResizable(True)
        # self.main_layout.addWidget(self.scroll_area)
        # # self.main_layout.addWidget(self.selectfiles)
        # self.main_layout.addLayout(self.dataset_attributes_dataset)

    def _generate_layout_attributes(self, layout):
        self.select_files_button = QPushButton("Select pkl Files")
        self.select_files_button.clicked.connect(self.select_files)
        layout.addWidget(self.select_files_button, 0, 0)

        return

    def _generate_layout_attributes_dataset(self, layout):
        self.scroll_area = QScrollArea()
        self.scroll_area_widget = QWidget()
        self.scroll_area_layout = QGridLayout()
        self.scroll_area_widget.setLayout(self.scroll_area_layout)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area, 1, 0)
        # self.main_layout.addWidget(self.selectfiles)
        # self.main_layout.addLayout(self.dataset_attributes_dataset)
        return

    def select_files(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Processed raw data Files (*.pkl)")
        if file_dialog.exec():
            self.selected_files = file_dialog.selectedFiles()
            self.plot_trajectories()

    def plot_trajectories(self):

        row, col = 0, 0
        for file in self.selected_files:
            try:
                with open(file, 'rb') as f:
                    tmpdf = pickle.load(f)
                df = tmpdf[['datetime', 'latitude', 'longitude']].copy()
                df = df.dropna()
                latitudes = df['latitude']
                longitudes = df['longitude']
                timestamps = df['datetime']

                # Validate the data
                if latitudes.isnull().any() or longitudes.isnull().any() or timestamps.isnull().any():
                    print(f"Data in {file} contains null values. Please check the pkl file.")
                    continue

                # Create a folium map centered around the average latitude and longitude
                folium_map = folium.Map(location=[latitudes.mean(), longitudes.mean()], zoom_start=13)

                # Add the GPS trajectory to the map
                coordinates = list(zip(latitudes, longitudes))
                folium.PolyLine(locations=coordinates, color='blue', weight=2.5).add_to(folium_map)

                # Save the map to an HTML file
                output_file = os.path.splitext(file)[0] + '_map.html'
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
