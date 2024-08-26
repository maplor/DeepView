
from deepview.gui.components import (
    DefaultTab,
    _create_grid_layout,
    _create_label_widget,
)


from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QWidget,
    QFileDialog,
    QScrollArea,
    QGridLayout
)
from PySide6.QtGui import QPixmap

import os
import pickle

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import contextily as ctx


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



    def _generate_layout_attributes(self, layout):
        self.select_files_button = QPushButton("Select csv Files")
        self.select_files_button.clicked.connect(self.select_files)
        self.select_files_button1 = QPushButton("Select pkl Files (slower, upsampled)")
        self.select_files_button1.clicked.connect(self.select_files)
        layout.addWidget(self.select_files_button, 0, 0)
        layout.addWidget(self.select_files_button1, 1, 0)

        return

    def _generate_layout_attributes_dataset(self, layout):
        self.scroll_area = QScrollArea()
        self.scroll_area_widget = QWidget()
        self.scroll_area_layout = QGridLayout()
        self.scroll_area_widget.setLayout(self.scroll_area_layout)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area, 2, 0)
        return

    def select_files(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Processed raw data Files (*.pkl *.csv)")

        # Execute the file dialog
        if file_dialog.exec():
            self.selected_files = file_dialog.selectedFiles()
            self.plot_trajectories()

    def plot_trajectories(self):

        row, col = 0, 0
        for file in self.selected_files:
            img_file = os.path.splitext(file)[0] + '_map.png'
            if os.path.exists(img_file):
                pixmap = QPixmap(img_file)
                # Create a QLabel widget to display the image
                label = QLabel()
                label.setPixmap(pixmap)
                self.scroll_area_layout.addWidget(label, row, col)
                col += 1
                if col >= 3:
                    col = 0
                    row += 1
            else:
                try:
                    if file.endswith('.pkl'):
                        with open(file, 'rb') as f:
                            tmpdf = pickle.load(f)
                    elif file.endswith('.csv'):
                        tmpdf = pd.read_csv(file, low_memory=False)
                        # print('1')
                    else:
                        print('Cannot load GPS data, please check file type, should be csv or pkl files.')
                        tmpdf = []
                    # print('2')
                    df = tmpdf[['latitude', 'longitude']].copy()
                    df = df.dropna()
                    latitudes = df['latitude']
                    longitudes = df['longitude']
                    # timestamps = df['datetime']

                    # Project the GeoDataFrame to Web Mercator (EPSG:3857) for compatibility with contextily basemaps
                    gdf = gpd.GeoDataFrame(
                        df, geometry=gpd.points_from_xy(df.longitude, df.latitude),
                        crs="EPSG:3857"
                    )
                    # Create a plot
                    fig, ax = plt.subplots()
                    gdf.plot(ax=ax, marker='o', color='red', markersize=50)

                    # Set plot title and labels
                    ax.set_title('GPS Coordinates on Map')
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')

                    # Add a basemap
                    ctx.add_basemap(ax, crs=gdf.crs.to_string())

                    plt.savefig(img_file)

                    # Validate the data
                    if latitudes.isnull().any() or longitudes.isnull().any():
                        print(f"Data in {file} contains null values. Please check the pkl file.")
                        continue

                    # Add the QWebEngineView to the grid layout
                    # Load the image
                    pixmap = QPixmap(img_file)
                    # Create a QLabel widget to display the image
                    label = QLabel()
                    label.setPixmap(pixmap)

                    self.scroll_area_layout.addWidget(label, row, col)
                    # self.scroll_area_layout.addWidget(web_view, row, col)
                    col += 1
                    if col >= 3:
                        col = 0
                        row += 1

                    # print(f"Trajectory map saved to {output_file}")

                except Exception as e:
                    print(f"An error occurred while processing {file}: {e}")
