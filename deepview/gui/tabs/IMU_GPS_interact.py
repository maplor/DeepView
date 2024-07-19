'''
1.选取一个文件（单天数据，pkl file），左边获得IMU数据，右边获得GPS数据
2.点击左边或者右边的某一个点，获得对面点的位置
'''



from PySide6.QtWidgets import (
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QWidget,
)

from deepview.gui.components import (
    DefaultTab,
    _create_grid_layout,
    _create_label_widget,
)

from deepview.utils.auxiliaryfunctions import (
    read_config,
    get_raw_data_folder,
)

from pathlib import Path
import os
import pickle
import geopandas as gpd
import pandas as pd
import contextily as ctx
from shapely.geometry import Point
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtGui import QShowEvent

class GPSIMU_Interaction(DefaultTab):

    def __init__(self, root, parent, h1_description):
        super(GPSIMU_Interaction, self).__init__(root, parent, h1_description)

        self.root = root
        # root_cfg = read_config(root.config)
        self.data = pd.DataFrame()

        config = self.root.config  # project/config.yaml
        # Read file path for pose_config file. >> pass it on
        self.cfg = read_config(config)

        # self._set_page()
    def firstShowEvent(self, event: QShowEvent) -> None:
        self._set_page()

    def _set_page(self):
        self.main_layout.addWidget(_create_label_widget("Select raw data", "font:bold"))
        self.layout_attributes = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.main_layout.addWidget(_create_label_widget(""))  # dummy label

        self.bottom_layout = QHBoxLayout()
        self.main_layout.addLayout(self.bottom_layout)

    def _generate_layout_attributes(self, layout):
        RawDatacomboBox = QComboBox()
        unsup_data_path = get_raw_data_folder()
        rawdata_file_path_list = list(
            Path(os.path.join(self.cfg["project_path"], unsup_data_path)).glob('*.csv'),
        )
        for path in rawdata_file_path_list:
            RawDatacomboBox.addItem(str(path.name))
        self.RawDatacomboBox = RawDatacomboBox

        # combbox change
        self.data = pd.read_csv(rawdata_file_path_list[0], low_memory=False)
        self.data = self.data[['timestamp', 'acc_x', 'acc_y', 'acc_z', 'latitude', 'longitude']]
        self.data = self.data.dropna()
        self.data['index'] = self.data.index  # Add an index column
        self.RawDatacomboBox.currentTextChanged.connect(
            self.get_data_from_csv
        )

        featureExtractBtn = QPushButton('Plot figure')
        featureExtractBtn.setFixedWidth(160)
        featureExtractBtn.clicked.connect(self.initUI)

        layout.addWidget(self.RawDatacomboBox, 0, 0)
        layout.addWidget(featureExtractBtn, 0, 1)
        return

    # @Slot()
    def get_data_from_csv(self, filename):
        raw_data_path = get_raw_data_folder()
        datapath = os.path.join(self.cfg["project_path"], raw_data_path, filename)
        self.data = pd.read_csv(datapath, low_memory=False)
        self.data = self.data[['timestamp', 'acc_x', 'acc_y', 'acc_z', 'latitude', 'longitude']]
        self.data = self.data.dropna()
        self.data['index'] = self.data.index  # Add an index column
        return


    def initUI(self):
        # todo 当连续按两次时，上次的图层还在，需要删除
        # https://stackoverflow.com/questions/33682645/pyqt-how-do-i-replace-a-widget-using-a-button

        # Create figures
        self.gps_fig = Figure()
        self.imu_fig = Figure()

        # Create canvas for figures
        self.gps_canvas = FigureCanvas(self.gps_fig)
        self.imu_canvas = FigureCanvas(self.imu_fig)

        # Add canvases to layout
        # self.bottom_layout.deleteLater()
        self.bottom_layout.addWidget(self.imu_canvas, 2)  # Add IMU canvas first for left position
        self.bottom_layout.addWidget(self.gps_canvas, 2)  # Add GPS canvas second for right position

        # Plot data
        self.plot_data()

    def plot_data(self):
        # Plot GPS data with geopandas and contextily
        self.gps_ax = self.gps_fig.add_subplot(111)

        # Convert to GeoDataFrame
        self.data['geometry'] = [Point(xy) for xy in zip(self.data['longitude'], self.data['latitude'])]
        gdf = gpd.GeoDataFrame(self.data, geometry='geometry', crs="EPSG:4326")

        # Plot GeoDataFrame
        gdf.plot(ax=self.gps_ax, marker='o', color='blue', markersize=5, alpha=0.5)
        ctx.add_basemap(self.gps_ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

        self.gps_ax.set_title("GPS Data")
        self.gps_ax.set_xlabel("Longitude")
        self.gps_ax.set_ylabel("Latitude")

        # Plot IMU data using the index as x-axis
        self.imu_ax = self.imu_fig.add_subplot(111)
        self.imu_ax.plot(self.data['index'], self.data['acc_x'], label='Accelerometer X',
                         color='blue')
        self.imu_ax.plot(self.data['index'], self.data['acc_y'], label='Accelerometer Y',
                         color='orange')
        self.imu_ax.plot(self.data['index'], self.data['acc_z'], label='Accelerometer Z',
                         color='green')

        self.imu_ax.set_title("IMU Data")
        self.imu_ax.set_xlabel("Timestamp")
        self.imu_ax.set_ylabel("IMU Values")
        self.imu_ax.legend()

        # Set x-ticks to display timestamps
        interval = int(len(self.data)/10)
        self.imu_ax.set_xticks(self.data['index'][::interval])
        self.imu_ax.set_xticklabels(self.data['timestamp'][::interval])
        # self.imu_ax.set_xticklabels(self.data['timestamp'][::interval].dt.strftime('%H:%M:%S'))

        # Connect events
        self.imu_fig.canvas.mpl_connect('button_press_event', self.on_imu_click)
        self.gps_fig.canvas.mpl_connect('button_press_event', self.on_gps_click)

        self.gps_canvas.draw()
        self.imu_canvas.draw()

    def on_imu_click(self, event):
        # Get the index of the clicked IMU point
        ind = self.get_closest_point(event,
                                     self.data[['index', 'acc_x', 'acc_y', 'acc_z']],
                                     is_imu=True)
        if ind is not None:
            gps_point = self.data.iloc[ind][['longitude', 'latitude']]

            # Clear previous highlights
            self.clear_highlights(self.gps_ax, None)

            # Highlight the corresponding GPS data point
            self.gps_ax.scatter([gps_point['longitude']], [gps_point['latitude']], color='red')

            self.gps_canvas.draw()

    def on_gps_click(self, event):
        # Get the index of the clicked GPS point
        ind = self.get_closest_point(event, self.data[['longitude', 'latitude']], is_imu=False)
        if ind is not None:
            imu_point = self.data.iloc[ind][['index', 'acc_x', 'acc_y', 'acc_z']]

            # Clear previous highlights
            self.clear_highlights(self.imu_ax, None)

            # Highlight the corresponding IMU data point
            self.imu_ax.plot(self.data['index'], self.data['acc_x'], label='Accelerometer X',
                             color='blue')
            self.imu_ax.plot(self.data['index'], self.data['acc_y'], label='Accelerometer Y',
                             color='orange')
            self.imu_ax.plot(self.data['index'], self.data['acc_z'], label='Accelerometer Z',
                             color='green')
            self.imu_ax.scatter([imu_point['index']], [imu_point['acc_x']], color='red')
            self.imu_ax.scatter([imu_point['index']], [imu_point['acc_y']], color='red')
            self.imu_ax.scatter([imu_point['index']], [imu_point['acc_z']], color='red')

            self.imu_canvas.draw()

    def get_closest_point(self, event, data, is_imu):
        # Ensure the click is within the plot area
        if event.xdata is None or event.ydata is None:
            return None

        if is_imu:
            # Calculate the distance of the click from each index
            distances = [abs(event.xdata - idx) for idx in data['index']]
        else:
            # Calculate the distance of the click from each GPS point
            distances = [(event.xdata - x) ** 2 + (event.ydata - y) ** 2 for x, y in data.values]

        min_dist = min(distances)
        min_index = distances.index(min_dist)

        # Return the index of the closest point if the distance is small enough
        return min_index

    def clear_highlights(self, ax, scatter):
        # Clear previous highlights by removing all collections except the original scatter plot
        while len(ax.collections) > 1:
            ax.collections.pop()

        if scatter:
            # Re-draw the original scatter plot
            ax.add_collection(scatter)


    def update_result(self, result):
        # self.result_label.setText(f"Result: {result}")
        print(result)