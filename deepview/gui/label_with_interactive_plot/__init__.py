import math
from typing import List
import torch
from sklearn.manifold import TSNE
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QCheckBox,
    QComboBox,
    QLabel,
    QSpinBox,
    QDialog,
    QRadioButton,
    QSplitter,
)
from PySide6.QtCore import QTimer, QRectF, Qt
from PySide6.QtGui import QColor
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QCheckBox
from functools import partial
import os
from pathlib import Path

clickedPen = pg.mkPen('b', width=2)

class LabelOption(QDialog):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        self.radio_button1 = QRadioButton("flying")
        self.radio_button2 = QRadioButton("stationary")
        self.radio_button3 = QRadioButton("foraging")

        layout.addWidget(self.radio_button1)
        layout.addWidget(self.radio_button2)
        layout.addWidget(self.radio_button3)

        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.confirm_selection)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)

    def confirm_selection(self):
        if self.radio_button1.isChecked():
            selected_option = "flying"
        elif self.radio_button2.isChecked():
            selected_option = "stationary"
        elif self.radio_button3.isChecked():
            selected_option = "foraging"
        else:
            selected_option = None

        self.accept()

        return selected_option

class LabelWithInteractivePlot(QWidget):

    def __init__(self, root, data: pd.DataFrame, cfg) -> None:
        super().__init__()

        self.root = root

        # init main_layout & top_layout & bottom_layout
        self.initLayout()

        # TODO simulation training time, delete it after finishing
        self.computeTimer = QTimer()

        # remove hardcode column name, read from data
        self.columnList = []
        full_columns = list(data.columns.values)
        for column in full_columns:
            if ('label' not in column) and ('time' not in column):
                self.columnList.append(column)

        self.selectColumn = self.columnList
        # self.selectColumn = ['acc_x', 'acc_y', 'acc_z']

        self.data = data
        self.cfg = cfg

        # status
        self.isTarining = False
        self.mode = 'add'

        self.createCheckBoxList()
        self.createLeftPlot()
        self.createCenterPlot()
        self.createRightSettingPannel()

        self.updateBtn()

    def initLayout(self):
        self.main_layout = QVBoxLayout()

        self.setLayout(self.main_layout)

        self.top_layout = QHBoxLayout()
        self.bottom_layout = QHBoxLayout()

        self.main_layout.addLayout(self.top_layout)
        self.main_layout.addLayout(self.bottom_layout)

    '''
    ==================================================
    top area checkbox: list
    - self.checkboxList list(QCheckBox)
    ==================================================
    '''

    def createCheckBoxList(self):
        self.checkboxList = []
        for column in self.columnList:
            cb = QCheckBox(column)
            if column in self.selectColumn:
                cb.setChecked(True)
            self.top_layout.addWidget(cb)
            self.checkboxList.append(cb)
            cb.stateChanged.connect(self.handleCheckBoxStateChange)

        # add a stretch to fill the remaining area and keep the checkbox on the left
        self.top_layout.addStretch()

    def handleCheckBoxStateChange(self):
        newSelectColumn = []
        for i, column in enumerate(self.columnList):
            if self.checkboxList[i].isChecked():
                newSelectColumn.append(column)
        self.selectColumn = newSelectColumn
        print('selectColumn: %s'%(newSelectColumn))

        self.updateLeftPlotList()
        self.updateBtn()

    '''
    ==================================================
    bottom left area: plot
    - self.viewL GraphicsLayoutWidget
    - self.leftPlotList list(PlotItem)
    ==================================================
    '''

    def createLeftPlot(self):
        viewL = QVBoxLayout()
        self.viewL = viewL
        self.splitter = QSplitter(Qt.Vertical)
        self.plot_widgets = [None] * len(self.columnList)
        self.click_begin = True
        self.start_line = [None] * len(self.columnList)
        self.end_line = [None] * len(self.columnList)
        self.regions = []
        for _ in range(len(self.columnList)):
            self.regions.append([])
        self.bottom_layout.addLayout(viewL)

        self.resetLeftPlot()
        self.splitter.setSizes([100] * len(self.columnList))
        self.viewL.addWidget(self.splitter)

    def resetLeftPlot(self):
        # reset widget
        for i in range(len(self.columnList)):
            if self.plot_widgets[i] is not None:
                self.splitter.removeWidget(self.plot_widgets[i])
                self.plot_widgets[i].close()
                self.plot_widgets[i] = None

        # add widget
        df = self.data
        for i, column in enumerate(self.columnList):
            # if (type(df[column].values[0]) == int) or (type(df[column].values[0]) == float):
            plot = pg.PlotWidget(title=column, name=column, axisItems={'bottom': pg.DateAxisItem()})
            plot.plot(df['datetime'], df[column], pen=pg.mkPen(i))
            plot.scene().sigMouseClicked.connect(self.mouse_clicked)
            plot.scene().sigMouseMoved.connect(self.mouse_moved)
            self.plot_widgets[i] = plot
            self.splitter.addWidget(plot)

    def updateLeftPlotList(self):
        for i, column in enumerate(self.columnList):
            if column in self.selectColumn:
                self.plot_widgets[i].show()
            else:
                self.plot_widgets[i].hide()

    def _to_idx(self, start_ts, end_ts):
        selected_indices = self.data[(self.data['datetime'] >= start_ts)
                                     & (self.data['datetime'] <= end_ts)].index
        return selected_indices.values[0], selected_indices.values[-1]

    def _add_region(self, pos):
        if self.click_begin:    # 记录开始位置
            self.click_begin = False
            for i, plot in enumerate(self.plot_widgets):
                self.start_line[i] = pg.InfiniteLine(pos.x(), angle=90, movable=False)
                self.end_line[i] = pg.InfiniteLine(pos.x(), angle=90, movable=False)
                plot.addItem(self.start_line[i])
                plot.addItem(self.end_line[i])
        else:                   # 记录结束位置
            self.click_begin = True
            for i, plot in enumerate(self.plot_widgets):
                plot.removeItem(self.start_line[i])
                plot.removeItem(self.end_line[i])

                region = pg.LinearRegionItem([self.start_line[i].value(), self.end_line[i].value()], brush=(0, 0, 255, 100))
                region.sigRegionChanged.connect(self._region_changed)

                self.start_line[i] = None
                self.end_line[i] = None

                plot.addItem(region)
                self.regions[i].append(region)
            start_idx, end_idx = self._to_idx(int(region.getRegion()[0]), int(region.getRegion()[1]))
            print(f'Selected range: from index {start_idx} to index {end_idx}')

    def _region_changed(self, region):
        idx = 0
        for reg_lst in self.regions:
            for i, reg in enumerate(reg_lst):
                if reg == region:
                    idx = i
                    break
        for reg_lst in self.regions:
            reg_lst[idx].setRegion(region.getRegion())

    def _del_region(self, pos):
        for i, pwidget in enumerate(self.plot_widgets):
            for reg in self.regions[i]:
                if reg.getRegion()[0] < pos.x() and reg.getRegion()[1] > pos.x():
                    pwidget.removeItem(reg)
                    self.regions[i].remove(reg)
                    break
        start_idx, end_idx = self._to_idx(int(reg.getRegion()[0]), int(reg.getRegion()[1]))
        print(f'Delete region({start_idx}, {int(end_idx)})')

    def _edit_region(self, pos):
        set_val = None
        for i, _ in enumerate(self.regions):
            for reg in self.regions[i]:
                if reg.getRegion()[0] < pos.x() and reg.getRegion()[1] > pos.x():
                    if set_val is None:
                        dialog = LabelOption()
                        if dialog.exec() == QDialog.Accepted:
                            set_val = dialog.confirm_selection()
                        else:
                            set_val = 'None'
                    reg.setBrush(self.checkColor(set_val))
        start_idx, end_idx = self._to_idx(int(reg.getRegion()[0]), int(reg.getRegion()[1]))
        print(f'Edit region({start_idx}, {end_idx}) label: {set_val}')

    def mouse_clicked(self, event):
        if event.button() == Qt.LeftButton and hasattr(self, 'scatterItem'):
            pos = self.plot_widgets[0].plotItem.vb.mapSceneToView(event.pos())

            if self.mode == 'add':
                self._add_region(pos)
            elif self.mode == 'edit':
                self._edit_region(pos)
            elif self.mode == 'del':
                self._del_region(pos)

    def mouse_moved(self, event):
        pos = self.plot_widgets[0].plotItem.vb.mapSceneToView(event)
        if not self.click_begin:
            for line in self.end_line:
                line.setPos(pos.x())

    '''
    ==================================================
    bottom center area: button
    - self.computeBtn QPushButton
    ==================================================
    '''

    def handleCompute(self):
        print('start trainning...')
        self.isTarining = True
        self.updateBtn()

        # TODO: simulation training time
        self.computeTimer.singleShot(1500, self.handleComputeFinish)

    def handleComputeFinish(self):
        #print('finish train')
        self.isTarining = False
        self.updateBtn()

        self.renderRightPlot()  # feature extraction function here
        self.updateLeftPlotList()

    def updateBtn(self):
        # enabled
        if len(self.selectColumn) == 0 or self.isTarining:
            self.computeBtn.setEnabled(False)
        else:
            self.computeBtn.setEnabled(True)

        # text
        if self.isTarining:
            self.computeBtn.setText('Extracting feature...')
        else:
            self.computeBtn.setText('Feature extraction')

    '''
    ==================================================
    bottom center area: result plot
    - self.viewC PlotWidget
    - self.selectRect QRect
    - self.lastChangePoint list(SpotItem)
    - self.lastMarkList list(LinearRegionItem)
    ==================================================
    '''
    def createCenterPlot(self):
        viewC = pg.PlotWidget()
        self.viewC = viewC
        self.bottom_layout.addWidget(viewC)

    def checkColor(self, label):
        if label == 'flying':
            return pg.mkBrush(0, 0, 255, 120)
        elif label == 'stationary':
            return pg.mkBrush(255, 0, 0, 120)
        elif label == 'foraging':
            return pg.mkBrush(0, 255, 0, 120)
        else:
            return pg.mkBrush(255, 255, 255, 120)

    def updateRightPlotColor(self):
        # reset spots
        spots = []
        for spot in self.scatterItem.points():
            pos = spot.pos()
            i, start, end = spot.data()
            color = self.checkColor(self.data.loc[start, 'label'])
            spot = {'pos': (pos.x(), pos.y()), 'data': (i, start, end),
                        'brush': pg.mkBrush(color)}
            spots.append(spot)

        # update spots
        for reg in self.regions[0]:
            for spot in spots:
                idx_begin, idx_end = self._to_idx(int(reg.getRegion()[0]), int(reg.getRegion()[1]))
                if idx_begin < spot['data'][1] and idx_end > spot['data'][2]:
                    spot['brush'] = reg.brush

        self.scatterItem.setData(spots=spots)

    def renderRightPlot(self):
        self.viewC.clear()
        df = self.data

        # rows = df.shape[0]

        # get model parameters from model_path
        # todo: 从下拉框中读取model_path
        model_path = self.model_path_list[0]

        from deepview.utils import auxiliaryfunctions
        model_name, data_length, column_names = \
            auxiliaryfunctions.get_param_from_path(model_path)  # todo now just test the first model


        # mock scatter number
        # n = 100
        # step = math.floor(rows / n)

        scatterItem = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None))
        # pos = np.random.normal(size=(2, n), scale=100)  # PCA result of latent feature

        # split the dataframe into segments to get latent feature and index
        start_indice, end_indice, pos = self.featureExtraction(df,
                                                               model_path,
                                                               model_name,
                                                               data_length,
                                                               column_names)

        # save something into data attribute
        n = len(start_indice)
        spots = [{'pos': pos[i,:], 'data': (i, start_indice[i], end_indice[i]), 'brush': self.checkColor(self.data.loc[i*data_length,'label'])} for i in range(n)]

        # plot spots on the scatter figure
        scatterItem.addPoints(spots)
        self.scatterItem = scatterItem

        self.viewC.addItem(scatterItem)

    def select_random_continuous_seconds(self, num_samples=100, points_per_second=90):
        df = self.data
        selected_dfs = []
        start_indice = []
        end_indice = []

        while len(selected_dfs) < num_samples:
            start_idx = np.random.randint(0, len(df) - points_per_second)
            end_idx = start_idx + points_per_second - 1
            selected_range = df.iloc[start_idx:end_idx+1]

            if not selected_range[['acc_x', 'acc_y', 'acc_z']].isna().any().any():
                selected_dfs.append(selected_range)  # dataframe segment from start_idx to end_idx
                start_indice.append(start_idx)
                end_indice.append(end_idx)

        return start_indice, end_indice, selected_dfs


    def featureExtraction(self, target_data, model_path, model_name, data_length, column_names):
        # TODO: Implement feature extraction function. Currently, the model output results in a NAN error, so random values are being used.

        # start_indice, end_indice, selected_dfs = self.select_random_continuous_seconds()

        """
        target_data = dataframe of selected data (oneday pkl file)
        model_path = full path of model weight
        model_name = unsupuervised model name in string format
        data_length = length of input for unsupervised model in int format
        column_names = ['acc_x', 'acc_y', 'acc_z']

        # 2 get model from dv-models
        modelcfg = load_config(modelconfigfile)
        net_type = modelcfg["net_type"]
        # project_path = cfg['project_path']
        model_path = list(
            Path(
            os.path.join(
                self.cfg["project_path"], str(modelfoldername), "train")
            ).glob('*.pth')
        )[0]
        device = 'cpu'
        model = get_model(p_backbone=net_type, p_setup='autoencoder')  # set backbone model=ResNet18, SSL=simclr, weight

        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.to(device)
        model.double()
        output, _ = model(input)

        output = output.detach().numpy().reshape(output.shape[0], -1)

        pca = PCA(n_components=2)
        pos = pca.fit_transform(output)

        """
        # preprocessing: find column names in dataframe
        new_column_names = find_data_columns(target_data, column_names)

        segments, start_indices = split_dataframe(target_data, data_length)
        end_indices = [i + data_length for i in start_indices]
        # get latent feature
        from deepview.clustering_pytorch.nnet.common_config import get_model
        from deepview.clustering_pytorch.datasets.factory import prepare_unsup_dataset
        from deepview.clustering_pytorch.nnet.train_utils import AE_eval_time_series

        model = get_model(p_backbone=model_name, p_setup='autoencoder')
        model.load_state_dict(torch.load(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        train_loader = prepare_unsup_dataset(segments, new_column_names)
        representation_list, _, _, _ = \
            AE_eval_time_series(train_loader, model, device)

        # PCA latent representation to shape=(2, len)
        repre_concat = np.concatenate(representation_list)
        repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1)
        repre_tsne = reduce_dimension_with_tsne(repre_reshape)

        # pos = np.random.normal(size=(len(start_indice), 2), scale=100)

        return start_indices, end_indices, repre_tsne

    '''
    ==================================================
    bottom right area: setting panel
    - self.settingPannel QVBoxLayout
    - self.currentLabel str
    - self.maxColumn int
    - self.maxRow int
    ==================================================
    '''

    def createRightSettingPannel(self):
        settingPannel = QVBoxLayout()
        self.settingPannel = settingPannel
        self.bottom_layout.addLayout(self.settingPannel)

        self.settingPannel.setAlignment(Qt.AlignTop)

        self.createRawDataComboBox()

        self.createModelComboBox()
        self.createFeatureExtractButton()

        self.createLabelButton()
        self.createSaveButton()

    def createModelComboBox(self):
        modelComboBoxLabel = QLabel('Select model:')
        self.settingPannel.addWidget(modelComboBoxLabel)
        modelComboBox = QComboBox()
        from deepview.utils import auxiliaryfunctions
        # Read file path for pose_config file. >> pass it on
        config = self.root.config
        cfg = auxiliaryfunctions.read_config(config)
        unsup_model_path = auxiliaryfunctions.get_unsup_model_folder(cfg)

        model_path_list = auxiliaryfunctions.grab_files_in_folder_deep(
            os.path.join(self.cfg["project_path"], unsup_model_path),
            ext='*.pth')
        self.model_path_list = model_path_list
        for path in model_path_list:
            modelComboBox.addItem(str(Path(path).name))
        modelComboBox.currentIndexChanged.connect(self.handleModelComboBoxChange)
        self.settingPannel.addWidget(modelComboBox)

    def createFeatureExtractButton(self):
        computeBtn = QPushButton('Extract feature')
        self.computeBtn = computeBtn
        computeBtn.setFixedWidth(160)
        computeBtn.setEnabled(False)
        computeBtn.clicked.connect(self.handleCompute)
        self.settingPannel.addWidget(computeBtn)

    def handleModelComboBoxChange(self):
        print('ModelComboBox changed')

    def createSaveButton(self):
        saveButton = QPushButton('Save')
        saveButton.clicked.connect(self.handleSaveButton)
        self.settingPannel.addWidget(saveButton)

    def handleSaveButton(self):
        os.makedirs(os.path.join(self.cfg["project_path"], "edit-data",), exist_ok=True)
        edit_data_path = os.path.join(self.cfg["project_path"], "edit-data", self.RawDatacomboBox.currentText().replace(".pkl", ".csv"))
        self.data.to_csv(edit_data_path)

    def createRawDataComboBox(self):
        # find data at here:C:\Users\dell\Desktop\aa-bbb-2024-04-28\unsupervised-datasets\allDataSet
        RawDataComboBoxLabel = QLabel('Select raw data:')
        self.settingPannel.addWidget(RawDataComboBoxLabel)
        RawDatacomboBox = QComboBox()
        from deepview.utils import auxiliaryfunctions
        unsup_data_path = auxiliaryfunctions.get_unsupervised_set_folder()
        rawdata_file_path_list = list(
        Path(os.path.join(self.cfg["project_path"], unsup_data_path)).glob('*.pkl'),
        )
        for path in rawdata_file_path_list:
            RawDatacomboBox.addItem(str(path.name))
        RawDatacomboBox.currentIndexChanged.connect(self.handleRawDataComboBoxChange)
        self.settingPannel.addWidget(RawDatacomboBox)
        self.RawDatacomboBox = RawDatacomboBox

    def handleRawDataComboBoxChange(self):
        self.updateRawData()
        self.resetLeftPlot()

        self.updateLeftPlotList()

    def updateRawData(self):
        edit_data_path = os.path.join(self.cfg["project_path"],"edit-data", self.RawDatacomboBox.currentText())
        if Path(os.path.join(self.cfg["project_path"],"edit-data", self.RawDatacomboBox.currentText())).exists():
            self.data = pd.read_csv(edit_data_path)
        else:
            raw_data_path = os.path.join(self.cfg["project_path"], "raw-data", self.RawDatacomboBox.currentText())
            self.data = pd.read_csv(raw_data_path)
            self.data['label'] = ""

        self.data['datetime'] = pd.to_datetime(self.data['timestamp']).apply(lambda x: x.timestamp())

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

    def _change_mode(self, mode: str):
        print(f'Change mode to "{mode}"')
        self.mode = mode

def split_dataframe(df, segment_size):
    '''
    split the dataframe into segments based on segment_size
    '''
    segments = []
    start_indices = []
    num_segments = len(df) // segment_size

    for i in range(num_segments):
        start_index = i * segment_size
        end_index = start_index + segment_size
        segment = df.iloc[start_index:end_index]
        segments.append(segment)
        start_indices.append(start_index)

    # Handle the last segment which may not have the full segment size
    start_index = num_segments * segment_size
    last_segment = df.iloc[start_index:]
    if len(last_segment) == segment_size:
        segments.append(last_segment)
        start_indices.append(start_index)

    return segments, start_indices

def find_data_columns(dataf, column_names):
    new_column_names = []
    original_columns = list(dataf.columns.values)
    clean_columns = [i.replace('_', '') for i in original_columns]
    for column_name in column_names:
        for orig_c, clean_c in zip(original_columns, clean_columns):
            if column_name == clean_c:
                new_column_names.append(orig_c)
    return new_column_names

def reduce_dimension_with_tsne(array, method='tsne'):
    # tsne or pca
    tsne = TSNE(n_components=2)
    reduced_array = tsne.fit_transform(array)
    return reduced_array
