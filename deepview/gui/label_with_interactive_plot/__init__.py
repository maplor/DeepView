import math
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QCheckBox, QComboBox, QLabel, QSpinBox
from PySide6.QtCore import QTimer, QRectF, Qt
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QCheckBox
import os
from pathlib import Path

clickedPen = pg.mkPen('b', width=2)

class LabelWithInteractivePlot(QWidget):
    def __init__(self, root, data: pd.DataFrame, cfg) -> None:
        super().__init__()

        # init main_layout & top_layout & bottom_layout
        self.initLayout()

        # TODO simulation training time, delete it after finish
        self.computeTimer = QTimer()

        # TODO remove hardcode column name, read from data
        self.columnList = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']
        self.selectColumn = []

        self.data = data
        self.cfg = cfg

        # status
        self.isTarining = False

        self.maxColumn = 1
        self.maxRow = 1

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
        self.renderLeftPlot()
        self.updateBtn()
    
    '''
    ==================================================
    bottom left area: plot
    - self.viewL GraphicsLayoutWidget
    - self.leftPlotList list(PlotItem)
    ==================================================
    '''
    def createLeftPlot(self):
        viewL = pg.GraphicsLayoutWidget()
        self.viewL = viewL
        self.bottom_layout.addWidget(viewL)

        self.updateLeftPlotList()
        self.renderLeftPlot()
    
    def updateLeftPlotList(self):
        df = self.data

        self.leftPlotList = []

        if not hasattr(self, 'scatterItem') or len(self.selectPoints) == 0:
            plotList = []
            for i, column in enumerate(self.selectColumn):
                plot = pg.PlotItem(title=column, name=column, axisItems={'bottom': pg.DateAxisItem()})
                plot.plot(df['datetime'], df[column], pen=pg.mkPen(i))
                plotList.append(plot)
            self.leftPlotList.append(plotList)

        else:
            for p_index, p in enumerate(self.selectPoints):
                index, start, end = p.data()
                plotList = []
                for c_index, column in enumerate(self.selectColumn):
                    plot = pg.PlotItem(title=column, name=column, axisItems={'bottom': pg.DateAxisItem()})
                    plot.plot(df['datetime'], df[column], pen=pg.mkPen(c_index))
                    plot.addItem(pg.LinearRegionItem(values=[df['datetime'][start], df['datetime'][end]], movable=False))
                    plot.setXRange(df['datetime'][start], df['datetime'][end])
                    plotList.append(plot)
                for plotA in plotList:
                    for plotB in plotList:
                        plotA.setXLink(plotB)
                self.leftPlotList.append(plotList)

    def renderLeftPlot(self):
        # clear all items
        self.viewL.clear()

        # add plot item according to select column
        index = 0
        for column in range(self.maxColumn):
            for row in range(self.maxRow):
                plotList = self.leftPlotList[index]
                for plot_index, plot in enumerate(plotList):
                    self.viewL.addItem(plot, col=column, row=row*len(plotList)+plot_index)
                index += 1
                if index >= len(self.leftPlotList):
                    return
            self.viewL.nextRow()
            
    '''
    ==================================================
    bottom center area: button
    - self.computeBtn QPushButton
    ==================================================
    '''
    
    def handleCompute(self):
        print('start train...')
        self.isTarining = True
        self.updateBtn()

        # simulation training time
        self.computeTimer.singleShot(1500, self.handleComputeFinish)

    def handleComputeFinish(self):
        #print('finish train')
        self.isTarining = False
        self.updateBtn()
        
        self.renderRightPlot()

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
        new_spots = []
        for spot in self.scatterItem.points():
            pos = spot.pos()
            i, start, end = spot.data()
            new_color = self.checkColor(self.data.loc[start, 'label'])
            new_spot = {'pos': (pos.x(), pos.y()), 'data': (i, start, end), 
                        'brush': pg.mkBrush(new_color)}
            new_spots.append(new_spot)
        self.scatterItem.setData(spots=new_spots)
    
    def renderRightPlot(self):
        self.viewC.clear()
        df = self.data

        rows = df.shape[0]
        
        # mock scatter number
        n = 100
        step = math.floor(rows / n)
        
        scatterItem = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None))
        pos = np.random.normal(size=(2,n), scale=100)

        start_indice, end_indice, pos = self.featureExtraction()

        # save something into data attribute
        spots = [{'pos': pos[i,:], 'data': (i, start_indice[i], end_indice[i]), 'brush': self.checkColor(self.data.loc[i*step,'label'])} for i in range(n)]
        
        scatterItem.addPoints(spots)
        self.scatterItem = scatterItem

        self.viewC.addItem(scatterItem)

        rect = self.viewC.viewCect()
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
        # 左上 - 旋转
        # roi.addRotateHandle([0, 1], [0.5, 0.5])

        self.viewC.addItem(roi)

        # cache select region
        self.selectRect = QRectF(0, 0, 1, 1)

        # cache highlight point
        self.lastChangePoint = []

        # cache plot mark
        self.lastMarkList = []

        roi.sigRegionChanged.connect(self.handleROIChange)
        roi.sigRegionChangeFinished.connect(self.handleROIChangeFinished)
        self.handleROIChange(roi)
        self.handleROIChangeFinished(roi)
    
    def clearSelection(self):
        self.selectPoints = []
        self.selectRect = None
    
    def clearPlot(self):
        self.viewL.clear()
        self.viewC.clear()

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
                selected_dfs.append(selected_range)
                start_indice.append(start_idx)
                end_indice.append(end_idx)
        
        return start_indice, end_indice, selected_dfs

    
    def featureExtraction(self):
        # TODO: Implement feature extraction function. Currently, the model output results in a NAN error, so random values are being used.

        start_indice, end_indice, selected_dfs = self.select_random_continuous_seconds()

        """

        sensor_columns=['acc_x', 'acc_y', 'acc_z']

        input = [df[sensor_columns].to_numpy() for df in selected_dfs]
    
        input = np.array(input)
        
        input = torch.from_numpy(input).double()

        modelfoldername = auxiliaryfunctions.get_model_folder(self.cfg)
        modelconfigfile = Path(
            os.path.join(
                self.cfg["project_path"], str(modelfoldername), "test", "model_cfg.yaml"
            )
        )

        os.chdir(
            str(Path(modelconfigfile).parents[0])
        )  # switch to folder of config_yaml (for logging)

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

        pos = np.random.normal(size=(len(start_indice),2), scale=100)

        return start_indice, end_indice, pos

    # highlight select point
    def handleROIChange(self, roi: pg.ROI):
        pos: pg.Point = roi.pos()
        size: pg.Point = roi.size()

        self.selectRect.setRect(pos.x(), pos.y(), size.x(), size.y())
        points = self.scatterItem.pointsAt(self.selectRect)
        # print('points: %s'%([i.data() for i in points]))

        # reset last change points
        for p in self.lastChangePoint:
            p.resetPen()

        # change point state and cache
        for p in points:
            p.setPen(clickedPen)
        self.lastChangePoint = points

    # add mark to plot, use sigRegionChangeFinished to reduce render
    def handleROIChangeFinished(self, roi: pg.ROI):
        pos: pg.Point = roi.pos()
        size: pg.Point = roi.size()

        self.selectRect.setRect(pos.x(), pos.y(), size.x(), size.y())
        points = self.scatterItem.pointsAt(self.selectRect)
        self.selectPoints = points

        self.updateLeftPlotList()
        self.renderLeftPlot()

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

        self.createLabelComboBox()
        self.createLabelButton()

        self.createSaveButton()
        self.createMaxColumnSpinBox()
        self.createMaxRowSpinBox()
    
    def createLabelComboBox(self):
        labelComboBox = QComboBox()
        self.currentLabel = "flying"
        labelComboBox.addItem("flying")
        labelComboBox.addItem("stationary")
        labelComboBox.addItem("foraging")
        labelComboBox.currentIndexChanged.connect(self.handleLabelComboBoxChange)
        self.settingPannel.addWidget(labelComboBox)
    
    def handleLabelComboBoxChange(self):
        self.currentLabel = self.comboBox.currentText()
    
    def createModelComboBox(self):
        modelComboBoxLabel = QLabel('Select model:')
        self.settingPannel.addWidget(modelComboBoxLabel)
        modelComboBox = QComboBox()
        modelComboBox.addItem("model 1")
        modelComboBox.addItem("model 2")
        modelComboBox.addItem("model 3")
        modelComboBox.currentIndexChanged.connect(self.handleModelComboBoxChange)
        self.settingPannel.addWidget(modelComboBox)
    
    def createFeatureExtractButton(self):
        computeBtn = QPushButton('Extract feature')
        self.computeBtn = computeBtn
        computeBtn.setFixedWidth(100)
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
        os.makedirs(os.path.join(self.cfg["project_path"],"edit-data",), exist_ok=True)
        edit_data_path = os.path.join(self.cfg["project_path"],"edit-data", self.RawDatacomboBox.currentText())
        self.data.to_csv(edit_data_path)
    
    def createMaxColumnSpinBox(self):
        maxColumnSpinBox = QSpinBox(self)
        maxColumnSpinBox.setMinimum(1)  # minimum value
        maxColumnSpinBox.setMaximum(5)  # maximum value
        maxColumnSpinBox.setValue(1)  # initial value
        maxColumnSpinBox.valueChanged.connect(self.maxColumnSpinBoxChange)
        self.settingPannel.addWidget(QLabel('Maximam column of left view'))
        self.settingPannel.addWidget(maxColumnSpinBox)
    
    def maxColumnSpinBoxChange(self):
        self.maxColumn = self.maxColumnSpinBox.value()
        self.renderLeftPlot()

    def createMaxRowSpinBox(self):
        maxRowSpinBox = QSpinBox(self)
        maxRowSpinBox.setMinimum(1)  # minimum value
        maxRowSpinBox.setMaximum(5)  # maximum value
        maxRowSpinBox.setValue(1)  # initial value
        maxRowSpinBox.valueChanged.connect(self.maxRowSpinBoxChange)
        self.settingPannel.addWidget(QLabel('Maximam row of left view'))
        self.settingPannel.addWidget(maxRowSpinBox)

    def maxRowSpinBoxChange(self): 
        self.maxRow = self.maxRowSpinBox.value()
        self.renderLeftPlot()
    
    def createRawDataComboBox(self):
        RawDataComboBoxLabel = QLabel('Select raw data:')
        self.settingPannel.addWidget(RawDataComboBoxLabel)
        RawDatacomboBox = QComboBox()
        rawdata_file_path_list = list(
        Path(os.path.join(self.cfg["project_path"], "raw-data")).glob('*.csv'),
        )
        for path in rawdata_file_path_list:
            RawDatacomboBox.addItem(str(path.name))
        RawDatacomboBox.currentIndexChanged.connect(self.handleRawDataComboBoxChange)
        self.settingPannel.addWidget(RawDatacomboBox)

    def handleRawDataComboBoxChange(self):
        self.updateRawData()
        self.clearPlot()
        self.clearSelection()
        
        self.updateLeftPlotList()
        self.renderLeftPlot()
    
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
        labelButton = QPushButton('Label')
        labelButton.clicked.connect(self.handleLabelButton)
        self.settingPannel.addWidget(labelButton)

    def handleLabelButton(self):
        points = self.scatterItem.pointsAt(self.selectRect)
        for p in points:
            index, start, end = p.data()
            self.data.loc[start:end,"label"] = self.currentLabel
        self.updateRightPlotColor()

if __name__ == '__main__':
    app = QApplication([])

    df = pd.DataFrame({
        'datetime': pd.date_range(start='1/1/2022', periods=10000),
        'acc_x': np.random.rand(10000),
        'acc_y': np.random.rand(10000),
        'acc_z': np.random.rand(10000),
        'gyro_x': np.random.rand(10000),
        'gyro_y': np.random.rand(10000),
        'gyro_z': np.random.rand(10000),
        'mag_x': np.random.rand(10000),
        'mag_y': np.random.rand(10000),
        'mag_z': np.random.rand(10000)
    })

    df['datetime'] = df['datetime'].apply(lambda x: x.timestamp())
    
    window = LabelWithInteractivePlot(df)
    window.show()
    app.exec()        
