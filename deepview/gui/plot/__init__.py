import math
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QCheckBox
from PySide6.QtCore import QTimer, QRectF, Qt
import numpy as np
import pandas as pd
import pyqtgraph as pg

clickedPen = pg.mkPen('b', width=2)

class PlotWithInteraction(QWidget):
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__()

        # init main_layout & top_layout & bottom_layout
        self.initLayout()

        # TODO simulation training time, delete it after finish
        self.computeTimer = QTimer()

        # TODO remove hardcode column name, read from data
        self.columnList = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']
        self.selectColumn = []

        self.data = data

        # status
        self.isTarining = False

        self.createCheckBoxList()

        self.createLeftPlot()
        self.createCenterBtn()
        self.createRightPlot()

        self.renderLeftPlot()
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

        df = self.data

        self.leftPlotList = []
        for column in self.columnList:
            plot = pg.PlotItem(title=column, name=column, axisItems = {'bottom': pg.DateAxisItem()})
            index = len(self.leftPlotList)
            
            plot.plot(df['datetime'], df[column], pen=pg.mkPen(index))
            # plot.plot(df['datetime'], df[column].values.astype(float), pen=pg.mkPen(index))

            # optional - link plot viewbox range
            # if index > 0:
            #     plot.setXLink(self.leftPlotList[0])
            #     plot.setYLink(self.leftPlotList[0])
            self.leftPlotList.append(plot)

    def renderLeftPlot(self):
        # clear all items
        self.viewL.clear()

        # add plot item according to select column
        for i in range(len(self.columnList)):
            if self.checkboxList[i].isChecked():
                self.viewL.addItem(self.leftPlotList[i])
                self.viewL.nextRow()
                self.leftPlotList[i].autoRange()

        # placeholder
        if len(self.selectColumn) == 0:
            self.viewL.addLabel('Please select the column of interest above')

    '''
    ==================================================
    bottom center area: button
    - self.computeBtn QPushButton
    ==================================================
    '''
    def createCenterBtn(self):
        btn = QPushButton('Train')
        btn.setFixedWidth(100)
        btn.setEnabled(False)
        btn.clicked.connect(self.handleCompute)
        self.bottom_layout.addWidget(btn)
        self.computeBtn = btn

    def handleCompute(self):
        print('start train...')
        self.isTarining = True
        self.updateBtn()

        # simulation training time
        self.computeTimer.singleShot(1500, self.handleComputeFinish)

    def handleComputeFinish(self):
        print('finish train')
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
            self.computeBtn.setText('Training...')
        else:
            self.computeBtn.setText('Train')

    '''
    ==================================================
    bottom right area: result plot
    - self.viewR PlotWidget
    - self.selectRect QRect
    - self.lastChangePoint list(SpotItem)
    - self.lastMarkList list(LinearRegionItem)
    ==================================================
    '''
    def createRightPlot(self):
        viewR = pg.PlotWidget()
        self.viewR = viewR
        self.bottom_layout.addWidget(viewR)

    def renderRightPlot(self):
    
        df = self.data

        rows = df.shape[0]
        
        # mock scatter number
        n = 100
        step = math.floor(rows / n)
        scatterItem = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        pos = np.random.normal(size=(2,n), scale=100)
        # save something into data attribute
        spots = [{'pos': pos[:,i], 'data': (i, i * step, i * step + step - 1)} for i in range(n)]
        scatterItem.addPoints(spots)
        self.scatterItem = scatterItem

        self.viewR.addItem(scatterItem)

        rect = self.viewR.viewRect()
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

        self.viewR.addItem(roi)

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
        print('roi select points data: %s'%([i.data() for i in points]))

        # remove last marks, removeItem() will ignore item not add
        for mark in self.lastMarkList:
            for p in self.leftPlotList:
                p.removeItem(mark)

        df = self.data

        markList = []
        # change point state and cache
        for p in points:
            # get data attribute
            index, start, end = p.data()

            # loop all plot and create mark
            # TODO only create mark for select column
            for p in self.leftPlotList:
                # highlight region
                lr = pg.LinearRegionItem(values=[df['datetime'][start], df['datetime'][end]], movable=False)
                # label
                pg.InfLineLabel(lr.lines[1], str(index), position=0.95, rotateAxis=(1,0), anchor=(1, 1))
                # add and cache
                markList.append(lr)
                p.addItem(lr)

        self.lastMarkList = markList


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
    
    # df2 = pd.read_csv('/Users/zyz/Git/DeepView/Umineko2018_small_data_LB07_lb0001.csv')
    # df2['datetime'] = pd.to_datetime(df2['timestamp']).apply(lambda x: x.timestamp())
    
    window = PlotWithInteraction(df)
    window.show()
    app.exec()        
