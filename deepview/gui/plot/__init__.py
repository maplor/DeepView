from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QPushButton
from PySide6.QtCore import QTimer, QRectF
import numpy as np
import pandas as pd
import pyqtgraph as pg

np.random.seed(0)
df = pd.DataFrame({'Col ' + str(i + 1): np.random.rand(30) for i in range(6)})  # 随机生成2组30以内的数

clickedPen = pg.mkPen('b', width=2)

class PlotWithInteraction(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.main_layout = QHBoxLayout()

        self.setLayout(self.main_layout)

        # 模拟计算时间
        self.computeTimer = QTimer()

        # 缓存框选区域
        self.selectRect = QRectF(0, 0, 1, 1)

        # 缓存改变的点
        self.lastChange = []

        # 创建图形元素
        self.createPlot()
    
    def createPlot(self):
        # 左边的列
        viewL = pg.GraphicsLayoutWidget()
        p2 = viewL.addPlot(row=1, col=0)
        p3 = viewL.addPlot(row=2, col=0)
        p1 = viewL.addPlot(row=0, col=0)

        p1.plot(df['Col 1'], df['Col 2'], pen=None, symbol='o', symbolBrush=pg.mkBrush(255, 0, 0))
        p2.plot(df['Col 3'], df['Col 4'], pen=None, symbol='o', symbolBrush=pg.mkBrush(0, 255, 0))
        p3.plot(df['Col 5'], df['Col 6'], pen=None, symbol='o', symbolBrush=pg.mkBrush(0, 0, 255))

        p2.setXLink(p1)
        p2.setYLink(p1)

        p3.setXLink(p1)
        p3.setYLink(p1)

        self.main_layout.addWidget(viewL)

        # 中间按钮
        btn = QPushButton('计算')
        btn.setFixedWidth(100)
        btn.clicked.connect(self.handleCompute)
        self.main_layout.addWidget(btn)
        self.btn = btn

        # 右侧结果
        viewR = pg.PlotWidget()
        self.viewR = viewR

        n = 100
        scatterItem = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        pos = np.random.normal(size=(2,n), scale=100)
        spots = [{'pos': pos[:,i], 'data': i} for i in range(n)]
        scatterItem.addPoints(spots)
        self.scatterItem = scatterItem

        viewR.addItem(scatterItem)


        rect = viewR.viewRect()
        w = rect.width()
        h = rect.height()
        x = rect.x()
        y = rect.y()

        roi = pg.ROI([x + w * 0.4, y + h * 0.4], [w * 0.2, h * 0.2])
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

        viewR.addItem(roi)

        roi.sigRegionChanged.connect(self.handleROIChange)

        self.handleROIChange(roi)

        self.main_layout.addWidget(viewR)

    def handleCompute(self):
        print('开始计算...')
        self.btn.setText('计算中...')
        self.btn.setEnabled(False)

        # 模拟计算时间
        self.computeTimer.singleShot(1500, self.handleComputeFinish)

    def handleComputeFinish(self):
        print('计算完成')
        self.btn.setText('计算')
        self.btn.setEnabled(True)

    def handleROIChange(self, roi: pg.ROI):
        pos: pg.Point = roi.pos()
        size: pg.Point = roi.size()

        self.selectRect.setRect(pos.x(), pos.y(), size.x(), size.y())
        points = self.scatterItem.pointsAt(self.selectRect)
        print('points: %s'%([i.data() for i in points]))

        # 重置上次点的状态
        for p in self.lastChange:
            p.resetPen()

        # 修改点状态，并记录点
        for p in points:
            p.setPen(clickedPen)
        self.lastChange = points

if __name__ == '__main__':
    app = QApplication([])
    window = PlotWithInteraction()
    window.show()
    app.exec()        
