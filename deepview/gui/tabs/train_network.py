#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from PySide6.QtCore import Signal
from PySide6.QtGui import QShowEvent

from deepview.gui.components import (
    DefaultTab,
    # ShuffleSpinBox,
)
from deepview.gui.tabs.train_network_ui import TrainNetworkUiMixin
from deepview.gui.tabs.train_network_training import TrainNetworkTrainingMixin
from deepview.utils import auxiliaryfunctions


class TrainNetwork(TrainNetworkTrainingMixin, TrainNetworkUiMixin, DefaultTab):
    # 定义进度信号
    progress_update = Signal(int)

    def __init__(self, root, parent, h1_description):
        super(TrainNetwork, self).__init__(root, parent, h1_description)
        self.root = root
        # get sensor/columns dictionary from config.yaml
        root_cfg = auxiliaryfunctions.read_config(self.root.config)
        self.sensor_dict = root_cfg['sensor_dict']

        self.models = ['AE_CNN', 'SimCLR_LSTM', 'shortAE']
        self.select_column = []
        self.max_iter = 30
        self.learning_rate = 0.0001
        self.batch_size = 512
        self.net_type = self.models[0]
        self.data_length = 180

    # 在第一次渲染 tab 时才构造内容
    def firstShowEvent(self, event: QShowEvent) -> None:
        self._set_page()
