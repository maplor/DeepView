import os
from pathlib import Path
from PySide6.QtWidgets import QLabel
import pandas as pd

from deepview.gui.components import (
    DefaultTab,
    TestfileSpinBox,
    _create_horizontal_layout,
    _create_label_widget,
    _create_vertical_layout,
)
from deepview.gui.plot import PlotWithInteraction

class InteractionPlot(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(InteractionPlot, self).__init__(root, parent, h1_description)

        self._set_page()


    def _set_page(self):
        config = self.root.config  # project/config.yaml
        df = get_plot_data(config)
        if df is None:
            self.main_layout.addWidget(QLabel('can not find raw data'))
            return
        
        self.main_layout.addWidget(PlotWithInteraction(df))

def get_plot_data(config):
    """
    Extracts the scoremap, locref, partaffinityfields (if available).

    Returns a dictionary
    read data and model structure in this function
    ----------
    config : string
        Full path of the config.yaml file as a string.
    """
    from deepview.utils import auxiliaryfunctions

    start_path = os.getcwd()

    # Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)

    rawdata_file = list(
        Path(os.path.join(cfg["project_path"], "raw-data")).glob('*.csv'),
    )[0]

    if not rawdata_file:
        print('can not find raw data')
        return
    
    df = pd.read_csv(rawdata_file)

    df['datetime'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.timestamp())

    os.chdir(str(start_path))  # Change the current working directory to the specified path

    return df
