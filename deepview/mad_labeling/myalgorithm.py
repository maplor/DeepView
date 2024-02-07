"""This is the content of my_algorithm.py, which holds my first algorithm plugin."""

import warnings
from typing import Dict
import pandas as pd
from mad_gui import BaseAlgorithm
from mad_gui.models.local import PlotData
from mad_gui.components.dialogs.user_information import UserInformation

class MyAlgorithm(BaseAlgorithm):
    @classmethod
    def name(cls) -> str:
        name = "Algorithm to do ..."
        warnings.warn("Please give you algorithm a meaningful name.")
        return name

    def process_data(self, data: Dict[str, PlotData]):
        for plot_name, sensor_plot in data.items():
            # Use the currently plotted data to create annotations
            annotations = self.create_annotations(sensor_plot.data, sensor_plot.sampling_rate_hz)
            UserInformation.inform(f"Found {len(annotations)} annotations for {plot_name}.")
            if not all(col in annotations.columns for col in ["start", "end"]):
                raise KeyError("Please make sure the dataframe returned from create_annotations has the columns "
                           "'start' and 'end'.")
            sensor_plot.annotations["Activity"].data = annotations

    @staticmethod
    def create_annotations(sensor_data: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:
        """Some code that creates a pd.DataFrame with the columns `start` and `end`.

        Each row corresponds to one annotation to be plotted.
        """
        #########################################################################
        ###                                 README                            ###
        ### Here you create a dataframe, which has the columns start and end. ###
        ###  For each of the columns, the GUI will then plot one annotation.  ###
        ###               You could for example do something like             ###
        ###     starts, ends = my_algorithm_to_find_regions(sensor_data)      ###
        #########################################################################
        data_length = len(sensor_data)
        starts = [int(0.1 * data_length), int(0.5 * data_length)]  # must be a list
        ends = [int(0.4 * data_length), int(0.9 * data_length)]  # must be a list

        warnings.warn("Using exemplary labels, please find starts and ends on your own.")

        annotations = pd.DataFrame(data=[starts, ends], index = ['start', 'end']).T
        return annotations