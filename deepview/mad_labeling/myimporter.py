"""These are the contents of custom_importer.py, which holds my first importer."""

from typing import Dict
import warnings
import pandas as pd
# from mad_gui_main.mad_gui import BaseImporter

class CustomImporter(BaseImporter):
    loadable_file_type = "*.*"

    @classmethod
    def name(cls) -> str:
        ################################################
        ###                   README                 ###
        ### Set your importer's name as return value ###
        ### This name will show up in the dropdown.  ###
        ################################################
        warnings.warn("The importer has no meaningful name yet."
                      " Simply change the return string and remove this warning.")
        return "My Importer"

    def load_sensor_data(self, file_path: str) -> Dict:
        ##################################################################
        ###                       README                               ###
        ### a) Use the argument `file_path` to load data. Transform    ###
        ###    it to a pandas dataframe (columns are sensor channels,  ###
        ###    as for example "acc_x". Assign it to sensor_data.       ###
        ###                                                            ###
        ### b) load the sampling rate (int or float)                   ###
        ##################################################################


        warnings.warn("Please load sensor data from your source."
                      " Just make sure, that sensor_data is a pandas.DataFrame."
                      " Afterwards, remove this warning.")
        sensor_data = pd.read_csv(file_path)


        warnings.warn("Please load the sampling frequency from your source in Hz"
                      " Afterwards, remove this warning.")
        sampling_rate_hz = 1 / sensor_data["time"].diff().mean()

        ##############################################################
        ###                      CAUTION                           ###
        ### If you only want to have one plot you do not need to   ###
        ### change the following lines! If you want several plots, ###
        ### just add another sensor like "IMU foot" to the `data`  ###
        ### dictionary, which again hase keys sensor_data and      ###
        ### and sampling_rate_hz for that plot.                    ###
        ##############################################################
        data = {
           "IMU Hip": {
           "sensor_data": sensor_data[["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]],
           "sampling_rate_hz": sampling_rate_hz,
           }
        }

        return data