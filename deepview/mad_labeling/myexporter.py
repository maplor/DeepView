import warnings
from mad_gui_main.mad_gui import start_gui, BaseExporter, BaseSettings
from mad_gui_main.mad_gui.components.dialogs import UserInformation
from mad_gui_main.mad_gui.models import GlobalData

import os
from pathlib import Path
from PySide6.QtWidgets import QFileDialog

class CustomExporter(BaseExporter):
    @classmethod
    def name(cls) -> str:
        # This will be shown as string in the dropdown menu of
        # mad_gui.components.dialogs.ExportResultsDialog upon pressing
        # the button "Export data" in the GUI
        warnings.warn("Please give your exporter a meaningful name.")
        return "Custom exporter"

    def process_data(self, global_data: GlobalData):
        # Here you can do whatever you like with our global data.
        # See the API Reference for more information about our GlobalData object
        # Here is an example on how to export all annotations and their descriptions:

        directory = QFileDialog().getExistingDirectory(
            None, "Save .csv results to this folder", str(Path(global_data.data_file).parent)
        )
        for plot_name, plot_data in global_data.plot_data.items():
            for label_name, annotations in plot_data.annotations.items():
                if len(annotations.data) == 0:
                    continue
                annotations.data.to_csv(
                    directory + os.sep
                    + plot_name.replace(" ", "_")
                    + "_"
                    + label_name.replace(" ", "_")
                    + ".csv"
                )

        UserInformation.inform(f"The results were saved to {directory}.")