from PySide6 import QtWidgets

from deepview.gui import components
from deepview.gui.tabs.create_training_dataset import CreateTrainingDataset
from deepview.gui.tabs.label_with_interactive_plot import LabelWithInteractivePlotTab
from deepview.gui.tabs.supervised_cl import SupervisedCLTab
from deepview.gui.tabs.supervised_learning_new_labels import SupervisedLearningNewLabels
from deepview.gui.tabs.train_network import TrainNetwork


class WindowTabsMixin:
    def refresh_active_tab(self):
        active_tab = self.tab_widget.currentWidget()

        tab_label = self.tab_widget.tabText(self.tab_widget.currentIndex())


        widget_to_attribute_map = {
            QtWidgets.QSpinBox: "setValue",
            components.TestfileSpinBox: "setValue",
            components.TrainingSetSpinBox: "setValue",
            QtWidgets.QLineEdit: "setText",
        }

        def _attempt_attribute_update(widget_name, updated_value):
            try:
                widget = getattr(active_tab, widget_name)
                method = getattr(widget, widget_to_attribute_map[type(widget)])
                self.logger.debug(
                    f"Setting {widget_name}={updated_value} in tab '{tab_label}'"
                )
                method(updated_value)
            except AttributeError:
                pass

        _attempt_attribute_update("testfile", self.testfile)
        _attempt_attribute_update("cfg_line", self.config)

    def add_tabs(self):
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setContentsMargins(0, 20, 0, 0)
        self.create_training_dataset = CreateTrainingDataset(
            root=self,
            parent=None,
            h1_description="DeepView - Step 1. Create training dataset",
        )
        self.train_network = TrainNetwork(
            root=self, parent=None,
            h1_description="Step 2. Train unsupervised learning network",
        )

        self.label_with_interactive_plot = LabelWithInteractivePlotTab(
            root=self,
            parent=None,
            h1_description="Step 3. Label with Interaction Plot",
        )
        self.supervised_contrastive_learning = LabelWithInteractivePlotTab(
            root=self,
            parent=None,
            h1_description="Step 4. Apply Supervised Contrastive Learning",
        )
        self.supervised_learning_gui = SupervisedLearningNewLabels(
            root=self,
            parent=None,
            h1_description="Step 6. Label with Interaction Plot",
        )
        self.supervised_cl = SupervisedCLTab(
            root=self,
            parent=None,
            h1_description="Step 7. SupervisedCL",
        )

        # self.tab_widget.addTab(self.manage_project, "Manage project")
        # self.tab_widget.addTab(self.extract_frames, "Extract frames")
        # self.tab_widget.addTab(self.label_frames, "Label frames")
        self.tab_widget.addTab(self.create_training_dataset, "Create training dataset")
        self.tab_widget.addTab(self.train_network, "Train network")
        # self.tab_widget.addTab(self.evaluate_network, "Evaluate network")
        # self.tab_widget.addTab(self.mad_gui, "Label data")
        # self.tab_widget.addTab(self.interaction_plot, "Interaction plot")
        # self.tab_widget.addTab(self.show_gps, "Display GPS on the map")
        # self.tab_widget.addTab(self.imu_gps_interact, "IMU GPS interaction")
        self.tab_widget.addTab(self.label_with_interactive_plot, "Label with interactive plot")
        self.tab_widget.addTab(self.supervised_learning_gui, "Supervised learning with new labels")
        self.tab_widget.addTab(self.supervised_cl, "SupervisedCL")
        # self.tab_widget.addTab(self.analyze_videos, "Analyze videos")
        # self.tab_widget.addTab(
        #     self.unsupervised_id_tracking, "Unsupervised ID Tracking (*)"
        # )
        # self.tab_widget.addTab(self.create_videos, "Create videos")
        # self.tab_widget.addTab(
        #     self.extract_outlier_frames, "Extract outlier frames (*)"
        # )
        # self.tab_widget.addTab(self.refine_tracklets, "Refine tracklets (*)")
        # self.tab_widget.addTab(self.modelzoo, "Model Zoo")
        # self.tab_widget.addTab(self.video_editor, "Video editor (*)")

        # if not self.is_multianimal:
        #     self.refine_tracklets.setEnabled(False)
        # self.unsupervised_id_tracking.setEnabled(self.is_transreid_available())

        self.setCentralWidget(self.tab_widget)
        self.tab_widget.currentChanged.connect(self.refresh_active_tab)

    def is_transreid_available(self):
        if self.is_multianimal:
            try:
                # from deeplabcut.pose_tracking_pytorch import transformer_reID

                return True
            except ModuleNotFoundError:
                return False
        else:
            return False