import os

from deepview import auxiliaryfunctions
from deepview.gui.tabs import ProjectCreator
from deepview.gui.tabs.open_project import OpenProject


class WindowProjectMixin:
    @property
    def cfg(self):
        try:
            cfg = auxiliaryfunctions.read_config(self.config)
        except TypeError:
            cfg = {}
        return cfg

    @property
    def project_folder(self) -> str:
        return self.cfg.get("project_path", os.path.expanduser("~\Desktop"))

    # @property
    # def is_multianimal(self) -> bool:
    #     return bool(self.cfg.get("multianimalproject"))
    #
    # @property
    # def all_bodyparts(self) -> List:
    #     if self.is_multianimal:
    #         return self.cfg.get("multianimalbodyparts")
    #     else:
    #         return self.cfg["bodyparts"]

    # @property
    # def all_individuals(self) -> List:
    #     if self.is_multianimal:
    #         return self.cfg.get("individuals")
    #     else:
    #         return [""]

    # @property
    # def pose_cfg_path(self) -> str:
    #     try:
    #         return os.path.join(
    #             self.cfg["project_path"],
    #             auxiliaryfunctions.get_model_folder(
    #                 self.cfg["TrainingFraction"][int(self.trainingset_index)],
    #                 int(self.shuffle_value),
    #                 self.cfg,
    #             ),
    #             "train",
    #             "pose_cfg.yaml",
    #         )
    #     except FileNotFoundError:
    #         return str(Path(deepview.__file__).parent / "pose_cfg.yaml")

    @property
    def inference_cfg_path(self) -> str:
        return os.path.join(
            self.cfg["project_path"],
            auxiliaryfunctions.get_model_folder(
                self.cfg["TrainingFraction"][int(self.trainingset_index)],
                int(self.testfile),
                self.cfg,
            ),
            "test",
            "inference_cfg.yaml",
        )

    def update_cfg(self, text):
        self.root.config = text
        self.unsupervised_id_tracking.setEnabled(self.is_transreid_available())

    # def update_shuffle(self, value):
    #     self.shuffle_value = value
    #     self.logger.info(f"Shuffle set to {self.shuffle_value}")

    def update_testfile(self, value):
        self.testdata = value
        self.logger.info(f"Select test set {self.testdata}")

    @property
    def file_type(self):
        return self.file_type

    @file_type.setter
    def file_type(self, ext):
        self.filetype = ext
        self.file_type_.emit(ext)
        self.logger.info(f"File type set to {self.file_type}")

    @property
    def text_files(self):
        return self.files

    @text_files.setter
    def video_files(self, video_files):
        self.files = set(video_files)
        self.video_files_.emit(self.files)
        self.logger.info(f"Files (.csv) selected to analyze:\n{self.files}")

    def _update_project_state(self, config, loaded):
        self.config = config
        self.loaded = loaded
        if loaded:
            self.add_recent_filename(self.config)
            self.add_tabs()

    def _create_project(self):
        dlg = ProjectCreator(self)
        dlg.show()

    def _open_project(self):
        open_project = OpenProject(self)
        open_project.load_config()
        if not open_project.config:
            return

        open_project.loaded = True
        self._update_project_state(
            open_project.config,
            open_project.loaded,
        )

    # def _goto_superanimal(self):
    #     self.tab_widget = QtWidgets.QTabWidget()
    #     self.tab_widget.setContentsMargins(0, 20, 0, 0)
    #     self.modelzoo = ModelZoo(
    #         root=self, parent=None, h1_description="DeepLabCut - Model Zoo"
    #     )
    #     self.tab_widget.addTab(self.modelzoo, "Model Zoo")
    #     self.setCentralWidget(self.tab_widget)

    def load_config(self, config):
        self.config = config
        self.config_loaded.emit()
        self.logger.info('Project "%s" successfully loaded.', self.cfg["Task"])
