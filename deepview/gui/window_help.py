from PySide6.QtWidgets import QMessageBox


class WindowHelpMixin:
    def _ask_for_help(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Ask for help")
        dlg.setText(
            """Ask our community for help on <a href='https://forum.image.sc/tag/deepview'>the forum</a>!"""
        )
        _ = dlg.exec()

    def _learn_dlc(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Learn DLC")
        dlg.setText(
            """Learn DLC with <a href='https://deepview.github.io/UseOverviewGuide.html'>our docs and how-to guides</a>!"""
        )
        _ = dlg.exec()