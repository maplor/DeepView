import qdarkstyle


class WindowThemeMixin:
    def darkmode(self):
        dark_stylesheet = qdarkstyle.load_stylesheet_pyside2()
        self.app.setStyleSheet(dark_stylesheet)
        try:
            self.label_with_interactive_plot.update_theme('dark')
            self.supervised_cl.update_theme('dark')
        except AttributeError:
            pass

        names = ["new_project2.png", "open2.png", "help2.png"]
        self.remove_action()
        self.create_actions(names)
        self.update_menu_bar()
        self.create_toolbar()

    def lightmode(self):
        from qdarkstyle.light.palette import LightPalette

        style = qdarkstyle.load_stylesheet(palette=LightPalette)
        self.app.setStyleSheet(style)
        try:
            self.label_with_interactive_plot.update_theme('light')
            self.supervised_cl.update_theme('light')
        except AttributeError:
            pass

        names = ["new_project.png", "open.png", "help.png"]
        self.remove_action()
        self.create_actions(names)
        self.create_toolbar()
        self.update_menu_bar()