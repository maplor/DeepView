# pylint: skip-file
"""The :py:mod:`mad_gui.consts` keeps different configuration classes.
"""
from typing import Type

from mad_gui_main.mad_gui.config.settings import BaseSettings
from mad_gui_main.mad_gui.config.theme import BaseTheme


class _Config:
    """See docs `Adjusting constants <https://mad-gui.readthedocs.io/en/latest/README.html#adjusting-constants>`."""

    settings: Type[BaseSettings]
    theme: Type[BaseTheme]

    def set_theme(self, theme: Type[BaseTheme]):
        if not issubclass(theme, BaseTheme):
            raise ValueError("A new theme must subclass the default theme")
        self.theme = theme

    def set_settings(self, settings: Type[BaseSettings]):
        if not issubclass(settings, BaseSettings):
            raise ValueError("A custom settings object must subclass the BaseSettings.")
        self.settings = settings


Config = _Config()

__all__ = ["Config", "BaseTheme", "BaseSettings"]
