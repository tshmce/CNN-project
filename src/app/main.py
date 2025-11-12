"""Entry point for the KivyMD application."""
from __future__ import annotations

from pathlib import Path

from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
from kivymd.app import MDApp

from src.app import screens
from src.app.state import AppState, load_state
from src.config import settings
from src.utils.logging_utils import configure_logging


class CrackDetectionApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state: AppState = load_state()
        self.screen_manager = ScreenManager()

    def build(self):
        configure_logging(settings.BASE_DIR / "src" / "config" / "logging.yaml")
        Window.minimum_width, Window.minimum_height = 1200, 700
        self.theme_cls.primary_palette = "BlueGray"
        self.title = "Crack Detection Desktop Suite"

        Builder.load_file(str(settings.BASE_DIR / "src" / "app" / "kv" / "dashboard.kv"))
        Builder.load_file(str(settings.BASE_DIR / "src" / "app" / "kv" / "training.kv"))
        Builder.load_file(str(settings.BASE_DIR / "src" / "app" / "kv" / "inference.kv"))
        Builder.load_file(str(settings.BASE_DIR / "src" / "app" / "kv" / "labeling.kv"))

        dashboard = screens.DashboardScreen(name="dashboard", state=self.state)
        training = screens.TrainingScreen(name="training", state=self.state)
        prediction = screens.PredictionScreen(name="prediction", state=self.state)
        labeling = screens.LabelingScreen(name="labeling", state=self.state)

        self.screen_manager.add_widget(dashboard)
        self.screen_manager.add_widget(training)
        self.screen_manager.add_widget(prediction)
        self.screen_manager.add_widget(labeling)

        return self.screen_manager


if __name__ == "__main__":
    CrackDetectionApp().run()
