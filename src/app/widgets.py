"""Reusable KivyMD widgets for the application."""
from __future__ import annotations

from kivy.properties import NumericProperty, StringProperty
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.progressbar import MDProgressBar


class ModelProgressBar(MDBoxLayout):
    model_name = StringProperty("")
    progress = NumericProperty(0.0)
    status_text = StringProperty("Idle")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.spacing = "8dp"
        self.padding = "8dp"
        self._label = MDLabel(text=self.model_name, bold=True)
        self._progress = MDProgressBar(value=self.progress)
        self._status = MDLabel(text=self.status_text, theme_text_color="Secondary")
        self.add_widget(self._label)
        self.add_widget(self._progress)
        self.add_widget(self._status)

    def on_model_name(self, _, value):
        self._label.text = value.upper()

    def on_progress(self, _, value):
        self._progress.value = value * 100

    def on_status_text(self, _, value):
        self._status.text = value
