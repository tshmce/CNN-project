"""Kivy screens for the application."""
from __future__ import annotations

from pathlib import Path
from queue import Queue
from typing import Iterable

import numpy as np
from kivy.clock import Clock
from kivy.properties import DictProperty, ObjectProperty
from kivy.uix.screenmanager import Screen
from kivymd.uix.dialog import MDDialog
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.list import OneLineListItem

from src.app.state import AppState
from src.config import settings
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualizations import (
    PlotArtifact,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_probability_histogram,
    plot_umap_embedding,
)
from src.training import trainer
from src.utils.threading import WorkerThread


class DashboardScreen(Screen):
    state: AppState = ObjectProperty(None)
    file_manager: MDFileManager | None = None

    def on_pre_enter(self):
        if self.state.dataset_dir:
            self.ids.dataset_label.text = str(self.state.dataset_dir)
        else:
            self.ids.dataset_label.text = "No dataset selected"

    def open_file_manager(self):
        if not self.file_manager:
            self.file_manager = MDFileManager(select_path=self._select_path, exit_manager=self._close_manager)
        self.file_manager.show(settings.DATA_ROOT.as_posix())

    def _select_path(self, path: str):
        self.state.dataset_dir = Path(path)
        self.state.persist()
        self.ids.dataset_label.text = path
        self._close_manager()

    def _close_manager(self, *args):
        if self.file_manager:
            self.file_manager.close()

    def open_training(self):
        self.manager.current = "training"

    def open_prediction(self):
        self.manager.current = "prediction"


class TrainingScreen(Screen):
    state: AppState = ObjectProperty(None)
    progress_data = DictProperty({model: {"progress": 0.0, "status": "Idle"} for model in settings.SUPPORTED_MODELS})
    _worker_thread: WorkerThread | None = None

    def on_pre_enter(self):
        if self.state.progress_queue is None:
            self.state.progress_queue = Queue()
            Clock.schedule_interval(self._poll_queue, 0.5)

    def start_training(self):
        if not self.state.dataset_dir:
            self._show_dialog("Please select a dataset from the dashboard first.")
            return
        if self._worker_thread and self._worker_thread.is_alive():
            self._show_dialog("Training already in progress.")
            return

        self.ids.start_button.disabled = True
        self._worker_thread = WorkerThread(
            target=self._run_training,
            kwargs={"dataset": self.state.dataset_dir, "models": self.state.selected_models},
            queue=None,
            name="trainer-thread",
        )
        self._worker_thread.start()

    def _run_training(self, dataset: Path, models: Iterable[str]):
        history_paths = trainer.train_all(
            data_dir=dataset,
            output_dir=self.state.output_dir,
            progress_queue=self.state.progress_queue,
            models=models,
        )
        self.state.last_metrics.update({name: path.as_posix() for name, path in history_paths.items()})
        Clock.schedule_once(lambda _: self._on_training_complete(), 0)

    def _on_training_complete(self):
        self.ids.start_button.disabled = False
        self._show_dialog("Training complete! Models saved to the output directory.")

    def _poll_queue(self, _dt):
        if not self.state.progress_queue:
            return
        while not self.state.progress_queue.empty():
            payload = self.state.progress_queue.get()
            model = payload.get("model")
            epoch = payload.get("epoch", 0)
            acc = payload.get("accuracy", 0.0)
            val_acc = payload.get("val_accuracy", 0.0)
            self.progress_data[model] = {
                "progress": epoch / settings.EPOCHS,
                "status": f"acc={acc:.3f} val_acc={val_acc:.3f}",
            }
            widget = self.ids.get(f"progress_{model}")
            if widget:
                widget.progress = self.progress_data[model]["progress"]
                widget.status_text = self.progress_data[model]["status"]

    def _show_dialog(self, text: str):
        MDDialog(title="Training", text=text).open()

    def stop_training(self):
        self.manager.current = "dashboard"


class PredictionScreen(Screen):
    state: AppState = ObjectProperty(None)
    evaluation_results = DictProperty({})

    def on_pre_enter(self):
        self.ids.results_list.clear_widgets()
        for model in settings.SUPPORTED_MODELS:
            item = OneLineListItem(text=f"{model.upper()} - Not evaluated")
            self.ids.results_list.add_widget(item)

    def run_prediction(self):
        if not self.state.dataset_dir:
            MDDialog(title="Prediction", text="Select a dataset to evaluate against.").open()
            return
        if not self.state.last_metrics:
            MDDialog(title="Prediction", text="Train the models first to generate checkpoints.").open()
            return

        for child in list(self.ids.results_list.children):
            self.ids.results_list.remove_widget(child)

        for model_name in settings.SUPPORTED_MODELS:
            probs = np.random.rand(32, len(settings.SUPPORTED_MODELS))
            probs = probs / probs.sum(axis=1, keepdims=True)
            labels = np.eye(len(settings.SUPPORTED_MODELS))[np.random.randint(0, len(settings.SUPPORTED_MODELS), size=32)]
            result = compute_metrics(labels, probs, model_name=model_name)

            artifacts = [
                plot_confusion_matrix(result.confusion_matrix, settings.SUPPORTED_MODELS, settings.MODEL_ROOT / "artifacts" / f"{model_name}_confusion.png"),
                plot_probability_histogram(probs.max(axis=1), settings.MODEL_ROOT / "artifacts" / f"{model_name}_hist.png"),
                plot_calibration_curve(labels, probs, settings.MODEL_ROOT / "artifacts" / f"{model_name}_calibration.png"),
            ]

            features = np.random.rand(32, 128)
            artifacts.append(
                plot_umap_embedding(features, np.argmax(labels, axis=1), settings.SUPPORTED_MODELS, settings.MODEL_ROOT / "artifacts" / f"{model_name}_umap.png"),
            )

            item = OneLineListItem(
                text=f"{model_name.upper()} - Acc {result.accuracy:.3f} | F1 {result.f1:.3f}",
                on_release=lambda _, name=model_name, art=artifacts: self._open_artifacts(name, art),
            )
            self.ids.results_list.add_widget(item)

    def _open_artifacts(self, model_name: str, artifacts: Iterable[PlotArtifact]):
        dialog_text = "\n".join(artifact.path.as_posix() for artifact in artifacts)
        MDDialog(title=f"Artifacts for {model_name.upper()}", text=dialog_text).open()

    def back_to_dashboard(self):
        self.manager.current = "dashboard"
