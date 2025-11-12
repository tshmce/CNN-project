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
from kivy.uix.image import Image
import shutil

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

    def open_training(self):
        self.manager.current = "training"

    def open_prediction(self):
        self.manager.current = "prediction"
        


class TrainingScreen(Screen):
    state: AppState = ObjectProperty(None)
    progress_data = DictProperty({model: {"progress": 0.0, "status": "Idle"} for model in settings.SUPPORTED_MODELS})
    _worker_thread: WorkerThread | None = None
    file_manager: MDFileManager | None = None
    current_label: str = "Positive"

    def on_pre_enter(self):
        if self.state.progress_queue is None:
            self.state.progress_queue = Queue()
            Clock.schedule_interval(self._poll_queue, 0.5)
        # Ensure default dataset root is set
        if not self.state.dataset_dir:
            self.state.dataset_dir = settings.RAW_DATA_DIR

    def set_label_positive(self):
        self.current_label = "Positive"
        self.ids.current_label.text = f"Selected label: {self.current_label}"

    def set_label_negative(self):
        self.current_label = "Negative"
        self.ids.current_label.text = f"Selected label: {self.current_label}"

    def choose_images_folder(self):
        if not self.file_manager:
            self.file_manager = MDFileManager(select_path=self._ingest_folder, exit_manager=self._close_manager)
        self.file_manager.show(settings.DATA_ROOT.as_posix())

    def _close_manager(self, *args):
        if self.file_manager:
            self.file_manager.close()

    def _ingest_folder(self, path: str):
        src_dir = Path(path)
        if not src_dir.exists() or not src_dir.is_dir():
            MDDialog(title="Training", text="Please select a valid folder.").open()
            self._close_manager()
            return
        dest_dir = (self.state.dataset_dir or settings.RAW_DATA_DIR) / self.current_label
        dest_dir.mkdir(parents=True, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        count = 0
        for p in src_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                dest = dest_dir / p.name
                i = 1
                while dest.exists():
                    dest = dest_dir / f"{p.stem}_{i}{p.suffix}"
                    i += 1
                try:
                    shutil.copy2(p.as_posix(), dest.as_posix())
                    count += 1
                except Exception:
                    pass
        self.ids.ingest_status.text = f"Copied {count} images to {dest_dir.name}"
        self._close_manager()

    def start_training(self):
        # Default to RAW_DATA_DIR if not explicitly set
        if not self.state.dataset_dir:
            self.state.dataset_dir = settings.RAW_DATA_DIR
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
    file_manager: MDFileManager | None = None
    selected_image_path: Path | None = None

    def on_pre_enter(self):
        self.ids.selected_image.text = "No image selected"

    def choose_image(self):
        if not self.file_manager:
            self.file_manager = MDFileManager(select_path=self._select_image, exit_manager=self._close_manager)
        # Let user browse anywhere under BASE_DIR by default
        self.file_manager.show(settings.BASE_DIR.as_posix())

    def _select_image(self, path: str):
        self.selected_image_path = Path(path)
        self.ids.selected_image.text = path
        self._close_manager()

    def _close_manager(self, *args):
        if self.file_manager:
            self.file_manager.close()

    def run_prediction(self):
        if not self.selected_image_path or not self.selected_image_path.exists():
            MDDialog(title="Prediction", text="Please choose an image to predict.").open()
            return
        try:
            from tensorflow import keras
        except Exception as exc:
            MDDialog(title="Prediction", text=f"TensorFlow not available: {exc}").open()
            return

        def _latest_checkpoint(model_name: str) -> Path | None:
            ckpt_dir = settings.CHECKPOINT_DIR / model_name
            if not ckpt_dir.exists():
                return None
            candidates = sorted(ckpt_dir.glob("*.keras"))
            return candidates[-1] if candidates else None

        # Pick the first available model with a checkpoint
        chosen_model = None
        ckpt_path = None
        for name in settings.SUPPORTED_MODELS:
            p = _latest_checkpoint(name)
            if p:
                chosen_model, ckpt_path = name, p
                break
        if not ckpt_path:
            MDDialog(title="Prediction", text="No trained checkpoints found. Train the model first.").open()
            return

        try:
            model = keras.models.load_model(ckpt_path)
        except Exception as exc:
            MDDialog(title="Prediction", text=f"Failed to load model: {exc}").open()
            return

        # Load and preprocess single image
        from tensorflow.keras.utils import load_img, img_to_array
        img = load_img(self.selected_image_path.as_posix(), target_size=settings.IMAGE_SIZE)
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        probs = model.predict(arr, verbose=0)[0]
        classes = ["Negative", "Positive"] if len(probs) == 2 else [str(i) for i in range(len(probs))]
        pred_idx = int(np.argmax(probs))
        pred_label = classes[pred_idx]
        pred_conf = float(probs[pred_idx])

        MDDialog(title="Prediction", text=f"Model: {chosen_model.upper()}\nLabel: {pred_label}\nConfidence: {pred_conf:.3f}").open()

    def _open_artifacts(self, model_name: str, artifacts: Iterable[PlotArtifact]):
        dialog_text = "\n".join(artifact.path.as_posix() for artifact in artifacts)
        MDDialog(title=f"Artifacts for {model_name.upper()}", text=dialog_text).open()

    def back_to_dashboard(self):
        self.manager.current = "dashboard"


class LabelingScreen(Screen):
    state: AppState = ObjectProperty(None)
    file_manager: MDFileManager | None = None
    unlabeled_dir: Path | None = None
    _images: list[Path] = []
    _index: int = 0

    def on_pre_enter(self):
        # Reset any status text
        if hasattr(self.ids, "unlabeled_label"):
            self.ids.unlabeled_label.text = "No unlabeled folder selected"
        if hasattr(self.ids, "image_view"):
            self.ids.image_view.source = ""

    def choose_unlabeled(self):
        if not self.state.dataset_dir:
            MDDialog(title="Labeling", text="Select a dataset from the dashboard first.").open()
            return
        if not self.file_manager:
            self.file_manager = MDFileManager(select_path=self._select_unlabeled, exit_manager=self._close_manager)
        self.file_manager.show(settings.DATA_ROOT.as_posix())

    def _select_unlabeled(self, path: str):
        self.unlabeled_dir = Path(path)
        self._images = self._gather_images(self.unlabeled_dir)
        self._index = 0
        self.ids.unlabeled_label.text = f"Unlabeled: {path} ({len(self._images)} images)"
        self._show_current()
        self._close_manager()

    def _close_manager(self, *args):
        if self.file_manager:
            self.file_manager.close()

    def _gather_images(self, root: Path) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])

    def _show_current(self):
        if not self._images:
            self.ids.image_view.source = ""
            MDDialog(title="Labeling", text="No images found in selected folder.").open()
            return
        if self._index >= len(self._images):
            MDDialog(title="Labeling", text="All images labeled.").open()
            self._index = len(self._images) - 1
        self.ids.image_view.source = self._images[self._index].as_posix()

    def _label_current(self, label: str):
        if not self.state.dataset_dir or not self._images:
            return
        src = self._images[self._index]
        dest_dir = self.state.dataset_dir / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        # Ensure unique name if conflict
        counter = 1
        while dest.exists():
            dest = dest_dir / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        try:
            shutil.move(src.as_posix(), dest.as_posix())
        except Exception as exc:
            MDDialog(title="Labeling", text=f"Move failed: {exc}").open()
            return
        # Remove from list and advance
        del self._images[self._index]
        if self._index >= len(self._images):
            self._index = max(0, len(self._images) - 1)
        self._show_current()

    def mark_positive(self):
        self._label_current("Positive")

    def mark_negative(self):
        self._label_current("Negative")

    def skip(self):
        if not self._images:
            return
        self._index = (self._index + 1) % len(self._images)
        self._show_current()

    def back_to_dashboard(self):
        self.manager.current = "dashboard"
