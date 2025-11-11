"""Custom training callbacks for progress reporting and visualisation."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from tensorflow import keras

from src.evaluation.visualizations import GradCAMGenerator

ProgressCallback = Callable[[Dict[str, float]], None]


class TrainingProgressCallback(keras.callbacks.Callback):
    """Pushes per-epoch metrics to the UI via a callback."""

    def __init__(self, model_name: str, on_update: ProgressCallback | None = None) -> None:
        super().__init__()
        self.model_name = model_name
        self.on_update = on_update

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] | None = None) -> None:
        if logs and self.on_update:
            payload = {"model": self.model_name, "epoch": epoch + 1}
            payload.update({k: float(v) for k, v in logs.items()})
            self.on_update(payload)


class GradCAMCallback(keras.callbacks.Callback):
    """Generates Grad-CAM overlays for a batch of validation images."""

    def __init__(
        self,
        model_name: str,
        output_dir: Path,
        sample_images: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
    ) -> None:
        super().__init__()
        self.generator = GradCAMGenerator(model_name, output_dir, class_names)
        self.sample_images = sample_images
        self.labels = labels

    def on_train_end(self, logs: Dict[str, float] | None = None) -> None:
        self.generator.generate(self.model, self.sample_images, self.labels)
