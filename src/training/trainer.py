"""Multi-model training orchestration."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from queue import Queue
from typing import Dict, Iterable, List

import numpy as np
from rich.console import Console
from tensorflow import keras

from src.config import settings
from src.data_pipeline.loader import load_datasets
from src.models import efficientnet, resnet50, vgg16, vgg19
from src.training.callbacks import GradCAMCallback, TrainingProgressCallback
from src.utils.file_io import dump_json, ensure_dir

console = Console()

MODEL_BUILDERS = {
    "vgg16": vgg16.build_model,
    "vgg19": vgg19.build_model,
    "resnet50": resnet50.build_model,
    "efficientnet": efficientnet.build_model,
}


def _sorted_models(models: Iterable[str]) -> List[str]:
    return sorted(models, key=lambda name: settings.TRAINING_ORDER.get(name, 99))


def train_all(
    data_dir: Path,
    output_dir: Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    progress_queue: Queue | None = None,
    models: Iterable[str] | None = None,
) -> Dict[str, Path]:
    output_dir = output_dir or settings.MODEL_ROOT
    ensure_dir(output_dir)

    bundle = load_datasets(data_dir, batch_size=batch_size, seed=settings.RANDOM_SEED)
    num_classes = len(bundle.class_names)

    history_paths: Dict[str, Path] = {}
    selected_models = _sorted_models(models or settings.SUPPORTED_MODELS)

    for model_name in selected_models:
        builder = MODEL_BUILDERS.get(model_name)
        if not builder:
            console.print(f"[red]Unknown model: {model_name}[/red]")
            continue

        console.rule(f"Training {model_name.upper()}")
        model = builder(num_classes=num_classes)

        history_dir = ensure_dir(settings.HISTORY_DIR / model_name)
        checkpoint_dir = ensure_dir(settings.CHECKPOINT_DIR / model_name)
        timestamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        checkpoint_path = checkpoint_dir / f"{model_name}-{timestamp}.keras"

        sample_images, sample_labels = _sample_batch(bundle.val)

        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path), save_best_only=True, monitor="val_accuracy"),
            TrainingProgressCallback(model_name, on_update=progress_queue.put if progress_queue else None),
            GradCAMCallback(
                model_name=model_name,
                output_dir=ensure_dir(output_dir / "gradcam" / model_name),
                sample_images=sample_images,
                labels=sample_labels,
                class_names=bundle.class_names,
            ),
        ]

        history = model.fit(
            bundle.train,
            epochs=epochs or settings.EPOCHS,
            validation_data=bundle.val,
            callbacks=callbacks,
        )

        history_path = history_dir / f"{model_name}-{timestamp}.json"
        history_paths[model_name] = history_path
        dump_json(history_path, {"history": history.history, "class_names": bundle.class_names})

    return history_paths


def _sample_batch(dataset) -> tuple[np.ndarray, np.ndarray]:
    for images, labels in dataset.take(1):
        limit = min(len(images), settings.GRADCAM_SAMPLE_COUNT)
        return images.numpy()[:limit], labels.numpy()[:limit]
    raise RuntimeError("Validation dataset is empty; cannot sample batch for Grad-CAM generation")
