"""Dataset loading utilities relying on tf.data."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

from src.config import settings
from src.data_pipeline.augment import build_augmentation_pipeline, build_validation_pipeline


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@dataclass
class DatasetBundle:
    train: tf.data.Dataset
    val: tf.data.Dataset
    class_names: List[str]
    steps_per_epoch: int
    validation_steps: int


def load_datasets(
    data_dir: Path,
    image_size: Tuple[int, int] | None = None,
    batch_size: int | None = None,
    validation_split: float | None = None,
    seed: int | None = None,
) -> DatasetBundle:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory {data_dir} does not exist")

    image_size = image_size or settings.IMAGE_SIZE
    batch_size = batch_size or settings.BATCH_SIZE
    validation_split = validation_split or settings.VALIDATION_SPLIT
    seed = seed or settings.RANDOM_SEED

    _seed_everything(seed)

    common_kwargs = dict(
        directory=str(data_dir),
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
    )

    train_ds = image_dataset_from_directory(subset="training", **common_kwargs)
    val_ds = image_dataset_from_directory(subset="validation", **common_kwargs)

    class_names = list(train_ds.class_names)

    augment = build_augmentation_pipeline()
    val_aug = build_validation_pipeline()

    def _augment(image, label):
        def aug_fn(img: np.ndarray) -> np.ndarray:
            return augment(image=img)["image"]

        image = tf.numpy_function(aug_fn, [image], tf.uint8)
        image.set_shape((*image_size, 3))
        return tf.cast(image, tf.float32) / 255.0, label

    def _validate(image, label):
        def val_fn(img: np.ndarray) -> np.ndarray:
            return val_aug(image=img)["image"]

        image = tf.numpy_function(val_fn, [image], tf.uint8)
        image.set_shape((*image_size, 3))
        return tf.cast(image, tf.float32) / 255.0, label

    autotune = tf.data.AUTOTUNE
    if settings.CACHE_DATASET:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    train_ds = train_ds.map(_augment, num_parallel_calls=autotune).prefetch(settings.PREFETCH_BUFFER)
    val_ds = val_ds.map(_validate, num_parallel_calls=autotune).prefetch(settings.PREFETCH_BUFFER)

    cardinality_train = tf.data.experimental.cardinality(train_ds).numpy()
    cardinality_val = tf.data.experimental.cardinality(val_ds).numpy()
    steps_per_epoch = int(cardinality_train) if cardinality_train > 0 else 0
    validation_steps = int(cardinality_val) if cardinality_val > 0 else 0

    train_ds.class_names = class_names  # type: ignore[attr-defined]
    val_ds.class_names = class_names  # type: ignore[attr-defined]

    return DatasetBundle(
        train=train_ds,
        val=val_ds,
        class_names=class_names,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )


def iter_image_paths(data_dir: Path) -> Iterable[Path]:
    return sorted(path for path in data_dir.glob("**/*") if path.is_file())


def label_map(data_dir: Path) -> Dict[str, int]:
    classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    return {label: idx for idx, label in enumerate(classes)}
