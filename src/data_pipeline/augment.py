"""Data augmentation utilities built on top of Albumentations."""
from __future__ import annotations

from typing import Callable

import albumentations as A
import numpy as np

from src.config.settings import IMAGE_SIZE


def build_augmentation_pipeline() -> A.BasicTransform:
    return A.Compose(
        [
            A.RandomResizedCrop(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1], scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(p=0.1),
            A.MotionBlur(p=0.1),
        ]
    )


def build_validation_pipeline() -> A.BasicTransform:
    return A.Compose([A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1])])


def apply_augmentation(image: np.ndarray, pipeline_factory: Callable[[], A.BasicTransform] | None = None) -> np.ndarray:
    pipeline = pipeline_factory() if pipeline_factory else build_validation_pipeline()
    augmented = pipeline(image=image)
    return augmented["image"]
