"""EfficientNetB0 model builder."""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from src.models.base_model import ModelFactory

factory = ModelFactory(base_builder=EfficientNetB0, preprocessor=preprocess_input, name="efficientnet")


def build_model(num_classes: int, *, weights: str | None = "imagenet") -> keras.Model:
    return factory.build(num_classes=num_classes, weights=weights)
