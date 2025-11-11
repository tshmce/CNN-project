"""ResNet50 model builder."""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input

from src.models.base_model import ModelFactory

factory = ModelFactory(base_builder=ResNet50, preprocessor=preprocess_input, name="resnet50")


def build_model(num_classes: int, *, weights: str | None = "imagenet") -> keras.Model:
    return factory.build(num_classes=num_classes, weights=weights)
