"""VGG19 model builder."""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

from src.models.base_model import ModelFactory

factory = ModelFactory(base_builder=VGG19, preprocessor=preprocess_input, name="vgg19")


def build_model(num_classes: int, *, weights: str | None = "imagenet") -> keras.Model:
    return factory.build(num_classes=num_classes, weights=weights)
