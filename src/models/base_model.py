"""Model factory utilities for transfer-learning architectures."""
from __future__ import annotations

from typing import Callable, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import settings


PreprocessorFn = Callable[[tf.Tensor], tf.Tensor]
ModelBuilder = Callable[..., keras.Model]


class ModelFactory:
    """Factory class to build consistent classification heads on top of base CNNs."""

    def __init__(self, base_builder: ModelBuilder, preprocessor: PreprocessorFn, name: str) -> None:
        self.base_builder = base_builder
        self.preprocessor = preprocessor
        self.name = name

    def build(
        self,
        input_shape: Tuple[int, int, int] | None = None,
        num_classes: int = 2,
        trainable_layers: int | None = None,
        weights: str | None = "imagenet",
    ) -> keras.Model:
        input_shape = input_shape or (*settings.IMAGE_SIZE, 3)

        inputs = keras.Input(shape=input_shape, name=f"{self.name}_input")
        x = self.preprocessor(inputs)
        base_model = self.base_builder(include_top=False, weights=weights, input_tensor=x)
        base_model.trainable = False

        if trainable_layers and trainable_layers > 0:
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True

        x = base_model.output
        x = layers.GlobalAveragePooling2D(name=f"{self.name}_gap")(x)
        x = layers.Dropout(0.3, name=f"{self.name}_dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name=f"{self.name}_predictions")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name=f"{self.name}_classifier")

        optimizer = keras.optimizers.Adam(learning_rate=settings.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy" if num_classes > 1 else "binary_crossentropy",
            metrics=["accuracy"],
        )
        return model


def summary(model: keras.Model) -> str:
    string_list: list[str] = []
    model.summary(print_fn=string_list.append)
    return "\n".join(string_list)
