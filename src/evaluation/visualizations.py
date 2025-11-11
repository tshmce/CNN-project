"""Visualisation helpers for evaluation outputs."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import umap
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve

from src.utils.file_io import ensure_dir

sns.set_style("whitegrid")


@dataclass
class PlotArtifact:
    name: str
    path: Path


def _save_figure(fig: Figure, output_path: Path) -> PlotArtifact:
    ensure_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return PlotArtifact(name=output_path.stem, path=output_path)


def plot_confusion_matrix(confusion: np.ndarray, class_names: List[str], output_path: Path) -> PlotArtifact:
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    return _save_figure(fig, output_path)


def plot_probability_histogram(probabilities: np.ndarray, output_path: Path) -> PlotArtifact:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(probabilities, bins=20, alpha=0.7)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Frequency")
    ax.set_title("Probability Histogram")
    return _save_figure(fig, output_path)


def plot_calibration_curve(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> PlotArtifact:
    true_labels = np.argmax(y_true, axis=1)
    confidences = np.max(y_pred, axis=1)
    prob_true, prob_pred = calibration_curve(true_labels, confidences, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted value")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    return _save_figure(fig, output_path)


def plot_umap_embedding(features: np.ndarray, labels: np.ndarray, class_names: List[str], output_path: Path) -> PlotArtifact:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean")
    embedding = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", alpha=0.8)
    legend_handles = scatter.legend_elements()[0]
    ax.legend(legend_handles, class_names, title="Classes")
    ax.set_title("UMAP Embedding")
    ax.set_xticks([])
    ax.set_yticks([])
    return _save_figure(fig, output_path)


def plot_phash_distances(distances: Dict[str, float], output_path: Path) -> PlotArtifact:
    fig, ax = plt.subplots(figsize=(6, 4))
    items = sorted(distances.items(), key=lambda item: item[1])
    labels, values = zip(*items) if items else ([], [])
    ax.barh(labels, values)
    ax.set_xlabel("Hamming distance")
    ax.set_title("Duplicate/Leakage pHash Distances")
    return _save_figure(fig, output_path)


class GradCAMGenerator:
    """Generate Grad-CAM overlays for a batch of images."""

    def __init__(self, model_name: str, output_dir: Path, class_names: List[str]) -> None:
        self.model_name = model_name
        self.output_dir = ensure_dir(output_dir)
        self.class_names = class_names

    def _make_gradcam_heatmap(self, image_tensor: tf.Tensor, model: tf.keras.Model) -> np.ndarray:
        target_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer.output, tf.Tensor) and len(layer.output.shape) == 4:
                target_layer = layer
                break
        if target_layer is None:
            raise ValueError("Could not find a 4D convolutional layer for Grad-CAM.")
        grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_tensor)
            loss = predictions[:, tf.argmax(predictions[0])]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    def generate(self, model: tf.keras.Model, images: np.ndarray, labels: np.ndarray) -> List[Path]:
        output_paths: List[Path] = []
        for idx, (image, label) in enumerate(zip(images, labels)):
            input_tensor = tf.expand_dims(image, axis=0)
            heatmap = self._make_gradcam_heatmap(input_tensor, model)
            heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], (image.shape[0], image.shape[1]))
            heatmap_resized = tf.squeeze(heatmap_resized)

            fig, ax = plt.subplots(figsize=(4, 4))
            base_image = image.astype("float32")
            if base_image.max() > 1.0:
                base_image = base_image / 255.0
            ax.imshow(base_image)
            ax.imshow(heatmap_resized, cmap="jet", alpha=0.4)
            label_idx = int(np.argmax(label))
            ax.set_title(f"{self.class_names[label_idx]}")
            ax.axis("off")
            output_path = self.output_dir / f"gradcam_{self.model_name}_{idx}.png"
            artifact = _save_figure(fig, output_path)
            output_paths.append(artifact.path)
        return output_paths
