from pathlib import Path

import numpy as np

from src.evaluation.visualizations import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_probability_histogram,
)


def test_confusion_matrix_plot(tmp_path: Path):
    confusion = np.array([[5, 1], [2, 4]])
    artifact = plot_confusion_matrix(confusion, ["A", "B"], tmp_path / "confusion.png")
    assert artifact.path.exists()


def test_probability_histogram(tmp_path: Path):
    probabilities = np.random.rand(100)
    artifact = plot_probability_histogram(probabilities, tmp_path / "hist.png")
    assert artifact.path.exists()


def test_calibration_curve(tmp_path: Path):
    y_true = np.eye(2)[np.random.randint(0, 2, size=50)]
    y_pred = np.random.rand(50, 2)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    artifact = plot_calibration_curve(y_true, y_pred, tmp_path / "calibration.png")
    assert artifact.path.exists()
