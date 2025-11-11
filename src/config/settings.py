"""Global configuration defaults for the Crack Detection Desktop Suite."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = BASE_DIR / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
METADATA_DIR = DATA_ROOT / "metadata"
MODEL_ROOT = BASE_DIR / "models"
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"
HISTORY_DIR = MODEL_ROOT / "history"
EXPORTED_DIR = MODEL_ROOT / "exported"

SUPPORTED_MODELS: List[str] = ["vgg16", "vgg19", "resnet50", "efficientnet"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
PREFETCH_BUFFER = 16
CACHE_DATASET = True

GRADCAM_SAMPLE_COUNT = 4
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

TRAINING_ORDER: Dict[str, int] = {
    "vgg16": 0,
    "vgg19": 1,
    "resnet50": 2,
    "efficientnet": 3,
}

APP_STATE_FILE = BASE_DIR / "app_state.json"

__all__ = [
    "BASE_DIR",
    "DATA_ROOT",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "METADATA_DIR",
    "MODEL_ROOT",
    "CHECKPOINT_DIR",
    "HISTORY_DIR",
    "EXPORTED_DIR",
    "SUPPORTED_MODELS",
    "IMAGE_SIZE",
    "BATCH_SIZE",
    "EPOCHS",
    "LEARNING_RATE",
    "VALIDATION_SPLIT",
    "RANDOM_SEED",
    "PREFETCH_BUFFER",
    "CACHE_DATASET",
    "GRADCAM_SAMPLE_COUNT",
    "UMAP_N_NEIGHBORS",
    "UMAP_MIN_DIST",
    "TRAINING_ORDER",
    "APP_STATE_FILE",
]
