"""Application state shared between screens."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Dict, Optional

from src.config import settings
from src.utils.file_io import dump_json, load_json


@dataclass
class AppState:
    dataset_dir: Path | None = None
    output_dir: Path = settings.MODEL_ROOT
    selected_models: list[str] = field(default_factory=lambda: list(settings.SUPPORTED_MODELS))
    progress_queue: Queue | None = None
    last_metrics: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        return {
            "dataset_dir": self.dataset_dir.as_posix() if self.dataset_dir else "",
            "output_dir": self.output_dir.as_posix(),
            "selected_models": self.selected_models,
        }

    def persist(self) -> None:
        dump_json(settings.APP_STATE_FILE, self.to_dict())


def load_state() -> AppState:
    payload = load_json(settings.APP_STATE_FILE, default={})
    dataset_dir = Path(payload.get("dataset_dir", "")) if payload.get("dataset_dir") else None
    output_dir = Path(payload.get("output_dir", settings.MODEL_ROOT.as_posix()))
    selected_models = payload.get("selected_models", list(settings.SUPPORTED_MODELS))
    return AppState(dataset_dir=dataset_dir, output_dir=output_dir, selected_models=selected_models)
