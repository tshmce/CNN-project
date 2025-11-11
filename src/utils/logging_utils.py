"""Logging setup utilities."""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Optional

from .file_io import ensure_dir


def configure_logging(config_path: Path, default_level: int = logging.INFO) -> None:
    """Configure logging from a YAML config file."""
    import yaml

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        ensure_dir(Path(config_path).parent)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name if name else __name__)
