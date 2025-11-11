"""Duplicate/leakage detection utilities using perceptual hashes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import imagehash
from PIL import Image

from src.utils.file_io import ensure_dir


@dataclass
class HashRecord:
    path: Path
    phash: str


def compute_phash(image_path: Path) -> HashRecord:
    with Image.open(image_path) as img:
        hash_val = imagehash.phash(img)
    return HashRecord(path=image_path, phash=str(hash_val))


def build_phash_table(image_paths: Iterable[Path]) -> List[HashRecord]:
    return [compute_phash(path) for path in image_paths]


def save_phash_table(records: List[HashRecord], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("path,phash\n")
        for record in records:
            handle.write(f"{record.path},{record.phash}\n")
