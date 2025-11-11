from pathlib import Path

import pytest

from src.data_pipeline.loader import label_map


def test_label_map_empty(tmp_path: Path):
    (tmp_path / "class_a").mkdir()
    (tmp_path / "class_b").mkdir()
    mapping = label_map(tmp_path)
    assert mapping == {"class_a": 0, "class_b": 1}


def test_load_datasets_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        from src.data_pipeline.loader import load_datasets

        load_datasets(Path("/non/existent/path"))
