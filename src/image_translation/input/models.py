"""Input models – clean typed application model for resolved inputs."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional


class InputType(str, Enum):
    FOLDER = "folder"
    SINGLE_IMAGE = "single_image"


class AppInput:
    """Resolved application input – the single source of truth for paths."""

    __slots__ = (
        "input_path",
        "input_type",
        "input_folder",
        "single_image_path",
        "output_folder",
        "config_path",
    )

    def __init__(
        self,
        input_path: Path,
        input_type: InputType,
        input_folder: Optional[Path],
        single_image_path: Optional[Path],
        output_folder: Path,
        config_path: Optional[Path],
    ) -> None:
        self.input_path = input_path
        self.input_type = input_type
        self.input_folder = input_folder
        self.single_image_path = single_image_path
        self.output_folder = output_folder
        self.config_path = config_path

    def __repr__(self) -> str:
        return (
            f"AppInput(type={self.input_type.value}, "
            f"input={self.input_path}, output={self.output_folder})"
        )
