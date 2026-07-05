"""Utilities for downloading Lichess game archives."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen


def download_file(url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url) as response, output_path.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)

    return output_path
