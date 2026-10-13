"""Content fingerprints and atomic metadata writes for experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_fingerprint(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def code_fingerprint(root: Path, relative_paths: tuple[str, ...]) -> str:
    return content_fingerprint({name: file_sha256(root / name) for name in relative_paths})


def package_versions(names: tuple[str, ...]) -> dict[str, str]:
    return {name: version(name) for name in names}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)
