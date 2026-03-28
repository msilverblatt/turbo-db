"""TurboDB: database manager for collections."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from turbodb.collection import Collection
from turbodb.exceptions import CollectionExistsError, CollectionNotFoundError

__all__ = ["TurboDB"]

_DB_CONFIG_FILE = "turbodb.json"


class TurboDB:
    """Embedded vector database backed by TurboQuant compression."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        config_path = self._path / _DB_CONFIG_FILE
        if not config_path.exists():
            config_path.write_text(json.dumps({"version": "0.1.0"}, indent=2))

    def create_collection(
        self,
        name: str,
        dim: int,
        metric: str = "cosine",
        bit_width: int = 2,
    ) -> Collection:
        col_path = self._path / name
        if col_path.exists():
            raise CollectionExistsError(name)
        return Collection.create(
            path=col_path,
            name=name,
            dim=dim,
            bit_width=bit_width,
            metric=metric,
        )

    def get_collection(self, name: str) -> Collection:
        col_path = self._path / name
        if not col_path.exists():
            raise CollectionNotFoundError(name)
        return Collection.open(col_path, name)

    def get_or_create_collection(
        self,
        name: str,
        dim: int,
        metric: str = "cosine",
        bit_width: int = 2,
    ) -> Collection:
        col_path = self._path / name
        if col_path.exists():
            return Collection.open(col_path, name)
        return Collection.create(
            path=col_path,
            name=name,
            dim=dim,
            bit_width=bit_width,
            metric=metric,
        )

    def delete_collection(self, name: str) -> None:
        col_path = self._path / name
        if not col_path.exists():
            raise CollectionNotFoundError(name)
        shutil.rmtree(col_path)

    def list_collections(self) -> list[str]:
        return [
            p.name
            for p in self._path.iterdir()
            if p.is_dir() and (p / "metadata.db").exists()
        ]
