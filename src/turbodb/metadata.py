"""SQLite metadata store for collection IDs, metadata, and positions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

__all__ = ["MetadataStore"]


class MetadataStore:
    """SQLite-backed metadata store for a single collection."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                position INTEGER NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_position ON vectors(position);
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

    def insert(self, id: str, position: int, metadata: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO vectors (id, position, metadata) VALUES (?, ?, ?)",
            (id, position, json.dumps(metadata)),
        )
        self._conn.commit()

    def insert_batch(self, rows: list[tuple[str, int, dict[str, Any]]]) -> None:
        self._conn.executemany(
            "INSERT INTO vectors (id, position, metadata) VALUES (?, ?, ?)",
            [(id, pos, json.dumps(meta)) for id, pos, meta in rows],
        )
        self._conn.commit()

    def upsert_batch(self, rows: list[tuple[str, int, dict[str, Any]]]) -> list[int]:
        """Insert or replace rows. Returns positions of replaced rows."""
        replaced_positions = []
        for id, pos, meta in rows:
            row = self._conn.execute(
                "SELECT position FROM vectors WHERE id = ?", (id,)
            ).fetchone()
            if row is not None:
                replaced_positions.append(row["position"])
                self._conn.execute(
                    "UPDATE vectors SET position = ?, metadata = ? WHERE id = ?",
                    (pos, json.dumps(meta), id),
                )
            else:
                self._conn.execute(
                    "INSERT INTO vectors (id, position, metadata) VALUES (?, ?, ?)",
                    (id, pos, json.dumps(meta)),
                )
        self._conn.commit()
        return replaced_positions

    def get_by_id(self, id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, position, metadata FROM vectors WHERE id = ?", (id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "position": row["position"],
            "metadata": json.loads(row["metadata"]),
        }

    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        placeholders = ", ".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, position, metadata FROM vectors WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return [
            {"id": r["id"], "position": r["position"], "metadata": json.loads(r["metadata"])}
            for r in rows
        ]

    def delete_by_ids(self, ids: list[str]) -> list[int]:
        placeholders = ", ".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT position FROM vectors WHERE id IN ({placeholders})", ids
        ).fetchall()
        positions = [r["position"] for r in rows]
        self._conn.execute(
            f"DELETE FROM vectors WHERE id IN ({placeholders})", ids
        )
        self._conn.commit()
        return positions

    def delete_by_filter(self, where_clause: str, params: list) -> list[int]:
        rows = self._conn.execute(
            f"SELECT position FROM vectors WHERE {where_clause}", params
        ).fetchall()
        positions = [r["position"] for r in rows]
        if positions:
            self._conn.execute(
                f"DELETE FROM vectors WHERE {where_clause}", params
            )
            self._conn.commit()
        return positions

    def get_positions_by_filter(self, where_clause: str, params: list) -> list[int]:
        rows = self._conn.execute(
            f"SELECT position FROM vectors WHERE {where_clause}", params
        ).fetchall()
        return [r["position"] for r in rows]

    def get_by_positions(self, positions: list[int]) -> list[dict[str, Any]]:
        placeholders = ", ".join("?" for _ in positions)
        rows = self._conn.execute(
            f"SELECT id, position, metadata FROM vectors WHERE position IN ({placeholders})"
            f" ORDER BY position",
            positions,
        ).fetchall()
        return [
            {"id": r["id"], "position": r["position"], "metadata": json.loads(r["metadata"])}
            for r in rows
        ]

    def get_all_live_positions(self) -> list[int]:
        rows = self._conn.execute("SELECT position FROM vectors ORDER BY position").fetchall()
        return [r["position"] for r in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as n FROM vectors").fetchone()
        return row["n"]

    def max_position(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(position), -1) as m FROM vectors").fetchone()
        return row["m"]

    def set_config(self, config: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES ('collection_config', ?)",
            (json.dumps(config),),
        )
        self._conn.commit()

    def get_config(self) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT value FROM config WHERE key = 'collection_config'"
        ).fetchone()
        if row is None:
            return {}
        return json.loads(row["value"])

    def rewrite_positions(self, position_map: dict[int, int]) -> None:
        """Remap positions during compaction."""
        for old_pos, new_pos in position_map.items():
            self._conn.execute(
                "UPDATE vectors SET position = ? WHERE position = ?",
                (new_pos, old_pos),
            )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
