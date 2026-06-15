from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
from typing import Any


class StateStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.db_path = data_dir / "state.db"
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    file_path TEXT PRIMARY KEY,
                    collection TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    last_modified REAL NOT NULL,
                    ingested_at TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    chunk_hash TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    FOREIGN KEY (file_path) REFERENCES files(file_path) ON DELETE CASCADE
                );
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(chunk_id UNINDEXED, text, collection UNINDEXED, tokenize='unicode61');
            """)
            conn.commit()

    def get_file(self, file_path: str) -> dict[str, Any] | None:
        """Retrieves file details from the state store."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT file_path, collection, file_hash, chunk_count, last_modified, ingested_at
                   FROM files WHERE file_path = ?""",
                (file_path,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "file_path": row[0],
                    "collection": row[1],
                    "file_hash": row[2],
                    "chunk_count": row[3],
                    "last_modified": row[4],
                    "ingested_at": row[5],
                }
            return None

    def get_file_chunks(self, file_path: str) -> list[dict[str, Any]]:
        """Retrieves chunks mapped to a specific file."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT chunk_id, chunk_hash, chunk_index FROM chunks WHERE file_path = ? ORDER BY chunk_index ASC",
                (file_path,),
            )
            return [
                {"chunk_id": row[0], "chunk_hash": row[1], "chunk_index": row[2]}
                for row in cursor.fetchall()
            ]

    def save_file_state(
        self,
        file_path: str,
        collection: str,
        file_hash: str,
        last_modified: float,
        chunks: list[tuple[str, str, int]],
    ) -> None:
        """Saves file metadata and its chunks in a single transaction."""
        with self._get_conn() as conn:
            # First, delete existing chunks from FTS index
            conn.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE file_path = ?)",
                (file_path,),
            )
            # Delete existing file metadata (cascade delete chunks automatically)
            conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))

            # Insert file metadata
            conn.execute(
                """INSERT INTO files (file_path, collection, file_hash, chunk_count, last_modified, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    file_path,
                    collection,
                    file_hash,
                    len(chunks),
                    last_modified,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            # Insert chunks metadata
            conn.executemany(
                "INSERT INTO chunks (chunk_id, file_path, chunk_hash, chunk_index) VALUES (?, ?, ?, ?)",
                [
                    (chunk_id, file_path, chunk_hash, chunk_index)
                    for chunk_id, chunk_hash, chunk_index in chunks
                ],
            )
            conn.commit()

    def delete_file(self, file_path: str) -> None:
        """Deletes file metadata and cascaded chunks from state store."""
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE file_path = ?)",
                (file_path,),
            )
            conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
            conn.commit()

    def save_chunks_text(self, chunks: list[tuple[str, str, str]]) -> None:
        """Saves multiple chunks text in the FTS5 virtual table at once."""
        with self._get_conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO chunks_fts (chunk_id, text, collection) VALUES (?, ?, ?)",
                chunks,
            )
            conn.commit()

    def search_fts(
        self, query: str, collection: str | None = None, top_k: int = 10
    ) -> list[tuple[str, str, float]]:
        """Returns [(chunk_id, collection, bm25_score), ...] ranked by relevance using FTS5 BM25 search."""
        with self._get_conn() as conn:
            # Simple sanitization: keep alphanumeric and spaces, replace everything else with space
            cleaned_query = re.sub(r'[^\w\s]', ' ', query).strip()
            if not cleaned_query:
                cleaned_query = query

            if collection:
                rows = conn.execute(
                    """SELECT chunk_id, collection, bm25(chunks_fts) 
                       FROM chunks_fts 
                       WHERE collection = ? AND chunks_fts MATCH ? 
                       ORDER BY bm25(chunks_fts) ASC LIMIT ?""",
                    (collection, cleaned_query, top_k)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT chunk_id, collection, bm25(chunks_fts) 
                       FROM chunks_fts 
                       WHERE chunks_fts MATCH ? 
                       ORDER BY bm25(chunks_fts) ASC LIMIT ?""",
                    (cleaned_query, top_k)
                ).fetchall()
            return [(row[0], row[1], float(row[2])) for row in rows]

    def list_all_files(self) -> list[dict[str, Any]]:
        """Lists metadata of all tracked files."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path, collection, file_hash, last_modified FROM files")
            return [
                {
                    "file_path": row[0],
                    "collection": row[1],
                    "file_hash": row[2],
                    "last_modified": row[3],
                }
                for row in cursor.fetchall()
            ]

    def list_collection_files(self, collection: str) -> list[dict[str, Any]]:
        """Lists metadata of all tracked files in a specific collection."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_path, file_hash, last_modified FROM files WHERE collection = ?",
                (collection,),
            )
            return [
                {"file_path": row[0], "file_hash": row[1], "last_modified": row[2]}
                for row in cursor.fetchall()
            ]
