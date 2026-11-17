# CS5491: Reasoning Augmented FunSearch
# Author: 朱泳颐 (72657023)
# Responsibility: Search framework and logging

"""Atomic, crash-safe attempt accounting. Reservations are never recycled."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path, global_cap=2200):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.global_cap = global_cap
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS attempts(
                id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL, phase TEXT NOT NULL, created TEXT NOT NULL,
                metadata TEXT NOT NULL, response TEXT, event TEXT,
                UNIQUE(run_id,ordinal));
            CREATE TABLE IF NOT EXISTS states(run_id TEXT PRIMARY KEY,state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=60)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=60000")
        return db

    def count(self, run_id=None, phase=None):
        clauses, args = [], []
        if run_id is not None:
            clauses.append("run_id=?")
            args.append(run_id)
        if phase is not None:
            clauses.append("phase=?")
            args.append(phase)
        with self.connect() as db:
            return db.execute(
                "SELECT count(*) FROM attempts"
                + (" WHERE " + " AND ".join(clauses) if clauses else ""),
                args,
            ).fetchone()[0]

    def reserve(self, run_id, phase, run_cap, metadata):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            total = db.execute("SELECT count(*) FROM attempts").fetchone()[0]
            used = db.execute(
                "SELECT count(*) FROM attempts WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            development = db.execute(
                "SELECT count(*) FROM attempts WHERE phase='development'"
            ).fetchone()[0]
            if (
                total >= self.global_cap
                or used >= run_cap
                or (phase == "development" and development >= 200)
            ):
                raise BudgetExceeded("Physical attempt budget exhausted")
            cursor = db.execute(
                "INSERT INTO attempts(run_id,ordinal,phase,created,metadata) VALUES(?,?,?,?,?)",
                (run_id, used + 1, phase, utcnow(), json.dumps(metadata)),
            )
            return cursor.lastrowid, used + 1

    def save_response(self, attempt_id, response):
        with self.connect() as db:
            old = db.execute(
                "SELECT response FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()[0]
            if old is not None:
                if json.loads(old) != response:
                    raise ValueError("Attempt responses are immutable")
                return
            db.execute(
                "UPDATE attempts SET response=? WHERE id=?",
                (json.dumps(response), attempt_id),
            )

    def finalize(self, attempt_id, event, state):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT event,run_id FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if row["event"] is not None:
                raise ValueError("Attempt already finalized")
            db.execute(
                "UPDATE attempts SET event=? WHERE id=?",
                (json.dumps(event), attempt_id),
            )
            db.execute(
                "INSERT INTO states VALUES(?,?) ON CONFLICT(run_id) DO UPDATE SET state=excluded.state",
                (row["run_id"], json.dumps(state)),
            )

    def save_state(self, run_id, state):
        with self.connect() as db:
            db.execute(
                "INSERT INTO states VALUES(?,?) ON CONFLICT(run_id) DO UPDATE SET state=excluded.state",
                (run_id, json.dumps(state)),
            )

    def get_state(self, run_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT state FROM states WHERE run_id=?", (run_id,)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def attempts(self, run_id):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM attempts WHERE run_id=? ORDER BY ordinal", (run_id,)
            ).fetchall()
        return [
            {
                **dict(r),
                **{
                    k: json.loads(r[k]) if r[k] else None
                    for k in ("metadata", "response", "event")
                },
            }
            for r in rows
        ]

    def cache_get(self, key):
        with self.connect() as db:
            row = db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def cache_put(self, key, value):
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO cache VALUES(?,?)", (key, json.dumps(value))
            )

    def reconciliation(self):
        with self.connect() as db:
            rows = db.execute(
                "SELECT run_id,phase,count(*) attempts,sum(response IS NOT NULL) responses,sum(event IS NOT NULL) finalized FROM attempts GROUP BY run_id,phase ORDER BY run_id"
            ).fetchall()
        return {
            "global_cap": self.global_cap,
            "reserved_attempts": sum(r["attempts"] for r in rows),
            "remaining": self.global_cap - sum(r["attempts"] for r in rows),
            "runs": [dict(r) for r in rows],
        }
