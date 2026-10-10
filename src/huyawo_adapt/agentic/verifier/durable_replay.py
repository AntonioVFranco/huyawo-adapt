"""Project-local synthetic nonce registry, never an independent verifier."""

from __future__ import annotations

import os
import re
import secrets
import sqlite3
import stat
from contextlib import closing
from pathlib import Path
from typing import Literal
from urllib.parse import quote


class DurableReplayUnavailableError(RuntimeError):
    """Fail-closed synthetic replay-state storage failure."""


ConsumeResult = Literal["consumed", "replayed", "unknown", "binding_conflict"]
_ROOT = Path(__file__).resolve().parents[4]
_TRIAL_PATTERN = re.compile(r"[a-z][a-z0-9-]{0,47}\Z", re.ASCII)
_SCHEMA = 1
_SCOPE = "LOCAL_SYNTHETIC_NONPROMOTING"


def _directory(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
        if not stat.S_ISDIR(mode):
            raise DurableReplayUnavailableError("unsafe_replay_directory")
    except OSError as exc:
        raise DurableReplayUnavailableError("replay_directory_unavailable") from exc


def _regular(path: Path) -> None:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise DurableReplayUnavailableError("unsafe_replay_database")
    except OSError as exc:
        raise DurableReplayUnavailableError("replay_database_unavailable") from exc


class DurableReplayStore:
    """Explicit persistent fake state; never root-independent or rollback-proof."""

    def __init__(self, database: Path) -> None:
        self._database = database

    @classmethod
    def _paths(cls, project_root: Path, trial_id: str) -> tuple[Path, Path, Path]:
        if type(trial_id) is not str or _TRIAL_PATTERN.fullmatch(trial_id) is None:
            raise DurableReplayUnavailableError("invalid_trial_id")
        root = Path(project_root)
        if root != _ROOT or not root.is_absolute():
            raise DurableReplayUnavailableError("wrong_project_root")
        _directory(root)
        _directory(root.parent)
        parent = root / "runs"
        scope = parent / "local-synthetic-replay"
        return parent, scope, scope / trial_id

    @classmethod
    def create_new(cls, project_root: Path, trial_id: str) -> DurableReplayStore:
        parent, scope, trial = cls._paths(project_root, trial_id)
        try:
            for directory in (parent, scope):
                try:
                    directory.mkdir(mode=0o700)
                except FileExistsError:
                    _directory(directory)
            trial.mkdir(mode=0o700)
            database = trial / "replay.sqlite3"
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(database, flags, 0o600)
            os.close(fd)
            store = cls(database)
            with closing(store._connect(validate=False)) as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute(
                        "CREATE TABLE metadata ("
                        "schema_version INTEGER NOT NULL, scope TEXT NOT NULL)"
                    )
                    connection.execute("INSERT INTO metadata VALUES (?, ?)", (_SCHEMA, _SCOPE))
                    connection.execute(
                        "CREATE TABLE challenges ("
                        "nonce TEXT PRIMARY KEY NOT NULL, "
                        "task_id TEXT NOT NULL, episode_id TEXT NOT NULL, "
                        "issuer_id TEXT NOT NULL, fixture_fingerprint TEXT NOT NULL, "
                        "state TEXT NOT NULL CHECK (state IN ('issued', 'consumed')))"
                    )
                    connection.execute("COMMIT")
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
            store._verify()
            return store
        except (OSError, sqlite3.Error, DurableReplayUnavailableError) as exc:
            raise DurableReplayUnavailableError("replay_store_create_failed") from exc

    @classmethod
    def open_existing(cls, project_root: Path, trial_id: str) -> DurableReplayStore:
        parent, scope, trial = cls._paths(project_root, trial_id)
        _directory(parent)
        _directory(scope)
        _directory(trial)
        database = trial / "replay.sqlite3"
        _regular(database)
        store = cls(database)
        store._verify()
        return store

    def _connect(self, *, validate: bool = True) -> sqlite3.Connection:
        _directory(self._database.parent)
        _regular(self._database)
        try:
            uri = f"file:{quote(str(self._database), safe='/')}?mode=rw"
            connection = sqlite3.connect(uri, uri=True, timeout=0.5, isolation_level=None)
            try:
                journal = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
                connection.execute("PRAGMA synchronous=EXTRA")
                sync = connection.execute("PRAGMA synchronous").fetchone()
                if journal != ("delete",) or sync != (3,):
                    raise DurableReplayUnavailableError("replay_durability_mode_mismatch")
                if validate:
                    self._check_schema(connection)
            except BaseException:
                connection.close()
                raise
            return connection
        except sqlite3.Error as exc:
            raise DurableReplayUnavailableError("replay_database_unavailable") from exc

    @staticmethod
    def _check_schema(connection: sqlite3.Connection) -> None:
        try:
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            if integrity != ("ok",):
                raise DurableReplayUnavailableError("replay_database_corrupt")
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            if tables != [("challenges",), ("metadata",)]:
                raise DurableReplayUnavailableError("replay_schema_invalid")
            row = connection.execute("SELECT schema_version, scope FROM metadata").fetchall()
            if row != [(_SCHEMA, _SCOPE)]:
                raise DurableReplayUnavailableError("replay_schema_invalid")
            columns = connection.execute("PRAGMA table_info(challenges)").fetchall()
            if tuple(col[1] for col in columns) != (
                "nonce",
                "task_id",
                "episode_id",
                "issuer_id",
                "fixture_fingerprint",
                "state",
            ):
                raise DurableReplayUnavailableError("replay_schema_invalid")
        except sqlite3.Error as exc:
            raise DurableReplayUnavailableError("replay_schema_unavailable") from exc

    def _verify(self) -> None:
        connection = self._connect(validate=True)
        connection.close()

    def issue(self, task_id: str, episode_id: str, issuer_id: str, fixture_fingerprint: str) -> str:
        if any(
            type(s) is not str or not s.startswith("synthetic:")
            for s in (task_id, episode_id, issuer_id)
        ):
            raise DurableReplayUnavailableError("invalid_synthetic_binding")
        if type(fixture_fingerprint) is not str or len(fixture_fingerprint) != 64:
            raise DurableReplayUnavailableError("invalid_fixture_fingerprint")
        nonce = secrets.token_hex(32)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO challenges VALUES (?, ?, ?, ?, ?, 'issued')",
                    (nonce, task_id, episode_id, issuer_id, fixture_fingerprint),
                )
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        except (sqlite3.Error, DurableReplayUnavailableError) as exc:
            raise DurableReplayUnavailableError("replay_issue_not_committed") from exc
        finally:
            connection.close()
        return nonce

    def consume(
        self,
        nonce: str,
        task_id: str,
        episode_id: str,
        issuer_id: str,
        fixture_fingerprint: str,
    ) -> ConsumeResult:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT task_id, episode_id, issuer_id, fixture_fingerprint, state "
                    "FROM challenges WHERE nonce = ?",
                    (nonce,),
                ).fetchone()
                if row is None:
                    result: ConsumeResult = "unknown"
                elif row[:4] != (task_id, episode_id, issuer_id, fixture_fingerprint):
                    result = "binding_conflict"
                elif row[4] == "consumed":
                    result = "replayed"
                elif row[4] == "issued":
                    changed = connection.execute(
                        "UPDATE challenges SET state = 'consumed' "
                        "WHERE nonce = ? AND state = 'issued'",
                        (nonce,),
                    ).rowcount
                    if changed != 1:
                        raise DurableReplayUnavailableError("replay_atomicity_conflict")
                    result = "consumed"
                else:
                    raise DurableReplayUnavailableError("replay_state_invalid")
                connection.execute("COMMIT")
                return result
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        except (sqlite3.Error, DurableReplayUnavailableError) as exc:
            raise DurableReplayUnavailableError("replay_consume_not_committed") from exc
        finally:
            connection.close()
