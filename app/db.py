"""Database setup using SQLAlchemy."""
from __future__ import annotations

from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine

from app.models import Base


def _set_wal_mode(dbapi_connection, connection_record):  # noqa: ARG001
    """Enable WAL mode for better concurrent read performance."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine(db_path: Path):
    """Create a SQLAlchemy engine for the given SQLite database path."""
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    event.listen(engine, "connect", _set_wal_mode)
    Base.metadata.create_all(engine)
    return engine


def get_session(db_path: Path) -> Session:
    """Return a new SQLAlchemy session."""
    engine = get_engine(db_path)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
