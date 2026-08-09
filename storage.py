"""Local persistence.

Deliberately boring: a single CSV on disk. It keeps the journal portable
(open it in Excel, sync it with any cloud drive, diff it in git) and means the
app has zero database dependencies. Swap `save`/`load` for SQLite or Supabase
without touching any other module.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from schema import enrich, load_csv, normalise, to_export_frame

DATA_DIR = Path(__file__).resolve().parent / "data"
JOURNAL_PATH = DATA_DIR / "trades.csv"
SAMPLE_PATH = DATA_DIR / "sample_trades.csv"


class StorageError(RuntimeError):
    """Raised when the journal cannot be written to disk."""


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist the canonical columns only; derived fields are recomputed on load.

    Raises `StorageError` instead of crashing the app when the filesystem is
    read-only -- which is exactly what happens on Streamlit Community Cloud, so
    the UI can fall back to offering a CSV download instead.
    """
    target = Path(path) if path else JOURNAL_PATH
    try:
        ensure_dirs()
        to_export_frame(df).to_csv(target, index=False)
    except OSError as exc:
        raise StorageError(
            "Could not write to disk (hosted deployments use a read-only or "
            "ephemeral filesystem). Use Export CSV to keep your changes."
        ) from exc
    return target


def load(path: Path | None = None) -> pd.DataFrame:
    target = Path(path) if path else JOURNAL_PATH
    if not target.exists():
        return enrich(normalise(pd.DataFrame()))
    return load_csv(target)


def load_sample() -> pd.DataFrame:
    if SAMPLE_PATH.exists():
        return load_csv(SAMPLE_PATH)
    return enrich(normalise(pd.DataFrame()))


def append_trade(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    """Add one manually-entered trade and re-derive the analytics columns."""
    base = to_export_frame(df)
    new = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
    return enrich(normalise(new))
