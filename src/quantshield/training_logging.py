"""Structured training-event emission and metadata helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EVENT_PREFIX = "QS_EVENT "


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return {key: _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def emit_training_event(event_type: str, /, **payload: Any) -> None:
    """Emit one structured training event on stdout."""
    record = {"event": str(event_type), **{key: _json_safe(value) for key, value in payload.items()}}
    print(f"{EVENT_PREFIX}{json.dumps(record, sort_keys=True)}", flush=True)


def write_model_metadata(output_dir: str | Path, metadata: dict[str, Any]) -> Path:
    """Persist run metadata for app-side discovery and registration."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "model_metadata.json"
    path.write_text(json.dumps(_json_safe(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return path
