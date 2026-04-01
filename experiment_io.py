from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def make_run_name(prefix: str, seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_seed{seed}"


def prepare_run_artifacts(
    *,
    output_dir: str,
    run_name: str,
    checkpoint_path: str | None,
) -> dict[str, str]:
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path else run_dir / "model.zip"
    resolved_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "checkpoint_path": str(resolved_checkpoint),
        "summary_path": str(run_dir / "summary.json"),
    }


def write_summary(path: str, payload: dict[str, Any]) -> None:
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
