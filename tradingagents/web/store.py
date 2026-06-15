from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SENSITIVE_KEY_PARTS = ("api_key", "apikey", "password", "secret", "token")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _sanitize(model_dump(mode="json"))
    if isinstance(value, dict):
        return {
            str(key): _sanitize(item)
            for key, item in value.items()
            if not any(part in str(key).lower() for part in _SENSITIVE_KEY_PARTS)
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


class RunStore:
    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        run = {
            "run_id": uuid.uuid4().hex,
            "status": "queued",
            "request": _sanitize(request),
            "progress": {
                "current_stage": "queued",
                "stages": {
                    "analysts": "pending",
                    "research": "pending",
                    "trader": "pending",
                    "risk": "pending",
                    "portfolio_manager": "pending",
                },
            },
            "reports": {},
            "result": {},
            "error": "",
            "created_at": now,
            "updated_at": now,
        }
        self._write(run)
        return run

    def get(self, run_id: str) -> dict[str, Any]:
        path = self._path(run_id)
        if not path.exists():
            raise KeyError(run_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for path in self.root.glob("*/run.json"):
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(run, dict):
                runs.append(run)
        runs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return runs[: max(1, min(int(limit), 200))]

    def update(self, run_id: str, **changes: Any) -> dict[str, Any]:
        run = self.get(run_id)
        for key, value in changes.items():
            run[key] = _sanitize(value)
        run["updated_at"] = utc_now()
        self._write(run)
        return run

    def _path(self, run_id: str) -> Path:
        if not run_id or not run_id.isalnum():
            raise KeyError(run_id)
        return self.root / run_id / "run.json"

    def _write(self, run: dict[str, Any]) -> None:
        path = self._path(str(run["run_id"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".run-", suffix=".json", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(_sanitize(run), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
