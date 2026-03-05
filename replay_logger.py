"""Replay logging helpers for blackjack episodes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def save_replay(path: str | Path, replay: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(replay, f, indent=2)


def save_replay_bundle(path: str | Path, replays: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"episodes": list(replays)}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_replay(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "episodes" in data:
        return data["episodes"]
    return [data]
