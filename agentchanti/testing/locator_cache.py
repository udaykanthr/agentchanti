"""
LocatorCache — remembers which selector worked for each step, keyed by
(spec_name, step_id).

Purpose
-------
Replay determinism. A semantic label like "Place order button" may map to
multiple plausible selectors (``[data-testid=place-order]``,
``button[type=submit]``, ``text=Place order``) and the LLM could pick a
different one on each run — flaky CI, silent skew between runs, lost
trust. The cache pins the first-successful selector so subsequent replays
use the same one without calling the LLM.

Invalidation
------------
A cache entry is invalidated when:
  * the selector fails to match during replay — replayer deletes the
    entry before asking the LLM for a fresh one, so the next successful
    match is re-pinned;
  * the spec's step ``id`` changes — the key changes, so the old entry
    becomes dead and is garbage-collected on next save;
  * ``ttl_days`` has elapsed since ``last_verified_at`` — caller's choice.

On-disk format
--------------
``.agentchanti/testing/locator-cache.json`` in the project root::

    {
      "version": "1",
      "entries": {
        "checkout flow::step-1": {
          "selector": "button[data-testid=add-to-cart]",
          "last_verified_at": "2026-04-20T12:00:00Z"
        }
      }
    }

The file is rewritten atomically on each save — write to a ``.tmp``
sibling, then ``os.replace`` — so interrupted writes can't corrupt it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

CACHE_VERSION = "1"
DEFAULT_CACHE_PATH = Path(".agentchanti") / "testing" / "locator-cache.json"


@dataclass
class CacheEntry:
    selector: str
    last_verified_at: str  # ISO-8601 UTC

    def is_expired(self, ttl_days: int | None) -> bool:
        if ttl_days is None:
            return False
        try:
            ts = datetime.fromisoformat(self.last_verified_at.replace("Z", "+00:00"))
        except ValueError:
            # Corrupt timestamp → treat as expired so we re-verify.
            return True
        return ts < datetime.now(timezone.utc) - timedelta(days=ttl_days)


class LocatorCache:
    """Disk-backed (spec_name, step_id) → selector map."""

    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH):
        self.path = Path(path)
        self._entries: dict[str, CacheEntry] = {}
        self._load()

    # ---- Public API ------------------------------------------------------

    def get(
        self,
        spec_name: str,
        step_id: str,
        *,
        ttl_days: int | None = None,
    ) -> str | None:
        """Return the cached selector or ``None`` when absent/expired."""
        entry = self._entries.get(_key(spec_name, step_id))
        if entry is None:
            return None
        if entry.is_expired(ttl_days):
            return None
        return entry.selector

    def set(self, spec_name: str, step_id: str, selector: str) -> None:
        """Pin ``selector`` for this step and persist immediately."""
        self._entries[_key(spec_name, step_id)] = CacheEntry(
            selector=selector,
            last_verified_at=_now_iso(),
        )
        self._save()

    def invalidate(self, spec_name: str, step_id: str) -> None:
        """Drop the cached selector for this step (e.g. after a mismatch)."""
        if self._entries.pop(_key(spec_name, step_id), None) is not None:
            self._save()

    def clear(self) -> None:
        """Drop every cached entry — use when the spec schema version bumps."""
        self._entries.clear()
        self._save()

    def __len__(self) -> int:
        return len(self._entries)

    # ---- I/O -------------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Corrupt cache → start fresh. Never let a bad file block replay.
            return
        if raw.get("version") != CACHE_VERSION:
            return
        self._entries = {
            k: CacheEntry(**v) for k, v in raw.get("entries", {}).items()
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "entries": {k: asdict(v) for k, v in self._entries.items()},
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


def _key(spec_name: str, step_id: str) -> str:
    return f"{spec_name}::{step_id}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
