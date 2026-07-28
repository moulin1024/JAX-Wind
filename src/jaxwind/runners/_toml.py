"""Small deterministic TOML writer for resolved runner configurations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import re
from typing import Any


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(value: str) -> str:
    if _BARE_KEY.fullmatch(value):
        return value
    return json.dumps(value, ensure_ascii=False)


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("resolved TOML values must be finite")
        return repr(value)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return f"[{', '.join(_value(item) for item in value)}]"
    raise TypeError(f"unsupported resolved TOML value: {type(value).__name__}")


def dumps(document: Mapping[str, Any]) -> str:
    """Serialize nested scalar mappings as deterministic TOML.

    ``None`` entries are omitted because TOML intentionally has no null value.
    Resolved runner configurations contain only scalar values, scalar arrays,
    and nested tables, so supporting the broader TOML data model is unnecessary.
    """

    lines: list[str] = []

    def emit_table(table: Mapping[str, Any], path: tuple[str, ...]) -> None:
        scalar_items = [
            (key, value)
            for key, value in table.items()
            if value is not None and not isinstance(value, Mapping)
        ]
        child_items = [
            (key, value)
            for key, value in table.items()
            if isinstance(value, Mapping)
        ]

        if path:
            if lines:
                lines.append("")
            lines.append(f"[{'.'.join(_key(part) for part in path)}]")
        for key, value in scalar_items:
            lines.append(f"{_key(key)} = {_value(value)}")
        for key, value in child_items:
            emit_table(value, (*path, key))

    emit_table(document, ())
    return "\n".join(lines) + "\n"


__all__ = ["dumps"]
