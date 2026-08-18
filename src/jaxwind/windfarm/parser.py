"""Low-level, label-oriented parser for OpenFAST text inputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shlex

from .errors import OpenFASTInputError


@dataclass(frozen=True, slots=True)
class _Line:
    number: int
    tokens: tuple[str, ...]


def _strip_comment(text: str) -> str:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(text):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote is not None:
            escaped = True
            continue
        if character in ("'", '"'):
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if quote is None and character in ("!", "#", "%"):
            return text[:index]
    return text


def _read_lines(path: Path) -> tuple[_Line, ...]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise OpenFASTInputError(
            f"cannot read OpenFAST file {path}: {error}"
        ) from error
    result: list[_Line] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        uncommented = _strip_comment(raw).strip()
        if not uncommented:
            continue
        try:
            tokens = tuple(shlex.split(uncommented, posix=True))
        except ValueError:
            # Output-channel descriptions in otherwise valid OpenFAST decks
            # occasionally contain unmatched prose apostrophes (for example
            # ``shaft's``). Required quoted path fields remain validated when
            # they are looked up; retaining these irrelevant prose lines as
            # whitespace tokens makes the label-oriented reader tolerant of
            # the upstream format.
            tokens = tuple(uncommented.split())
        if tokens:
            result.append(_Line(number, tokens))
    return tuple(result)


def _find(lines: tuple[_Line, ...], key: str, path: Path) -> tuple[int, int]:
    target = key.casefold().replace("_", "")
    for line_index, line in enumerate(lines):
        for token_index, token in enumerate(line.tokens):
            if token.casefold().replace("_", "") == target:
                return line_index, token_index
    raise OpenFASTInputError(f"{path}: required OpenFAST field {key!r} is missing")


def _value(
    lines: tuple[_Line, ...],
    key: str,
    path: Path,
) -> tuple[str, int]:
    line_index, token_index = _find(lines, key, path)
    tokens = lines[line_index].tokens
    value_index = token_index + 1 if token_index == 0 else token_index - 1
    if value_index >= len(tokens):
        raise OpenFASTInputError(
            f"{path}:{lines[line_index].number}: field {key!r} has no value"
        )
    return tokens[value_index], line_index


def _float_token(token: str, *, path: Path, context: str) -> float:
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise OpenFASTInputError(
            f"{path}: {context} must be numeric, got {token!r}"
        ) from error
    if not math.isfinite(value):
        raise OpenFASTInputError(f"{path}: {context} must be finite")
    return value


def _float_value(lines: tuple[_Line, ...], key: str, path: Path) -> float:
    token, _ = _value(lines, key, path)
    return _float_token(token, path=path, context=key)


def _integer_value(lines: tuple[_Line, ...], key: str, path: Path) -> int:
    value = _float_value(lines, key, path)
    result = int(value)
    if value != result:
        raise OpenFASTInputError(f"{path}: {key} must be an integer")
    return result


def _optional_integer_value(
    lines: tuple[_Line, ...],
    key: str,
    path: Path,
    *,
    default: int,
) -> int:
    try:
        _find(lines, key, path)
    except OpenFASTInputError:
        return default
    return _integer_value(lines, key, path)


def _boolean_token(token: str, *, path: Path, context: str) -> bool:
    normalized = token.casefold()
    if normalized in ("true", "t", "yes", "y", "1"):
        return True
    if normalized in ("false", "f", "no", "n", "0"):
        return False
    raise OpenFASTInputError(
        f"{path}: {context} must be a boolean, got {token!r}"
    )


def _boolean_value(lines: tuple[_Line, ...], key: str, path: Path) -> bool:
    token, _ = _value(lines, key, path)
    return _boolean_token(token, path=path, context=key)


def _optional_boolean_value(
    lines: tuple[_Line, ...],
    key: str,
    path: Path,
    *,
    default: bool,
) -> bool:
    try:
        _find(lines, key, path)
    except OpenFASTInputError:
        return default
    return _boolean_value(lines, key, path)


def _path_value(lines: tuple[_Line, ...], key: str, path: Path) -> Path:
    token, _ = _value(lines, key, path)
    if token.casefold() in ("unused", "none"):
        raise OpenFASTInputError(f"{path}: {key} may not be {token!r}")
    candidate = Path(token)
    return candidate if candidate.is_absolute() else (path.parent / candidate).resolve()


def _finite_number(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_number(value: float, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _numeric_row(
    line: _Line,
    *,
    columns: int,
) -> tuple[float, ...] | None:
    if len(line.tokens) < columns:
        return None
    try:
        values = tuple(
            float(token.replace("D", "E").replace("d", "e"))
            for token in line.tokens[:columns]
        )
    except ValueError:
        return None
    return values if all(math.isfinite(value) for value in values) else None


def _rows_after(
    lines: tuple[_Line, ...],
    line_index: int,
    *,
    count: int,
    columns: int,
    path: Path,
    context: str,
) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for line in lines[line_index + 1 :]:
        row = _numeric_row(line, columns=columns)
        if row is not None:
            rows.append(row)
            if len(rows) == count:
                return tuple(rows)
    raise OpenFASTInputError(
        f"{path}: expected {count} numeric {context} rows, found {len(rows)}"
    )
