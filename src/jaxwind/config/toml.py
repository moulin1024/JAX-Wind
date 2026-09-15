"""Small TOML writer for resolved configuration data (no arbitrary objects)."""
import json
import math


def _value(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return "{ " + ", ".join(json.dumps(key) + " = " + _value(item) for key, item in value.items()) + " }"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    raise ValueError(f"unsupported TOML value: {type(value).__name__}")


def dumps(document):
    lines = []
    def table(data, path):
        if path:
            lines.append("[" + ".".join(json.dumps(key) for key in path) + "]")
        for key, value in data.items():
            if not isinstance(value, dict):
                lines.append(json.dumps(key) + " = " + _value(value))
        lines.append("")
        for key, value in data.items():
            if isinstance(value, dict):
                table(value, (*path, key))
    table(document, ())
    return "\n".join(lines)
