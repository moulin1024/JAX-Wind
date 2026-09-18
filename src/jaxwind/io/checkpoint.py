"""Atomic, versioned full-state checkpoints; no pickle or executable metadata."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np

SCHEMA = "jaxwind.checkpoint.v1"


def _flatten(value, path, arrays):
    if hasattr(value, "_fields"):
        return {"record": type(value).__name__, "fields": {key: _flatten(getattr(value, key), path + "/" + key, arrays) for key in value._fields}}
    if isinstance(value, dict):
        return {"dict": {key: _flatten(item, path + "/" + key, arrays) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        return {"sequence": [_flatten(item, path + f"/{i}", arrays) for i, item in enumerate(value)], "tuple": isinstance(value, tuple)}
    if value is None or isinstance(value, (str, bool, int)) or (isinstance(value, float) and np.isfinite(value)):
        return {"literal": value}
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"object arrays cannot be checkpointed: {path}")
    arrays[path] = array
    return {"array": path}


def _restore(node, arrays, template=None):
    if "array" in node:
        array = np.array(arrays[node["array"]], copy=True)
        if template is not None and (array.shape != template.shape or array.dtype != template.dtype):
            raise ValueError(f"checkpoint shape/precision mismatch: {node['array']}")
        if template is not None:
            import jax.numpy as jnp
            return jnp.asarray(array)
        return array
    if "record" in node:
        # Historical DPM checkpoints predate separators: collection is zero.
        if (node["record"] == "DPMLedger" and template is not None
            and set(template._fields) - node["fields"].keys() == {"collected"}
            and not node["fields"].keys() - set(template._fields)):
            return type(template)(*(
                getattr(template, key) * 0 if key == "collected"
                else _restore(node["fields"][key], arrays, getattr(template, key))
                for key in template._fields))
        if template is None or type(template).__name__ != node["record"] or set(template._fields) != node["fields"].keys():
            raise ValueError("checkpoint state type does not match formulation")
        return type(template)(*(_restore(node["fields"][key], arrays, getattr(template, key)) for key in template._fields))
    if "dict" in node:
        return {key: _restore(item, arrays) for key, item in node["dict"].items()}
    if "sequence" in node:
        result = [_restore(item, arrays) for item in node["sequence"]]
        return tuple(result) if node["tuple"] else result
    return node["literal"]


def save_checkpoint(path, state, *, metadata, observer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    tree = _flatten(state, "state", arrays)
    observer_tree = _flatten(observer or {}, "observer", arrays)
    header = {**metadata, "schema": SCHEMA, "state": tree, "observer": observer_tree}
    arrays["metadata"] = np.asarray(json.dumps(header, sort_keys=True, allow_nan=False))
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_metadata(path):
    with np.load(path, allow_pickle=False) as arrays:
        if "metadata" not in arrays:
            raise ValueError("historical checkpoint format is unsupported; start a new run")
        header = json.loads(str(arrays["metadata"]))
    if header.get("schema") != SCHEMA:
        raise ValueError("unsupported checkpoint schema")
    return header


def load_checkpoint(path, template, *, fingerprint):
    header = checkpoint_metadata(path)
    if header.get("fingerprint") != fingerprint:
        raise ValueError("checkpoint configuration differs; exact resume requires the original case")
    if header.get("initialization_only"):
        raise ValueError("transformed checkpoint is initialization input, not an exact-resume run")
    with np.load(path, allow_pickle=False) as arrays:
        state = _restore(header["state"], arrays, template)
        observer = _restore(header["observer"], arrays)
    return state, observer, header
