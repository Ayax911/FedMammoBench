"""YAML <-> dataclass loader for fedmammo experiment configs.

The loader supports a ``defaults`` key for compositional configs: any path
listed there is loaded first, then the current file is merged on top with
deep-merge semantics.

Example::

    # configs/fedavg_synthetic.yaml
    defaults: base.yaml
    name: fedavg_synthetic
    federated:
      num_clients: 4
      strategy:
        name: fedavg
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any, TypeVar

import yaml

from fedmammo.configs.schema import ExperimentConfig

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a YAML config cannot be mapped onto the schema."""


def _is_dataclass_instance(obj: Any) -> bool:
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``.

    Dicts are merged key-wise; everything else is replaced wholesale by the
    override value. Returns a new dict; inputs are not mutated.
    """
    out: dict[str, Any] = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _from_dict(cls: type[T], data: dict[str, Any], _path: str = "") -> T:
    """Convert a plain dict into a (possibly nested) dataclass instance.

    Raises :class:`ConfigError` if ``data`` contains keys that are not fields
    on ``cls``. This is intentional — silent typos cost hours in research code.
    """
    if not dataclasses.is_dataclass(cls):
        raise ConfigError(f"{_path or '<root>'}: target type {cls!r} is not a dataclass")

    field_map = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data.keys()) - set(field_map.keys())
    if unknown:
        raise ConfigError(
            f"{_path or '<root>'}: unknown config keys {sorted(unknown)} "
            f"(allowed: {sorted(field_map.keys())})"
        )

    kwargs: dict[str, Any] = {}
    for fname, fdef in field_map.items():
        if fname not in data:
            continue
        value = data[fname]
        ftype = fdef.type

        # When ``from __future__ import annotations`` is in effect, ``f.type``
        # is a string. Resolve via the module's globals.
        if isinstance(ftype, str):
            import fedmammo.configs.schema as schema_mod

            try:
                ftype = eval(ftype, vars(schema_mod))  # noqa: S307
            except Exception:  # noqa: BLE001
                # Fall through and treat as opaque — useful for ``dict[str, Any]`` etc.
                ftype = None  # type: ignore[assignment]

        sub_path = f"{_path}.{fname}" if _path else fname

        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[fname] = _from_dict(ftype, value, _path=sub_path)
        else:
            kwargs[fname] = value
    return cls(**kwargs)  # type: ignore[arg-type]


def _to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass tree to plain dicts/lists for YAML dump."""
    if _is_dataclass_instance(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


def _resolve_defaults(
    raw: dict[str, Any], base_dir: Path, _seen: set[Path] | None = None
) -> dict[str, Any]:
    """Apply ``defaults: <path>`` references, depth-first."""
    if "defaults" not in raw:
        return raw

    defaults_value = raw.pop("defaults")
    if isinstance(defaults_value, str):
        default_paths = [defaults_value]
    elif isinstance(defaults_value, list):
        default_paths = list(defaults_value)
    else:
        raise ConfigError(
            f"`defaults` must be a string or a list of strings, got {type(defaults_value)!r}"
        )

    _seen = _seen or set()
    merged: dict[str, Any] = {}
    for rel in default_paths:
        ref = (base_dir / rel).resolve()
        if ref in _seen:
            raise ConfigError(f"Circular defaults reference involving {ref}")
        _seen = _seen | {ref}
        if not ref.is_file():
            raise ConfigError(f"`defaults` references missing file: {ref}")
        with ref.open("r", encoding="utf-8") as f:
            parent_raw = yaml.safe_load(f) or {}
        if not isinstance(parent_raw, dict):
            raise ConfigError(f"Top-level YAML in {ref} must be a mapping, got {type(parent_raw)!r}")
        parent_raw = _resolve_defaults(parent_raw, ref.parent, _seen=_seen)
        merged = _deep_merge(merged, parent_raw)
    return _deep_merge(merged, raw)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML file and return a fully populated :class:`ExperimentConfig`.

    Raises:
        ConfigError: if the YAML can't be mapped onto the schema (unknown
            keys, type mismatches, missing default-reference files, etc.).
        FileNotFoundError: if ``path`` does not exist.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level YAML in {p} must be a mapping, got {type(raw)!r}")
    resolved = _resolve_defaults(raw, p.parent)
    try:
        return _from_dict(ExperimentConfig, resolved)
    except TypeError as exc:  # dataclass __init__ rejected something
        raise ConfigError(f"Could not build ExperimentConfig from {p}: {exc}") from exc


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """Dump a fully resolved config to YAML (no ``defaults`` indirection).

    Useful for snapshotting the *effective* config alongside experiment logs.
    """
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_dict(cfg), f, sort_keys=False)


__all__ = ["ConfigError", "load_config", "save_config"]
