from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("Top-level configuration must be a mapping")
    return value


def _parse_scalar(value: str) -> Any:
    parsed = yaml.safe_load(value)
    return parsed


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError("Override must have key=value form: %s" % override)
        dotted_key, raw_value = override.split("=", 1)
        keys = [item for item in dotted_key.split(".") if item]
        if not keys:
            raise ValueError("Empty override key")
        cursor = config
        for key in keys[:-1]:
            if key not in cursor:
                cursor[key] = {}
            if not isinstance(cursor[key], dict):
                raise ValueError("Cannot descend into non-mapping key: %s" % key)
            cursor = cursor[key]
        cursor[keys[-1]] = _parse_scalar(raw_value)
    return config
