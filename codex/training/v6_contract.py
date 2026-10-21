"""Separately bound V6 decisions: typed evidence, hard constraints, discretion.

No provider/broker is instantiated here. Validation proves internal consistency
of supplied evidence and worksheets, never economic skill or execution authority.
The V5 profile and its exact-policy teacher remain untouched.
"""

from __future__ import annotations

import hashlib

import json

from copy import deepcopy

from datetime import datetime, timezone

from decimal import Decimal, InvalidOperation

from typing import Any

def canonical(value: Any) -> str:
    def keys(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for nested in item.values():
                keys(nested)
        elif isinstance(item, list):
            for nested in item:
                keys(nested)

    keys(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

def _number(value: Any) -> Decimal:
    if type(value) not in {int, float, str, Decimal}:
        raise ValueError("number must be finite and non-boolean")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("number must be finite") from exc
    if not number.is_finite():
        raise ValueError("number must be finite")
    return number

def parse_response(text: str) -> dict:
    """Strict native final JSON: no repair, markdown, duplicate keys or NaN."""
    def pairs(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def constant(_: str):
        raise ValueError("nonfinite JSON constant")

    try:
        result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("final output is not strict JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("final response must be one JSON object")
    canonical(result)
    return result
