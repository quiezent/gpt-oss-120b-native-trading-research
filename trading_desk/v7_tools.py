"""Offline, read-only V7 host tools; receipts never authorize broker mutation.

The host owns the immutable input context. Persisted receipts are evidence only:
validation requires replay into a fresh host-issued receipt registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Iterator

from copy import deepcopy

from decimal import Decimal

import hashlib

import re

from typing import Any

from training import v6_contract as v6

_CREDENTIAL_KEYS = {"api_key", "apikey", "access_token", "refresh_token", "id_token", "password",
    "passwd", "secret", "client_secret", "private_key", "authorization", "credentials",
    "credential", "tinker_api_key", "openai_api_key", "ibkr_password", "session_cookie"}

_SECRET_TEXT = re.compile(r"(?:\bBearer\s+[A-Za-z0-9._~+/=-]{8,}|-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{16,}|\bAKIA[A-Z0-9]{16}\b)", re.I)

_CONTROL_TEXT = re.compile(r"<\|[^<>\r\n]*\|>")

def guard_content(value: Any) -> None:
    """Reject credentials/control text at every host and model trust boundary."""
    if type(value) not in {dict, list, str, int, float, bool, type(None)}:
        raise ValueError("only finite JSON-native values are permitted")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            normalized = key.lower().replace("-", "_")
            if normalized in _CREDENTIAL_KEYS or normalized.endswith(("_password", "_secret", "_api_key")):
                raise ValueError("credential material is forbidden")
            guard_content(key)
            guard_content(item)
    elif isinstance(value, list):
        for item in value:
            guard_content(item)
    elif isinstance(value, str):
        if _CONTROL_TEXT.search(value):
            raise ValueError("literal native control token is forbidden")
        if _SECRET_TEXT.search(value):
            raise ValueError("credential material is forbidden")
        if value.strip().lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
            raise ValueError("nonfinite numeric string is forbidden")
    elif type(value) is float and not v6._number(value).is_finite():
        raise ValueError("nonfinite number is forbidden")
    # Canonical JSON also rejects unsupported Python types, NaN and Infinity.
    try:
        v6.canonical(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("only finite JSON-native values are permitted") from exc
