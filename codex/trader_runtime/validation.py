"""Deterministic validation for the replay desk; never grants broker authority."""

from __future__ import annotations

from datetime import datetime

from decimal import Decimal, InvalidOperation

from typing import Any

PAPER_BOUNDARY = {"host": "127.0.0.1", "port": 4002, "account": "PAPER_ACCOUNT", "client_id": 9901}

def money(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("boolean is not a monetary amount")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("invalid monetary amount") from exc
    if not result.is_finite():
        raise ValueError("non-finite monetary amount")
    return result

def positive_int(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("positive whole integer required")
    return value

def timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid ISO-8601 timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return result

def strict_fields(value: Any, required: set[str], optional: set[str] | None = None) -> None:
    if not isinstance(value, dict) or not required <= value.keys():
        raise ValueError("object is missing required fields")
    if value.keys() - required - (optional or set()):
        raise ValueError("object has unsupported fields")

def contract_key(value: dict[str, Any]) -> tuple[int, str, str]:
    conid = positive_int(value["conid"])
    symbol = value["symbol"]
    exchange = value["primary_exchange"]
    if not isinstance(symbol, str) or not symbol or not isinstance(exchange, str) or not exchange:
        raise ValueError("contract identity is incomplete")
    return conid, symbol, exchange
