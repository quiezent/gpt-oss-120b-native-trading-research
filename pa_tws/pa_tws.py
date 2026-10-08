"""Small, fail-closed TWS API utility for the principal's Personal Assistant.

The module is deliberately independent of the legacy portfolio-manager code in
this repository. Read operations are the default. Mutations are limited to one
explicitly described stock order submission or cancellation of an exact
PA-owned order; both require explicit command-line and environment gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import sys
import threading
import time
from contextlib import contextmanager
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import TracebackType
from typing import Any, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.execution import ExecutionFilter
from ibapi.order import UNSET_DOUBLE, UNSET_INTEGER, Order
from ibapi.wrapper import EWrapper

if __package__:
    from . import paper_trial as _paper_trial
    from . import sdk_compat as _sdk_compat
    from . import fee_capture as _fee_capture
else:
    import paper_trial as _paper_trial
    import sdk_compat as _sdk_compat
    import fee_capture as _fee_capture

NativePaperDailyLossObservation = _paper_trial.NativePaperDailyLossObservation
PAPER_DELAYED_TYPE3 = _paper_trial.PROFILE

HOST = "127.0.0.1"
PORT = 4002
ACCOUNT = "PAPER_ACCOUNT"
DEFAULT_CLIENT_ID = 9901
PA_TWS_DIR = Path(__file__).resolve().parent
CANONICAL_AUDIT_PATH = PA_TWS_DIR / "audit" / "pa_tws_audit.jsonl"
CANONICAL_MUTATION_LOCK_PATH = PA_TWS_DIR / "audit" / "pa_tws_mutation.lock"

# Commissioned manager/order and research clients are not available to the PA.
PROTECTED_CLIENT_IDS = frozenset({0, 1101, 1201, 1301, 1401, 2101, 2201, 2301, 2401})
PA_REF_RE = re.compile(r"^pa:[A-Za-z0-9][A-Za-z0-9_.:-]{4,60}$")
HARD_MAX_SHARES = 10_000
HARD_MAX_NOTIONAL_USD = Decimal("1000000.00")
MONEY_TICK = Decimal("0.01")
US_PRIMARY_EXCHANGES = frozenset({"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "IEX"})
SUPPORTED_STOCK_ORDER_TYPES = frozenset(
    {"LMT", "MKT", "STP", "STP LMT", "MIT", "LIT", "TRAIL", "TRAIL LIMIT"}
)
SUPPORTED_TIFS = frozenset({"DAY", "GTC", "GTD", "IOC", "FOK", "OPG"})
SUPPORTED_TRIGGER_METHODS = frozenset({0, 1, 2, 3, 4, 7, 8})
INFORMATIONAL_ERROR_CODES = frozenset(
    {
        2104,  # market data farm connected
        2106,  # historical data farm connected
        2107,  # historical data farm inactive until requested
        2108,  # market data farm inactive until requested
        2158,  # sec-def farm connected
    }
)
FATAL_ORDER_REJECTION_ERROR_CODES = frozenset({201})
LOCKED_RESEARCH_COMMANDS = frozenset({
    "historical-daily", "news-providers", "historical-news", "news-article",
    "live-news", "broadtape-news",
})
NEWS_PROVIDER_RE = re.compile(r"^[A-Za-z0-9_]{1,24}$")
NEWS_ARTICLE_RE = re.compile(r"^[A-Za-z0-9$_.:-]{1,256}$")


class PATWSError(RuntimeError):
    """Base exception for local validation and TWS failures."""


class BoundaryError(PATWSError):
    """The requested connection or account falls outside the PA boundary."""


class GateError(PATWSError):
    """An explicit mutation gate was absent or incorrect."""


class PolicyError(PATWSError):
    """An order does not satisfy the local deterministic policy."""


class BrokerTimeout(PATWSError):
    """A broker callback did not arrive before the bounded timeout."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc_timestamp(value: str, label: str) -> datetime:
    """Require an explicit zero-offset timestamp; never apply a local timezone."""
    try:
        if not isinstance(value, str):
            raise TypeError("timestamp must be text")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ValueError("timestamp must have a UTC offset")
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        raise PolicyError(f"{label} must be an explicit UTC ISO timestamp") from None


def news_publication_time(value: Any) -> dict[str, Any]:
    """Parse proven UTC formats, retaining ambiguous legacy callback timestamps.

    Current IBKR docs describe epoch seconds, while the installed legacy decoder
    delivers text. Naive calendar text is retained without inventing a timezone.
    Neither an article's original version nor its arrival at a past decision is
    proven merely by today's historical retrieval.
    """
    raw = str(value)
    if type(value) is int or (isinstance(value, str) and value.isdigit()):
        epoch = int(value)
        if epoch <= 0:
            raise ValueError("news epoch must be positive")
        parsed = datetime.fromtimestamp(epoch, timezone.utc)
        status = "epoch_seconds_utc"
    elif isinstance(value, str):
        if re.fullmatch(r"\d{8}-\d{2}:\d{2}:\d{2}", raw):
            parsed = datetime.strptime(raw, "%Y%m%d-%H:%M:%S").replace(tzinfo=timezone.utc)
            status = "explicit_native_utc"
        elif raw.endswith(" UTC"):
            text = raw[:-4]
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is not None and parsed.utcoffset() != timedelta(0):
                raise ValueError("conflicting UTC suffix and offset")
            parsed = parsed.replace(tzinfo=timezone.utc)
            status = "explicit_utc_suffix"
        else:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                return {"native_time": raw, "published_at_utc": None,
                        "timestamp_status": "timezone_unknown", "decision_time_eligible": False}
            parsed = parsed.astimezone(timezone.utc)
            status = "explicit_offset_utc"
    else:
        raise ValueError("unsupported news callback timestamp")
    return {"native_time": raw,
            "published_at_utc": parsed.isoformat().replace("+00:00", "Z"),
            "timestamp_status": status, "decision_time_eligible": False}


def validate_news_providers(value: str) -> tuple[str, ...]:
    codes = value.split("+") if isinstance(value, str) else []
    if (not codes or len(codes) > 20 or len(set(codes)) != len(codes)
            or any(not NEWS_PROVIDER_RE.fullmatch(code) for code in codes)):
        raise PolicyError("providers must contain distinct valid codes separated by '+'")
    return tuple(codes)


def validate_news_capture(seconds: float, max_headlines: int) -> None:
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or not 0 < seconds <= 60
            or type(max_headlines) is not int or not 1 <= max_headlines <= 1000):
        raise PolicyError("news capture requires seconds in (0, 60] and max-headlines in 1..1000")


def make_news_contract(provider: str, symbol: str) -> Contract:
    codes = validate_news_providers(provider)
    if (len(codes) != 1 or not isinstance(symbol, str)
            or not re.fullmatch(re.escape(provider) + r":[A-Za-z0-9_]{1,64}", symbol)):
        raise PolicyError("BroadTape requires one provider and its exact PROVIDER:NEWS_SYMBOL")
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "NEWS"
    contract.exchange = provider
    return contract


def live_news_publication_time(value: Any) -> dict[str, Any]:
    """Retain the integer epoch and explicitly label inferred seconds/millis.

    IBKR's tickNews docs say epoch time without specifying the unit. Never feed
    a millisecond epoch into the historical-news seconds parser or silently
    claim this normalization proves point-in-time publication provenance.
    """
    if type(value) is not int or value <= 0:
        raise ValueError("live news requires a positive integer epoch")
    milliseconds = value >= 100_000_000_000
    seconds, remainder = divmod(value, 1000) if milliseconds else (value, 0)
    parsed = datetime.fromtimestamp(seconds, timezone.utc) + timedelta(milliseconds=remainder)
    if not 2000 <= parsed.year <= 2100:
        raise ValueError("live news epoch is outside the supported calendar range")
    return {"native_time": str(value), "native_timestamp": value,
            "published_at_utc": parsed.isoformat().replace("+00:00", "Z"),
            "timestamp_status": "epoch_milliseconds_inferred" if milliseconds else "epoch_seconds_inferred",
            "timestamp_unit_inferred": True, "decision_time_eligible": False}


def _number(value: Any) -> int | float | str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return int(number) if number.is_integer() else number


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _decimal_value(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise PolicyError(f"invalid numeric broker value: {value!r}") from exc
    if not result.is_finite():
        raise PolicyError(f"non-finite broker value: {value!r}")
    return result


def _remaining_quantity(order: Order) -> Decimal:
    total = _decimal_value(order.totalQuantity)
    try:
        filled_number = float(order.filledQuantity)
    except (TypeError, ValueError):
        return max(total, Decimal(0))
    if not math.isfinite(filled_number):
        return max(total, Decimal(0))
    filled = Decimal(str(filled_number))
    if filled < 0 or filled > total:
        return max(total, Decimal(0))
    return max(total - filled, Decimal(0))


def validate_boundary(
    host: str = HOST,
    port: int = PORT,
    account: str = ACCOUNT,
    client_id: int = DEFAULT_CLIENT_ID,
) -> None:
    if host != HOST:
        raise BoundaryError(f"host must be exactly {HOST}")
    if port != PORT:
        raise BoundaryError(f"port must be exactly {PORT}")
    if account != ACCOUNT:
        raise BoundaryError(f"account must be exactly {ACCOUNT}")
    if client_id != DEFAULT_CLIENT_ID or isinstance(client_id, bool):
        raise BoundaryError(f"client ID must be exactly the PA client {DEFAULT_CLIENT_ID}")


def assert_account_access(accounts: Iterable[str]) -> None:
    observed = {item.strip() for item in accounts if item and item.strip()}
    if ACCOUNT not in observed:
        raise BoundaryError(
            f"TWS did not positively expose paper account {ACCOUNT}; observed={sorted(observed)}"
        )


def require_mutation_gate(
    action: str,
    confirmation: str | None,
    environ: Mapping[str, str] | None = None,
) -> None:
    expected = {
        "submit": "SUBMIT_PAPER_ORDER",
        "cancel": "CANCEL_PAPER_ORDER",
    }.get(action)
    if expected is None:
        raise GateError(f"unsupported mutation action: {action}")
    env = os.environ if environ is None else environ
    if env.get("IBKR_PA_ALLOW_PAPER_ORDER") != "YES":
        raise GateError("set IBKR_PA_ALLOW_PAPER_ORDER=YES for this one invocation")
    if confirmation != expected:
        raise GateError(f"confirmation must be exactly {expected}")


def validate_order_ref(order_ref: str) -> str:
    if not PA_REF_RE.fullmatch(order_ref):
        raise PolicyError(
            "orderRef must be a unique PA reference beginning 'pa:' (5-64 safe characters)"
        )
    return order_ref


def parse_money(value: str | float | Decimal) -> Decimal:
    try:
        price = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PolicyError("price must be a finite decimal") from exc
    if not price.is_finite() or price <= 0:
        raise PolicyError("limit price must be positive and finite")
    if price != price.quantize(MONEY_TICK):
        raise PolicyError("limit price must use a $0.01 tick")
    return price


def parse_percent(value: str | float | Decimal) -> Decimal:
    try:
        percent = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PolicyError("percentage must be a finite decimal") from exc
    if not percent.is_finite() or percent <= 0 or percent > Decimal(100):
        raise PolicyError("percentage must be greater than 0 and no more than 100")
    return percent


def _optional_order_number(value: Any) -> int | float | str | None:
    """Return None for ibapi's huge UNSET_DOUBLE sentinel values."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or abs(number) >= 1e307:
        return None
    return _number(number)


@dataclass(frozen=True)
class RiskPolicy:
    max_quantity: int
    max_notional_usd: Decimal
    allowed_sides: frozenset[str]
    allowed_tif: frozenset[str]
    allowed_order_types: frozenset[str]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RiskPolicy:
        if raw.get("host") != HOST or raw.get("port") != PORT or raw.get("account") != ACCOUNT:
            raise PolicyError("risk policy must bind exactly to the PA paper boundary")
        if raw.get("security_type") != "STK":
            raise PolicyError("risk policy must permit only STK orders")
        configured_types = raw.get("order_types")
        if configured_types is None:
            legacy_type = raw.get("order_type")
            configured_types = [legacy_type] if legacy_type else []
        if not isinstance(configured_types, list):
            raise PolicyError("order_types must be a non-empty list")
        order_types = frozenset(str(item).upper() for item in configured_types)
        if not order_types or not order_types <= SUPPORTED_STOCK_ORDER_TYPES:
            raise PolicyError(
                "order_types must be a non-empty subset of supported stock order types"
            )
        quantity = raw.get("max_quantity")
        if not isinstance(quantity, int) or isinstance(quantity, bool) or not 1 <= quantity <= HARD_MAX_SHARES:
            raise PolicyError(f"max_quantity must be in 1..{HARD_MAX_SHARES}")
        notional = parse_money(raw.get("max_notional_usd", "0"))
        if notional > HARD_MAX_NOTIONAL_USD:
            raise PolicyError(f"max_notional_usd exceeds hard ceiling {HARD_MAX_NOTIONAL_USD}")
        sides = frozenset(str(item).upper() for item in raw.get("allowed_sides", []))
        tif = frozenset(str(item).upper() for item in raw.get("allowed_tif", []))
        if not sides or not sides <= {"BUY", "SELL"}:
            raise PolicyError("allowed_sides must be a non-empty subset of BUY/SELL")
        if not tif or not tif <= SUPPORTED_TIFS:
            raise PolicyError("allowed_tif contains an unsupported time-in-force")
        return cls(quantity, notional, sides, tif, order_types)

    @classmethod
    def load(cls, path: Path) -> RiskPolicy:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PolicyError(f"cannot read valid risk policy {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise PolicyError("risk policy root must be an object")
        return cls.from_mapping(raw)

    def validate_order(
        self,
        *,
        side: str,
        quantity: int,
        order_type: str = "LMT",
        reference_price: Decimal | None = None,
        tif: str,
        current_position: Decimal | None = None,
    ) -> None:
        side = side.upper()
        order_type = order_type.upper()
        tif = tif.upper()
        if order_type not in self.allowed_order_types:
            raise PolicyError(f"order type {order_type} is not permitted by policy")
        if side not in self.allowed_sides:
            raise PolicyError(f"side {side} is not permitted by policy")
        if tif not in self.allowed_tif:
            raise PolicyError(f"TIF {tif} is not permitted by policy")
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            raise PolicyError("quantity must be a positive whole-share integer")
        if quantity > self.max_quantity:
            raise PolicyError("quantity exceeds policy maximum")
        if reference_price is None or reference_price <= 0:
            raise PolicyError("order requires a positive price reference for the notional check")
        if Decimal(quantity) * reference_price > self.max_notional_usd:
            raise PolicyError("order notional exceeds policy maximum")
        if side == "SELL":
            if current_position is None:
                raise PolicyError("SELL requires a positively reconciled current position")
            if current_position < Decimal(quantity):
                raise PolicyError("short sales and sales above the current long position are prohibited")


class AuditLog:
    """Append-only JSONL audit with a simple SHA-256 hash chain."""

    def __init__(self, path: Path):
        self.path = path

    def verify(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        expected_previous = "0" * 64
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PATWSError(
                        f"audit line {line_number} is invalid JSON: {self.path}"
                    ) from exc
                if not isinstance(value, dict):
                    raise PATWSError(f"audit line {line_number} is not an object")
                observed_hash = value.pop("hash", None)
                if value.get("prev_hash") != expected_previous:
                    raise PATWSError(f"audit chain break at line {line_number}: {self.path}")
                canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
                computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                if observed_hash != computed:
                    raise PATWSError(f"audit hash mismatch at line {line_number}: {self.path}")
                value["hash"] = observed_hash
                rows.append(value)
                expected_previous = computed
        return rows

    def _previous_hash(self) -> str:
        rows = self.verify()
        return rows[-1]["hash"] if rows else "0" * 64

    def append(self, event: str, details: Mapping[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "timestamp_utc": utc_now(),
            "event": event,
            "details": dict(details),
            "prev_hash": self._previous_hash(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return payload

    def has_attempt(self, event: str, order_ref: str, order_id: int | None = None) -> bool:
        validate_order_ref(order_ref)
        for item in self.verify():
            if item.get("event") != event:
                continue
            details = item.get("details", {})
            record = details.get("terms") or details.get("order") or {}
            seen_ref = str(record.get("order_ref") or "")
            refs_collide = seen_ref.startswith("pa:") and (
                seen_ref == order_ref
                or seen_ref.startswith(order_ref)
                or order_ref.startswith(seen_ref)
            )
            if refs_collide and (order_id is None or record.get("order_id") == order_id):
                return True
        return False


class MutationLock:
    """Atomic single-run lock; stale locks are intentionally never stolen."""

    def __init__(self, path: Path | None = None):
        self.path = CANONICAL_MUTATION_LOCK_PATH if path is None else path
        self._fd: int | None = None
        self._token: str | None = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise GateError(
                f"another mutation run may be active; inspect but do not steal {self.path}"
            ) from exc
        self._token = secrets.token_hex(16)
        payload = json.dumps(
            {"pid": os.getpid(), "created_at_utc": utc_now(), "token": self._token}
        ).encode()
        try:
            os.write(self._fd, payload)
            os.fsync(self._fd)
        except Exception:
            os.close(self._fd)
            self._fd = None
            self.path.unlink(missing_ok=True)
            self._token = None
            raise
        return self

    def owns(self, expected_path: Path) -> bool:
        if (
            self._fd is None
            or self._token is None
            or self.path.resolve() != expected_path.resolve()
        ):
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("token") == self._token

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        owned = self.owns(self.path)
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if owned:
            self.path.unlink(missing_ok=True)
        self._token = None


def require_mutation_context(audit: AuditLog, lock: MutationLock | None) -> None:
    if audit.path.resolve() != CANONICAL_AUDIT_PATH.resolve():
        raise GateError("mutation requires the single canonical PA audit path")
    if lock is None or not lock.owns(CANONICAL_MUTATION_LOCK_PATH):
        raise GateError("mutation requires the dispatcher-owned canonical single-run lock")


def make_stock_contract(
    *, conid: int = 0, symbol: str = "", primary_exchange: str = ""
) -> Contract:
    if conid < 0:
        raise PolicyError("conId cannot be negative")
    if not conid and not symbol.strip():
        raise PolicyError("provide conId or symbol")
    contract = Contract()
    contract.conId = int(conid)
    contract.symbol = symbol.strip().upper()
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    contract.primaryExchange = primary_exchange.strip().upper()
    return contract


def make_stock_order(
    *,
    order_type: str,
    side: str,
    quantity: int,
    tif: str,
    order_ref: str,
    limit_price: Decimal | None = None,
    stop_price: Decimal | None = None,
    trail_amount: Decimal | None = None,
    trail_percent: Decimal | None = None,
    trail_stop_price: Decimal | None = None,
    limit_offset: Decimal | None = None,
    trigger_method: int = 0,
    good_till_date: str | None = None,
    client_id: int = DEFAULT_CLIENT_ID,
) -> Order:
    order_type = order_type.upper()
    side = side.upper()
    tif = tif.upper()
    validate_order_ref(order_ref)
    if order_type not in SUPPORTED_STOCK_ORDER_TYPES:
        raise PolicyError(f"unsupported stock order type {order_type}")
    if side not in {"BUY", "SELL"}:
        raise PolicyError("side must be BUY or SELL")
    if tif not in SUPPORTED_TIFS:
        raise PolicyError("unsupported TIF")
    if tif == "GTD" and not good_till_date:
        raise PolicyError("GTD requires a good-till-date")
    if tif != "GTD" and good_till_date:
        raise PolicyError("good-till-date is only valid with GTD")
    if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
        raise PolicyError("quantity must be a positive whole-share integer")
    if trigger_method not in SUPPORTED_TRIGGER_METHODS:
        raise PolicyError("unsupported trigger method")
    validate_boundary(client_id=client_id)

    limit_types = {"LMT", "STP LMT", "LIT"}
    stop_types = {"STP", "STP LMT", "MIT", "LIT"}
    trailing_types = {"TRAIL", "TRAIL LIMIT"}
    if order_type in limit_types and limit_price is None:
        raise PolicyError(f"{order_type} requires a limit price")
    if order_type not in limit_types and order_type != "TRAIL LIMIT" and limit_price is not None:
        raise PolicyError(f"{order_type} does not accept a limit price")
    if order_type in stop_types and stop_price is None:
        raise PolicyError(f"{order_type} requires a stop price")
    if order_type not in stop_types and stop_price is not None:
        raise PolicyError(f"{order_type} does not accept a stop price")
    if order_type in trailing_types:
        if (trail_amount is None) == (trail_percent is None):
            raise PolicyError(
                f"{order_type} requires exactly one of trail amount or trail percent"
            )
    elif any(
        item is not None for item in (trail_amount, trail_percent, trail_stop_price, limit_offset)
    ):
        raise PolicyError(f"{order_type} does not accept trailing-order fields")
    if order_type == "TRAIL LIMIT":
        if limit_price is not None and limit_offset is not None:
            raise PolicyError("TRAIL LIMIT accepts either a limit price or limit offset, not both")
        if limit_price is None and limit_offset is None:
            raise PolicyError("TRAIL LIMIT requires a limit price or limit offset")
    elif limit_offset is not None:
        raise PolicyError(f"{order_type} does not accept a limit offset")

    order = Order()
    # The bundled ibapi version has legacy SmartRouting defaults that current
    # TWS rejects (errors 10268/10269). Keep deprecated fields explicitly
    # disabled/unset for every order created by this PA.
    order.eTradeOnly = False
    order.firmQuoteOnly = False
    order.nbboPriceCap = UNSET_DOUBLE
    order.auxPrice = UNSET_DOUBLE
    order.lmtPrice = UNSET_DOUBLE
    order.lmtPriceOffset = UNSET_DOUBLE
    order.trailStopPrice = UNSET_DOUBLE
    order.trailingPercent = UNSET_DOUBLE
    order.minQty = UNSET_INTEGER
    order.action = side
    order.totalQuantity = quantity
    order.orderType = order_type
    if limit_price is not None:
        order.lmtPrice = float(limit_price)
    if stop_price is not None:
        order.auxPrice = float(stop_price)
    if trail_amount is not None:
        order.auxPrice = float(trail_amount)
    if trail_percent is not None:
        order.trailingPercent = float(trail_percent)
    if trail_stop_price is not None:
        order.trailStopPrice = float(trail_stop_price)
    if limit_offset is not None:
        order.lmtPriceOffset = float(limit_offset)
    order.triggerMethod = trigger_method
    order.tif = tif
    order.goodTillDate = good_till_date or ""
    order.account = ACCOUNT
    order.clientId = client_id
    order.orderRef = order_ref
    order.outsideRth = False
    order.transmit = True
    return order


def make_limit_order(
    *,
    side: str,
    quantity: int,
    limit_price: Decimal,
    tif: str,
    order_ref: str,
    client_id: int = DEFAULT_CLIENT_ID,
) -> Order:
    """Backward-compatible LMT builder for callers of the original utility."""

    return make_stock_order(
        order_type="LMT",
        side=side,
        quantity=quantity,
        limit_price=limit_price,
        tif=tif,
        order_ref=order_ref,
        client_id=client_id,
    )


def duplicate_order_ref(order_ref: str, records: Iterable[Mapping[str, Any]]) -> bool:
    validate_order_ref(order_ref)
    for record in records:
        seen = str(record.get("order_ref") or "")
        if seen.startswith("pa:") and (seen == order_ref or seen.startswith(order_ref) or order_ref.startswith(seen)):
            return True
    return False


def eligible_cancel_record(
    record: Mapping[str, Any], *, order_id: int, order_ref: str, conid: int, client_id: int
) -> bool:
    validate_order_ref(order_ref)
    return (
        record.get("order_id") == order_id
        and record.get("order_ref") == order_ref
        and record.get("account") == ACCOUNT
        and record.get("conid") == conid
        and record.get("client_id") == client_id
        and record.get("status") not in {"Cancelled", "ApiCancelled", "Filled", "Inactive"}
    )


def contract_record(contract: Contract) -> dict[str, Any]:
    return {
        "conid": int(contract.conId or 0),
        "symbol": contract.symbol,
        "local_symbol": contract.localSymbol,
        "sec_type": contract.secType,
        "exchange": contract.exchange,
        "primary_exchange": contract.primaryExchange,
        "currency": contract.currency,
    }


def order_record(order_id: int, contract: Contract, order: Order, status: str = "") -> dict[str, Any]:
    order_type = str(order.orderType or "")
    aux_price = _optional_order_number(order.auxPrice)
    return {
        "order_id": int(order_id),
        "perm_id": _integer(order.permId),
        "client_id": _integer(order.clientId),
        "account": order.account,
        "order_ref": order.orderRef,
        "conid": int(contract.conId or 0),
        "symbol": contract.symbol,
        "sec_type": contract.secType,
        "currency": contract.currency,
        "primary_exchange": contract.primaryExchange,
        "side": order.action,
        "quantity": _number(order.totalQuantity),
        "remaining_quantity": _number(_remaining_quantity(order)),
        "order_type": order_type,
        "limit_price": _optional_order_number(order.lmtPrice),
        "aux_price": aux_price,
        "stop_price": aux_price if order_type in {"STP", "STP LMT", "MIT", "LIT"} else None,
        "trailing_amount": (
            aux_price
            if order_type in {"TRAIL", "TRAIL LIMIT"}
            and _optional_order_number(order.trailingPercent) is None
            else None
        ),
        "trailing_percent": _optional_order_number(order.trailingPercent),
        "trail_stop_price": _optional_order_number(order.trailStopPrice),
        "limit_offset": _optional_order_number(order.lmtPriceOffset),
        "trigger_method": _integer(order.triggerMethod),
        "tif": order.tif,
        "good_till_date": order.goodTillDate or None,
        "outside_rth": bool(order.outsideRth),
        "transmit": bool(order.transmit),
        "status": status,
    }


def _known_held_order_notice(item: Mapping[str, Any], order_id: int) -> bool:
    """A scoped session-hold notice, never a general code399 exemption."""
    message = item.get("message")
    return (item.get("req_id") == order_id and item.get("code") == 399
            and not item.get("advanced_reject") and isinstance(message, str)
            and re.fullmatch(r"Order Message:\r?\n(?:BUY|SELL) [1-9]\d*(?:\.\d+)? "
                          r"[A-Z0-9][A-Z0-9._/-]*(?: [A-Z])? [A-Z0-9][A-Z0-9._/-]*\r?\n"
                          r"Warning: Your order will not be placed at the exchange until "
                          r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [A-Za-z0-9_+./-]+\.$",
                          message.strip()) is not None)


@contextmanager
def _submission_notice_scope(app: "TWSApp", order_id: int):
    """Allow collection of this submission's evidence before classifying notice."""
    if getattr(app, "_reconciling_submission_notice_order_id", None) is not None:
        raise PATWSError("nested submission notice reconciliation refused")
    app._reconciling_submission_notice_order_id = order_id
    try:
        yield
    finally:
        app._reconciling_submission_notice_order_id = None


def _verify_held_order_notice(reconciled: dict[str, Any], terms: Mapping[str, Any], notices) -> None:
    """The warning is insufficient: require exact acknowledged native order."""
    if not notices:
        return
    rows = reconciled.get("open_orders", [])
    valid = (reconciled.get("outcome") == "WORKING" and bool(rows)
             and not reconciled.get("completed_orders") and not reconciled.get("executions")
             and not (reconciled.get("last_callback") or {}).get("fatal_rejection"))
    identifiers = ("account", "client_id", "conid", "order_id", "order_ref")
    exact = ("side", "order_type", "tif", "outside_rth", "symbol", "sec_type", "currency", "transmit", "trigger_method", "good_till_date")
    prices = ("limit_price", "stop_price", "trailing_amount", "trailing_percent", "trail_stop_price", "limit_offset")
    perm_ids = set()
    for row in rows:
        valid = valid and all(row.get(key) == terms.get(key) for key in (*identifiers, *exact))
        valid = valid and row.get("status") in {"PreSubmitted", "Submitted"}
        valid = valid and type(row.get("perm_id")) is int and row["perm_id"] > 0
        valid = valid and _decimal_value(row.get("quantity")) == _decimal_value(terms["quantity"])
        valid = valid and _decimal_value(row.get("remaining_quantity")) == _decimal_value(terms["quantity"])
        for key in prices:
            if terms.get(key) is not None:
                valid = valid and row.get(key) is not None and _decimal_value(row[key]) == _decimal_value(terms[key])
        perm_ids.add(row.get("perm_id"))
    valid = valid and len(perm_ids) == 1
    reconciled["held_order_notice_reconciliation"] = {
        "exact_native_working_order_verified": bool(valid), "notices": [dict(item) for item in notices],
        "warning_alone_proves_acceptance": False, "fill_or_closure_proven": False}
    if not valid:
        raise PATWSError("session-hold warning lacks exact native working-order reconciliation")


class TWSApp(EWrapper, EClient):
    """One-connection callback collector; it never retries a request or order."""

    def __init__(self) -> None:
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        self.next_order_id: int | None = None
        self.managed_accounts: list[str] = []
        self.errors: list[dict[str, Any]] = []
        self.server_time: int | None = None
        self.market_data_types: dict[int, int] = {}
        self._quote_lock = threading.RLock()
        self._quote_snapshot_active: int | None = None
        self._quote_requested_type: int | None = None
        self._quote_snapshot_frozen: dict[str, Any] | None = None
        self._quote_modes: set[int] = set()
        self._quote_started_utc: str | None = None
        self._quote_started_mono: int | None = None
        self._paper_pnl_lock = threading.RLock()
        self._paper_pnl_session_key: str | None = None
        self._paper_pnl_minimum: Decimal | None = None
        self._paper_pnl_origins: list[dict[str, Any]] = []
        self._paper_pnl_callback_count = 0
        self._paper_pnl_window_start_count = 0
        self._paper_pnl_overflow = False
        self.account_values: list[dict[str, Any]] = []
        self.positions: list[dict[str, Any]] = []
        self.portfolio: list[dict[str, Any]] = []
        self.pnl_values: dict[str, Any] | None = None
        self.pnl_samples: list[dict[str, Any]] = []
        self.open_orders: list[dict[str, Any]] = []
        self.completed_orders: list[dict[str, Any]] = []
        self.executions: list[dict[str, Any]] = []
        self._fee_capture = _fee_capture.FeeCapture(account=ACCOUNT)
        self._execution_request_id: int | None = None
        self._execution_query: dict[str, Any] | None = None
        self.contract_details: list[dict[str, Any]] = []
        self.historical_bars: list[dict[str, Any]] = []
        self.historical_request_id: int | None = None
        self.historical_request_completed = False
        self.historical_callback_failed = False
        self.historical_completed_at_utc: str | None = None
        self.news_providers: list[dict[str, str]] = []
        self.news_providers_active = False
        self.news_providers_completed = False
        self.news_providers_callback_failed = False
        self.news_providers_ready = threading.Event()
        self.news_request_id: int | None = None
        self.news_headlines: list[dict[str, Any]] = []
        self.news_page_limit = 300
        self.news_completed = False
        self.news_callback_failed = False
        self.news_has_more: bool | None = None
        self.news_completed_at_utc: str | None = None
        self.news_ready = threading.Event()
        self.news_article_request_id: int | None = None
        self.news_article_value: dict[str, Any] | None = None
        self.news_article_callback_failed = False
        self.news_article_ready = threading.Event()
        self.live_news_lock = threading.Lock()
        self.live_news_request_id: int | None = None
        self.live_news_next_id = 9401
        self.live_news_providers: tuple[str, ...] = ()
        self.live_news_headlines: list[dict[str, Any]] = []
        self.live_news_seen: set[tuple[Any, ...]] = set()
        self.live_news_max_headlines = 100
        self.live_news_callback_count = 0
        self.live_news_callback_failed = False
        self.live_news_transport_closed = False
        self.live_news_stop_reason: str | None = None
        self.live_news_deadline = 0.0
        self.live_news_ready = threading.Event()
        self.quotes: dict[int, dict[str, Any]] = {}
        self.order_updates: dict[int, dict[str, Any]] = {}
        self.ready = threading.Event()
        self.accounts_ready = threading.Event()
        self.current_time_ready = threading.Event()
        self.account_ready = threading.Event()
        self.positions_ready = threading.Event()
        self.portfolio_ready = threading.Event()
        self._expected_account_unsubscribe = False
        self._expected_account_unsubscribe_at: float | None = None
        self._account_subscription_active = False
        self._account_unsubscribe_ack = threading.Event()
        self.pnl_ready = threading.Event()
        self.open_orders_ready = threading.Event()
        self.completed_orders_ready = threading.Event()
        self.executions_ready = threading.Event()
        self.contract_ready = threading.Event()
        self.quote_ready = threading.Event()
        self.historical_ready = threading.Event()
        self.order_events: dict[int, threading.Event] = {}
        self._thread: threading.Thread | None = None

    def nextValidId(self, orderId: int) -> None:
        self.next_order_id = max(int(orderId), self.next_order_id or 0)
        self.ready.set()

    def cancelOrder(self, orderId: int) -> None:
        # Caller ownership/confirmation gates are unchanged. Modern vendor
        # clients require an empty OrderCancel object rather than one argument.
        try:
            from ibapi.order_cancel import OrderCancel
        except ImportError:
            return EClient.cancelOrder(self, orderId)
        return EClient.cancelOrder(self, orderId, OrderCancel())

    def managedAccounts(self, accountsList: str) -> None:
        self.managed_accounts = [item.strip() for item in accountsList.split(",") if item.strip()]
        self.accounts_ready.set()

    def currentTime(self, time: int) -> None:
        self.server_time = int(time)
        self.current_time_ready.set()

    def error(
        self,
        reqId: int,
        *args,
        **kwargs,
    ) -> None:
        callback = _sdk_compat.error_callback(reqId, *args, **kwargs)
        errorCode, errorString = callback.code, callback.message
        advancedOrderRejectJson = callback.advanced_reject
        expected_unsubscribe = (
            errorCode == 2100 and self._expected_account_unsubscribe
            and not self._account_subscription_active
            and self._expected_account_unsubscribe_at is not None
            and 0 <= time.monotonic() - self._expected_account_unsubscribe_at <= 2.0
        )
        if errorCode == 2100 and self._expected_account_unsubscribe:
            self._expected_account_unsubscribe = False
            self._expected_account_unsubscribe_at = None
        if expected_unsubscribe:
            self._account_unsubscribe_ack.set()
        delayed_quote_notice = (
            reqId == 9104 and errorCode == 10167 and self._quote_snapshot_active == 9104
            and self._quote_requested_type == 3
        )
        self.errors.append(
            {
                "received_at_utc": utc_now(),
                "req_id": int(reqId),
                "code": int(errorCode),
                "message": errorString,
                "advanced_reject": advancedOrderRejectJson or None,
                "broker_error_time": callback.broker_time,
                "informational": int(errorCode) in INFORMATIONAL_ERROR_CODES or expected_unsubscribe or delayed_quote_notice,
                "classification": ("expected_self_account_unsubscribe" if expected_unsubscribe else
                                   "delayed_data_notice_for_active_quote" if delayed_quote_notice else "broker_callback"),
                "classification_reason": (
                    "one bounded notice after this client's completed PAPER_ACCOUNT download and local unsubscribe"
                    if expected_unsubscribe else "broker callback retains its normal severity"
                ),
            }
        )
        if expected_unsubscribe or delayed_quote_notice:
            # Scoped acknowledgement of this client's own reqAccountUpdates(False).
            # An unrequested 2100 remains a material broker error.
            # The active Type3 notice is not snapshot completion. Retain it,
            # then await real price/mode callbacks and tickSnapshotEnd; 354
            # still fails normally and this does not turn a missing BBO valid.
            return
        if reqId in self.order_events and errorCode not in INFORMATIONAL_ERROR_CODES:
            self.order_updates[reqId] = {
                "order_id": int(reqId),
                "status": "BROKER_ERROR_REPORTED",
                "error_code": int(errorCode),
                "error_message": errorString,
                "fatal_rejection": int(errorCode) in FATAL_ORDER_REJECTION_ERROR_CODES,
                "received_at_utc": utc_now(),
            }
            self.order_events[reqId].set()
        if errorCode not in INFORMATIONAL_ERROR_CODES and reqId == 9101:
            self.account_ready.set()
        elif errorCode not in INFORMATIONAL_ERROR_CODES and reqId == 9102:
            self.executions_ready.set()
        elif errorCode not in INFORMATIONAL_ERROR_CODES and reqId == 9103:
            self.contract_ready.set()
        elif errorCode not in INFORMATIONAL_ERROR_CODES and reqId == 9104:
            self.quote_ready.set()
        elif errorCode not in INFORMATIONAL_ERROR_CODES and reqId == 9105:
            self.pnl_ready.set()
        elif errorCode not in INFORMATIONAL_ERROR_CODES and reqId == getattr(self, "historical_request_id", None):
            self.historical_ready.set()
        elif errorCode not in INFORMATIONAL_ERROR_CODES and reqId == self.news_request_id:
            self.news_ready.set()
        elif errorCode not in INFORMATIONAL_ERROR_CODES and reqId == self.news_article_request_id:
            self.news_article_ready.set()
        elif errorCode not in INFORMATIONAL_ERROR_CODES and reqId == self.live_news_request_id:
            self.live_news_ready.set()
        if errorCode not in INFORMATIONAL_ERROR_CODES and reqId == -1:
            # Unscoped material errors must wake active requests as failures.
            if self.news_providers_active:
                self.news_providers_ready.set()
            if self.news_request_id is not None:
                self.news_ready.set()
            if self.news_article_request_id is not None:
                self.news_article_ready.set()
            if self.live_news_request_id is not None:
                self.live_news_ready.set()

    def connectionClosed(self) -> None:
        with self.live_news_lock:
            if self.live_news_request_id is not None:
                self.live_news_transport_closed = True
                self.live_news_ready.set()

    def accountSummary(
        self, reqId: int, account: str, tag: str, value: str, currency: str
    ) -> None:
        if account == ACCOUNT:
            self.account_values.append(
                {"account": account, "tag": tag, "value": value, "currency": currency}
            )

    def accountSummaryEnd(self, reqId: int) -> None:
        self.account_ready.set()

    def position(self, account: str, contract: Contract, position: Any, avgCost: float) -> None:
        if account == ACCOUNT:
            self.positions.append(
                {
                    "account": account,
                    **contract_record(contract),
                    "position": _number(position),
                    "average_cost": _number(avgCost),
                }
            )

    def positionEnd(self) -> None:
        self.positions_ready.set()

    def updatePortfolio(
        self, contract: Contract, position: Any, marketPrice: float,
        marketValue: float, averageCost: float, unrealizedPNL: float,
        realizedPNL: float, accountName: str,
    ) -> None:
        if accountName == ACCOUNT:
            self.portfolio.append({
                "account": accountName, **contract_record(contract),
                "position": _number(position), "market_price": _number(marketPrice),
                "market_value": _number(marketValue), "average_cost": _number(averageCost),
                "unrealized_pnl": _number(unrealizedPNL), "realized_pnl": _number(realizedPNL),
                "received_at_utc": utc_now(), "price_timestamp_scope": "broker_portfolio_mark",
            })

    def accountDownloadEnd(self, accountName: str) -> None:
        if accountName == ACCOUNT:
            self.portfolio_ready.set()

    def pnl(self, reqId: int, dailyPnL: float, unrealizedPnL: float, realizedPnL: float) -> None:
        _paper_trial.record_native_pnl(sys.modules[__name__], self, reqId, dailyPnL, unrealizedPnL, realizedPnL)
        if reqId == 9105:
            self.pnl_values = {
                "account": ACCOUNT, "daily_pnl": _optional_order_number(dailyPnL),
                "unrealized_pnl": _optional_order_number(unrealizedPnL),
                "realized_pnl": _optional_order_number(realizedPnL),
                "received_at_utc": utc_now(), "source": "reqPnL",
                "reset_scope": "TWS_configuration_unknown", "currency_scope": "account_base",
            }
            if self.pnl_values["daily_pnl"] is not None:
                self.pnl_samples.append(dict(self.pnl_values))
                self.pnl_samples = self.pnl_samples[-128:]
            self.pnl_ready.set()

    def openOrder(self, orderId: int, contract: Contract, order: Order, orderState: Any) -> None:
        self.next_order_id = max(self.next_order_id or 0, int(orderId) + 1)
        if order.account == ACCOUNT:
            self.open_orders.append(order_record(orderId, contract, order, orderState.status))

    def openOrderEnd(self) -> None:
        self.open_orders_ready.set()

    def completedOrder(self, contract: Contract, order: Order, orderState: Any) -> None:
        if order.account == ACCOUNT:
            self.completed_orders.append(order_record(order.orderId, contract, order, orderState.status))

    def completedOrdersEnd(self) -> None:
        self.completed_orders_ready.set()

    def execDetails(self, reqId: int, contract: Contract, execution: Any) -> None:
        if execution.acctNumber == ACCOUNT:
            self.executions.append(
                {
                    "account": execution.acctNumber,
                    "exec_id": execution.execId,
                    "order_id": int(execution.orderId),
                    "perm_id": int(execution.permId),
                    "client_id": int(execution.clientId),
                    "order_ref": execution.orderRef,
                    "side": execution.side,
                    "shares": _number(execution.shares),
                    "price": _number(execution.price),
                    "time": execution.time,
                    **contract_record(contract),
                }
            )

    def execDetailsEnd(self, reqId: int) -> None:
        if reqId == self._execution_request_id:
            assert self._execution_query is not None
            self._execution_query["exec_details_end_received_at_utc"] = utc_now()
            self.executions_ready.set()

    def contractDetails(self, reqId: int, contractDetails: Any) -> None:
        item = contract_record(contractDetails.contract)
        item.update(
            {
                "long_name": contractDetails.longName,
                "market_name": contractDetails.marketName,
                "min_tick": _number(contractDetails.minTick),
                "valid_exchanges": contractDetails.validExchanges,
                "time_zone_id": contractDetails.timeZoneId,
                "trading_hours": contractDetails.tradingHours,
                "liquid_hours": contractDetails.liquidHours,
            }
        )
        self.contract_details.append(item)

    def contractDetailsEnd(self, reqId: int) -> None:
        self.contract_ready.set()

    def historicalData(self, reqId: int, bar: Any) -> None:
        if reqId == self.historical_request_id and not self.historical_callback_failed:
            try:
                if len(self.historical_bars) >= 1000:
                    raise ValueError("historical callback limit exceeded")
                self.historical_bars.append({
                    "native_date": str(bar.date), "open": _number(bar.open),
                    "high": _number(bar.high), "low": _number(bar.low),
                    "close": _number(bar.close), "volume": _number(bar.volume),
                    # Installed ibapi.common.BarData uses average; wap belongs to
                    # RealTimeBar. The average is optional for OHLCV research.
                    "average": _optional_order_number(getattr(bar, "average", None)),
                    "bar_count": _integer(bar.barCount), "received_at_utc": utc_now(),
                })
            except (AttributeError, TypeError, ValueError, OverflowError):
                # Callback failure must wake the request, not kill the API reader
                # thread and turn a malformed bar into an opaque timeout.
                self.historical_callback_failed = True
                self.historical_ready.set()

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        if reqId == self.historical_request_id and not self.historical_callback_failed:
            self.historical_completed_at_utc = utc_now()
            self.historical_request_completed = True
            self.historical_ready.set()

    def newsProviders(self, newsProviders: Any) -> None:
        if not self.news_providers_active:
            return
        try:
            if not isinstance(newsProviders, (list, tuple)) or len(newsProviders) > 100:
                raise ValueError("invalid news provider callback")
            providers = []
            for provider in newsProviders:
                if (not isinstance(provider.code, str)
                        or not NEWS_PROVIDER_RE.fullmatch(provider.code)
                        or not isinstance(provider.name, str) or not provider.name):
                    raise ValueError("invalid news provider")
                providers.append({"code": provider.code, "name": provider.name})
            if len({item["code"] for item in providers}) != len(providers):
                raise ValueError("duplicate news provider")
            self.news_providers = providers
            self.news_providers_completed = True
        except (AttributeError, ValueError, TypeError):
            self.news_providers_callback_failed = True
        self.news_providers_ready.set()

    def historicalNews(self, requestId: int, time: Any, providerCode: str,
                       articleId: str, headline: str) -> None:
        if requestId != self.news_request_id or self.news_callback_failed or self.news_completed:
            return
        try:
            if (len(self.news_headlines) >= self.news_page_limit
                    or not isinstance(providerCode, str) or not NEWS_PROVIDER_RE.fullmatch(providerCode)
                    or not isinstance(articleId, str) or not NEWS_ARTICLE_RE.fullmatch(articleId)
                    or not isinstance(headline, str) or not headline or len(headline) > 100_000):
                raise ValueError("invalid historical news callback")
            self.news_headlines.append({
                **news_publication_time(time), "provider_code": providerCode,
                "article_id": articleId, "headline": headline, "received_at_utc": utc_now(),
            })
        except (ValueError, TypeError, OverflowError, OSError):
            self.news_callback_failed = True
            self.news_ready.set()

    def historicalNewsEnd(self, requestId: int, hasMore: bool) -> None:
        if requestId != self.news_request_id or self.news_callback_failed:
            return
        if type(hasMore) is not bool:
            self.news_callback_failed = True
        else:
            self.news_completed = True
            self.news_has_more = hasMore
            self.news_completed_at_utc = utc_now()
        self.news_ready.set()

    def tickNews(self, tickerId: int, timeStamp: int, providerCode: str,
                 articleId: str, headline: str, extraData: str) -> None:
        with self.live_news_lock:
            if (tickerId != self.live_news_request_id or self.live_news_callback_failed
                    or self.live_news_stop_reason is not None):
                return
            if time.monotonic() >= self.live_news_deadline:
                self.live_news_stop_reason = "observation_window_elapsed"
                self.live_news_ready.set()
                return
            try:
                if (providerCode not in self.live_news_providers
                        or not isinstance(articleId, str) or not NEWS_ARTICLE_RE.fullmatch(articleId)
                        or not isinstance(headline, str) or not headline or len(headline) > 100_000
                        or not isinstance(extraData, str) or len(extraData) > 100_000):
                    raise ValueError("invalid or unrequested live news callback")
                headline.encode("utf-8")
                extraData.encode("utf-8")
                publication = live_news_publication_time(timeStamp)
                self.live_news_callback_count += 1
                identity = (providerCode, articleId, timeStamp, headline, extraData)
                if identity not in self.live_news_seen:
                    self.live_news_seen.add(identity)
                    self.live_news_headlines.append({
                        **publication, "request_id": tickerId, "provider_code": providerCode,
                        "article_id": articleId, "headline": headline, "extra_data": extraData,
                        "received_at_utc": utc_now(),
                    })
                if len(self.live_news_headlines) >= self.live_news_max_headlines:
                    self.live_news_stop_reason = "headline_limit"
                elif self.live_news_callback_count >= 10_000:
                    self.live_news_stop_reason = "callback_limit"
                if self.live_news_stop_reason is not None:
                    self.live_news_ready.set()
            except (ValueError, TypeError, OverflowError, OSError, UnicodeError):
                self.live_news_callback_failed = True
                self.live_news_ready.set()

    def newsArticle(self, requestId: int, articleType: int, articleText: str) -> None:
        if requestId != self.news_article_request_id:
            return
        try:
            if (type(articleType) is not int or articleType not in {0, 1}
                    or not isinstance(articleText, str) or not articleText
                    or len(articleText.encode("utf-8")) > 2_000_000):
                raise ValueError("invalid news article callback")
            self.news_article_value = {
                "article_type": articleType, "content": articleText,
                "content_encoding": "text_or_html" if articleType == 0 else "base64_binary",
                "received_at_utc": utc_now(),
            }
        except (UnicodeError, TypeError, ValueError):
            self.news_article_callback_failed = True
        self.news_article_ready.set()

    def marketDataType(self, reqId: int, marketDataType: int) -> None:
        with self._quote_lock:
            if reqId == 9104 and self._quote_snapshot_frozen is not None:
                return
            if reqId == self._quote_snapshot_active:
                self._quote_modes.add(int(marketDataType))
            self.market_data_types[int(reqId)] = int(marketDataType)

    def tickPrice(self, reqId: int, tickType: int, price: float, attrib: Any) -> None:
        with self._quote_lock:
            if reqId == 9104 and self._quote_snapshot_frozen is not None:
                return
            self._tick_price_locked(reqId, tickType, price, attrib)

    def _tick_price_locked(self, reqId: int, tickType: int, price: float, attrib: Any) -> None:
        names = {
            1: "bid",
            2: "ask",
            4: "last",
            6: "high",
            7: "low",
            9: "close",
            66: "bid",
            67: "ask",
            68: "last",
            72: "high",
            73: "low",
            75: "close",
        }
        if tickType in names and price >= 0:
            self.quotes.setdefault(reqId, {})[names[tickType]] = _number(price)
            if names[tickType] in {"bid", "ask"}:
                self.quotes[reqId][names[tickType] + "_received_at_utc"] = utc_now()
                self.quotes[reqId][names[tickType] + "_tick_type"] = int(tickType)
            self.quotes[reqId]["source_tick_type"] = int(tickType)
            self.quotes[reqId]["received_at_utc"] = utc_now()

    def tickSize(self, reqId: int, tickType: int, size: Any) -> None:
        with self._quote_lock:
            if reqId == 9104 and self._quote_snapshot_frozen is not None:
                return
            self._tick_size_locked(reqId, tickType, size)

    def _tick_size_locked(self, reqId: int, tickType: int, size: Any) -> None:
        names = {
            0: "bid_size",
            3: "ask_size",
            5: "last_size",
            8: "volume",
            69: "bid_size",
            70: "ask_size",
            71: "last_size",
            74: "volume",
        }
        if tickType in names:
            self.quotes.setdefault(reqId, {})[names[tickType]] = _number(size)
            self.quotes[reqId]["received_at_utc"] = utc_now()

    def tickString(self, reqId: int, tickType: int, value: str) -> None:
        with self._quote_lock:
            if reqId == 9104 and self._quote_snapshot_frozen is not None:
                return
            self._tick_string_locked(reqId, tickType, value)

    def _tick_string_locked(self, reqId: int, tickType: int, value: str) -> None:
        quote = self.quotes.setdefault(reqId, {})
        if tickType in {45, 88}:
            quote["last_timestamp_epoch"] = value
        elif tickType == 48:
            quote["rt_volume_raw"] = value
        quote["received_at_utc"] = utc_now()

    def tickSnapshotEnd(self, reqId: int) -> None:
        with self._quote_lock:
            if reqId == self._quote_snapshot_active and self._quote_snapshot_frozen is None:
                self._quote_snapshot_frozen = {
                    "quote": dict(self.quotes.get(reqId, {})),
                    "callback_type": self.market_data_types.get(reqId), "modes": set(self._quote_modes),
                    "completed_utc": utc_now(), "completed_mono": time.monotonic_ns()}
                self.quote_ready.set()

    def orderStatus(
        self,
        orderId: int,
        status: str,
        filled: Any,
        remaining: Any,
        avgFillPrice: float,
        permId: int,
        parentId: int,
        lastFillPrice: float,
        clientId: int,
        whyHeld: str,
        mktCapPrice: float = 0.0,
    ) -> None:
        self.order_updates[orderId] = {
            "order_id": int(orderId),
            "status": status,
            "filled": _number(filled),
            "remaining": _number(remaining),
            "average_fill_price": _number(avgFillPrice),
            "last_fill_price": _number(lastFillPrice),
            "perm_id": int(permId),
            "client_id": int(clientId),
            "why_held": whyHeld,
            "received_at_utc": utc_now(),
        }
        self.order_events.setdefault(orderId, threading.Event()).set()

    def connect_checked(self, client_id: int, timeout: float) -> None:
        validate_boundary(client_id=client_id)
        self.connect(HOST, PORT, client_id)
        if not self.isConnected():
            raise PATWSError(f"could not connect to TWS at {HOST}:{PORT}")
        _fee_capture.attach_decoder(self.decoder, self._fee_capture)
        self._thread = threading.Thread(target=self.run, name="pa-tws-api", daemon=True)
        self._thread.start()
        self._wait(self.ready, timeout, "nextValidId")
        if not self.accounts_ready.wait(min(timeout, 2.0)):
            self.reqManagedAccts()
            self._wait(self.accounts_ready, timeout, "managedAccounts")
        assert_account_access(self.managed_accounts)
        self.raise_material_errors(0)

    def raise_material_errors(
        self, since: int, request_ids: set[int] | None = None
    ) -> None:
        scoped = []
        for item in self.errors[since:]:
            if item["informational"]:
                continue
            order_id = getattr(self, "_reconciling_submission_notice_order_id", None)
            if order_id is not None and _known_held_order_notice(item, order_id):
                # Retain original severity/raw text. Only the scoped caller may
                # collect evidence, then independently verify exact held terms.
                continue
            if request_ids is None or item["req_id"] == -1 or item["req_id"] in request_ids:
                scoped.append(item)
        if scoped:
            raise PATWSError(f"material TWS error(s): {json.dumps(scoped, sort_keys=True)}")

    @staticmethod
    def _wait(event: threading.Event, timeout: float, name: str) -> None:
        if not event.wait(timeout):
            raise BrokerTimeout(f"timed out waiting for {name}; outcome may be uncertain")

    def disconnect_clean(self) -> None:
        if self.isConnected():
            self.disconnect()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if (getattr(self, "_paper_trial_audit", None) is not None
                and self._thread and self._thread.is_alive()):
            self._paper_pnl_final_reader_unsettled = True
            _paper_trial.persist_final_native_pnl(sys.modules[__name__], self)
            raise PolicyError("paper trial owned reader did not stop; session entries halted")
        _paper_trial.persist_final_native_pnl(sys.modules[__name__], self)

    def get_status(self, timeout: float) -> dict[str, Any]:
        error_start = len(self.errors)
        self.current_time_ready.clear()
        self.reqCurrentTime()
        self._wait(self.current_time_ready, timeout, "currentTime")
        self.raise_material_errors(error_start)
        return {
            "connected": self.isConnected(),
            "host": HOST,
            "port": PORT,
            "client_id": self.clientId,
            "account": ACCOUNT,
            "managed_accounts": self.managed_accounts,
            "server_time_epoch": self.server_time,
            "next_valid_order_id_observed": self.next_order_id,
            "errors": self.errors,
        }

    def get_account(self, timeout: float) -> list[dict[str, Any]]:
        error_start = len(self.errors)
        self.account_values.clear()
        self.account_ready.clear()
        request_id = 9101
        tags = (
            "NetLiquidation,TotalCashValue,GrossPositionValue,UnrealizedPnL,"
            "AvailableFunds,ExcessLiquidity,BuyingPower,InitMarginReq,MaintMarginReq,Leverage"
        )
        self.reqAccountSummary(request_id, "All", tags)
        self._wait(self.account_ready, timeout, "accountSummaryEnd")
        self.cancelAccountSummary(request_id)
        self.raise_material_errors(error_start, {request_id})
        return list(self.account_values)

    def get_positions(self, timeout: float) -> list[dict[str, Any]]:
        error_start = len(self.errors)
        self.positions.clear()
        self.positions_ready.clear()
        self.reqPositions()
        self._wait(self.positions_ready, timeout, "positionEnd")
        self.cancelPositions()
        self.raise_material_errors(error_start)
        return list(self.positions)

    def get_portfolio(self, timeout: float) -> list[dict[str, Any]]:
        error_start = len(self.errors)
        self.portfolio.clear()
        self.portfolio_ready.clear()
        self._expected_account_unsubscribe = False
        self._expected_account_unsubscribe_at = None
        self._account_unsubscribe_ack.clear()
        self._account_subscription_active = True
        self.reqAccountUpdates(True, ACCOUNT)
        download_complete = False
        try:
            self._wait(self.portfolio_ready, timeout, "accountDownloadEnd")
            self.raise_material_errors(error_start)
            download_complete = True
        finally:
            self._account_subscription_active = False
            self._expected_account_unsubscribe = download_complete
            self._expected_account_unsubscribe_at = time.monotonic() if self._expected_account_unsubscribe else None
            self.reqAccountUpdates(False, ACCOUNT)
        self.raise_material_errors(error_start)
        return list(self.portfolio)

    def get_pnl(self, timeout: float) -> dict[str, Any]:
        observation_error_start = len(self.errors)
        # Activate account portfolio data before trusting a potentially partial
        # first reqPnL callback during TWS cache warmup.
        positions = self.get_positions(timeout)
        self.get_portfolio(timeout)
        nonflat = any(_decimal_value(item["position"]) != 0 for item in positions)
        error_start = len(self.errors)
        self.pnl_values = None
        self.pnl_samples.clear()
        self.pnl_ready.clear()
        started = time.monotonic()
        self.reqPnL(9105, ACCOUNT, "")
        try:
            self._wait(self.pnl_ready, timeout, "pnl")
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= 2.0 and (not nonflat or len(self.pnl_samples) >= 2):
                    break
                if elapsed >= timeout:
                    raise PolicyError("native PnL did not settle within the bounded sampling window")
                self.raise_material_errors(error_start, {9105})
                threading.Event().wait(min(0.1, timeout - elapsed))
        finally:
            self.cancelPnL(9105)
        self.raise_material_errors(observation_error_start)
        samples = list(self.pnl_samples)
        if not samples or (nonflat and len(samples) < 2):
            raise PolicyError("native daily PnL is unavailable or unset")
        latest = samples[-1]
        return {**latest, "daily_pnl": min(item["daily_pnl"] for item in samples),
                "latest_daily_pnl": latest["daily_pnl"], "callback_count": len(samples),
                "sampling_window_seconds": time.monotonic() - started,
                "nonflat_positions_observed": nonflat,
                "daily_pnl_measurement": "minimum_valid_native_callback_in_sampling_window",
                "aggregate_warmup_completeness": "not_proven_by_sampling",
                "broker_notices": [dict(item) for item in self.errors[observation_error_start:]]}

    def get_open_orders(self, timeout: float, *, all_clients: bool) -> list[dict[str, Any]]:
        error_start = len(self.errors)
        self.open_orders.clear()
        self.open_orders_ready.clear()
        if all_clients:
            self.reqAllOpenOrders()
        else:
            self.reqOpenOrders()
        self._wait(self.open_orders_ready, timeout, "openOrderEnd")
        self.raise_material_errors(error_start)
        return list(self.open_orders)

    def get_completed_orders(self, timeout: float) -> list[dict[str, Any]]:
        error_start = len(self.errors)
        self.completed_orders.clear()
        self.completed_orders_ready.clear()
        self.reqCompletedOrders(False)
        self._wait(self.completed_orders_ready, timeout, "completedOrdersEnd")
        self.raise_material_errors(error_start)
        return list(self.completed_orders)

    def get_executions(self, timeout: float) -> list[dict[str, Any]]:
        error_start = len(self.errors)
        deadline = time.monotonic() + timeout
        self.executions.clear()
        self.executions_ready.clear()
        request_id = 9102
        self._execution_request_id = request_id
        execution_filter = ExecutionFilter()
        execution_filter.acctCode = ACCOUNT
        self._execution_query = {
            "request_id": request_id,
            "account": ACCOUNT,
            "connected_client_id": self.clientId,
            "filter": vars(execution_filter).copy(),
            "started_at_utc": utc_now(),
            "exec_details_end_received_at_utc": None,
            "fee_wait_completed": False,
            "all_account_history_complete": None,
            "account_total_fees": None,
        }
        self.reqExecutions(request_id, execution_filter)
        self._wait(self.executions_ready, max(0.0, deadline - time.monotonic()), "execDetailsEnd")
        self.raise_material_errors(error_start, {request_id})
        if self._execution_query["exec_details_end_received_at_utc"] is None:
            raise BrokerTimeout("execution end was not observed; outcome may be uncertain")
        executions = list(self.executions)
        self._execution_query["fee_wait_completed"] = self._fee_capture.wait_for_ids(
            (row["exec_id"] for row in executions), deadline
        )
        self.raise_material_errors(error_start, {request_id})
        self._execution_query["snapshot_at_utc"] = utc_now()
        return self._fee_capture.enrich(executions)

    def get_execution_report(self, timeout: float) -> dict[str, Any]:
        executions = self.get_executions(timeout)
        # Capture rows and raw report references under the same collector lock.
        # Internal callers retain matched raw; the CLI exposes each report once.
        report = self._fee_capture.report(executions)
        report["query"] = dict(self._execution_query or {})
        report["query"]["snapshot_at_utc"] = report.pop("snapshot_at_utc")
        return report

    def get_contracts(self, contract: Contract, timeout: float) -> list[dict[str, Any]]:
        error_start = len(self.errors)
        self.contract_details.clear()
        self.contract_ready.clear()
        self.reqContractDetails(9103, contract)
        self._wait(self.contract_ready, timeout, "contractDetailsEnd")
        self.raise_material_errors(error_start, {9103})
        return list(self.contract_details)

    def qualify_exact_contract(self, contract: Contract, timeout: float) -> Contract:
        details = self.get_contracts(contract, timeout)
        if len(details) != 1:
            raise PolicyError(f"contract must resolve uniquely; matches={len(details)}")
        item = details[0]
        if item["sec_type"] != "STK" or item["currency"] != "USD":
            raise PolicyError("resolved contract is not a USD stock")
        if contract.conId and item["conid"] != contract.conId:
            raise PolicyError("resolved conId differs from requested conId")
        if contract.symbol and item["symbol"] != contract.symbol:
            raise PolicyError("resolved symbol differs from requested symbol")
        if contract.primaryExchange and item["primary_exchange"] != contract.primaryExchange:
            raise PolicyError("resolved primary listing differs from requested primary listing")
        return make_stock_contract(
            conid=item["conid"],
            symbol=item["symbol"],
            primary_exchange=item["primary_exchange"],
        )

    def get_quote(self, contract: Contract, timeout: float, market_data_type: int) -> dict[str, Any]:
        if type(market_data_type) is not int or market_data_type not in {1, 3}:
            raise PolicyError("market data type must be 1 (live) or 3 (delayed)")
        request_id = 9104
        error_start = len(self.errors)
        self.quotes.pop(request_id, None)
        self.market_data_types.pop(request_id, None)
        self.quote_ready.clear()
        with self._quote_lock:
            self._quote_snapshot_active = request_id
            self._quote_requested_type = market_data_type
            self._quote_snapshot_frozen = None
            self._quote_modes = set()
            self._quote_started_utc = utc_now()
            self._quote_started_mono = time.monotonic_ns()
        try:
            self.reqMarketDataType(market_data_type)
            self.reqMktData(request_id, contract, "", True, False, [])
            self._wait(self.quote_ready, timeout, "tickSnapshotEnd")
        finally:
            try:
                self.cancelMktData(request_id)
            finally:
                self._quote_snapshot_active = None
                self._quote_requested_type = None
        self.raise_material_errors(error_start, {request_id})
        frozen = self._quote_snapshot_frozen
        if frozen is None:
            raise PolicyError("quote snapshot completion is unverified")
        callback_type = frozen["callback_type"]
        if frozen["modes"] != {callback_type}:
            raise PolicyError("quote marketDataType callbacks conflict")
        if callback_type not in {1, 3}:
            raise PolicyError("quote lacks positive live/delayed marketDataType callback proof")
        if market_data_type == 1 and callback_type != 1:
            raise PolicyError("live data was requested but the callback did not prove Type 1")
        quote = frozen["quote"]
        bid = quote.get("bid")
        ask = quote.get("ask")
        last = quote.get("last")
        has_bbo = (
            isinstance(bid, (int, float))
            and isinstance(ask, (int, float))
            and bid > 0
            and ask > 0
            and ask >= bid
        )
        has_last = isinstance(last, (int, float)) and last > 0
        if not has_bbo and not has_last:
            raise PolicyError("quote lacks a positive coherent BBO or positive last trade")
        return {
            "contract": contract_record(contract),
            "requested_market_data_type": market_data_type,
            "callback_market_data_type": callback_type,
            "snapshot_complete": True,
            "snapshot_request_started_at_utc": self._quote_started_utc,
            "snapshot_completed_at_utc": frozen["completed_utc"],
            "snapshot_request_started_monotonic_ns": self._quote_started_mono,
            "snapshot_completed_monotonic_ns": frozen["completed_mono"],
            "market_event_age_seconds": "UNKNOWN_NOT_RECEIPT_AGE",
            "quality": "BBO" if has_bbo else "LAST_ONLY",
            "quote": quote,
            "errors": [item for item in self.errors if item["req_id"] in {-1, request_id}],
        }

    def get_historical_daily(self, contract: Contract, timeout: float, duration: str,
                             as_of: str | None = None) -> dict[str, Any]:
        """Completed native daily TRADES request, with a conservative day cutoff.

        The request ends at exchange-local midnight, not an invented daily close.
        Today's retrieval cannot prove historical publication/adjustment vintages.
        """
        # Script-style invocation puts pa_tws/, not the project root, on sys.path.
        # Use this checked-in repository path; no inherited PYTHONPATH is needed.
        project_root = str(PA_TWS_DIR.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from trader_runtime.historical_data import DURATIONS, normalize_daily_history
        if duration not in DURATIONS:
            raise PolicyError("historical duration must be 6 M, 1 Y or 2 Y")
        resolved = self.qualify_exact_contract(contract, timeout)
        detail = dict(self.contract_details[0])
        if detail["primary_exchange"] not in US_PRIMARY_EXCHANGES:
            raise PolicyError("daily history requires an approved U.S. primary listing")
        before = self.get_status(timeout)
        broker_time = datetime.fromtimestamp(before["server_time_epoch"], timezone.utc)
        try:
            checkpoint = broker_time if as_of is None else datetime.fromisoformat(as_of.replace("Z", "+00:00"))
            if checkpoint.tzinfo is None or checkpoint.utcoffset() is None or checkpoint > broker_time:
                raise PolicyError("historical as-of must be timezone-aware and not after broker time")
            checkpoint = checkpoint.astimezone(timezone.utc)
            zone = ZoneInfo(detail["time_zone_id"])
            local_day = checkpoint.astimezone(zone).date()
            midnight = datetime.combine(local_day, datetime.min.time(), tzinfo=zone).astimezone(timezone.utc)
        except (ValueError, TypeError, ZoneInfoNotFoundError):
            raise PolicyError("daily history requires a verified timezone and valid as-of") from None
        end_time = midnight.strftime("%Y%m%d-%H:%M:%S")
        request_id = (self.historical_request_id or 9105) + 1
        self.historical_request_id = request_id
        self.historical_bars.clear()
        self.historical_request_completed = False
        self.historical_callback_failed = False
        self.historical_completed_at_utc = None
        self.historical_ready.clear()
        error_start = len(self.errors)
        started = utc_now()
        try:
            self.reqHistoricalData(request_id, resolved, end_time, duration, "1 day", "TRADES", 1, 1, False, [])
            self._wait(self.historical_ready, timeout, "historicalDataEnd")
            self.raise_material_errors(error_start, {request_id})
            if self.historical_callback_failed:
                raise PATWSError("native historical bar callback was malformed; request is unusable")
            if not self.historical_request_completed:
                raise PATWSError("daily history reception was not completed")
        finally:
            # Completed one-shot requests require no cancellation. Cancel only an
            # interrupted request, avoiding a spurious 'no query found' response.
            if not self.historical_request_completed:
                self.cancelHistoricalData(request_id)
        raw = {
            "source": "native_tws", "mode": "paper",
            "boundary": {"host": HOST, "port": PORT, "account": ACCOUNT, "client_id": DEFAULT_CLIENT_ID},
            "contract": detail, "request_completed": True,
            "request": {"duration": duration, "bar_size": "1 day", "what_to_show": "TRADES",
                        "use_rth": True, "format_date": 1, "keep_up_to_date": False,
                        "end_date_time_utc": end_time},
            "bars": list(self.historical_bars), "as_of_utc": checkpoint.isoformat(),
            "request_started_at_utc": started, "retrieved_at_utc": utc_now(),
            "request_completed_at_utc": self.historical_completed_at_utc,
            "server_time_epoch": before["server_time_epoch"],
            "broker_checkpoint_scope": "before_request",
        }
        try:
            return normalize_daily_history(raw, contract_record(resolved))
        except RuntimeError:
            raise PolicyError("native daily history validation failed") from None

    def get_news_providers(self, timeout: float) -> dict[str, Any]:
        error_start = len(self.errors)
        self.news_providers.clear()
        self.news_providers_completed = False
        self.news_providers_callback_failed = False
        self.news_providers_ready.clear()
        started = utc_now()
        self.news_providers_active = True
        try:
            self.reqNewsProviders()
            self._wait(self.news_providers_ready, timeout, "newsProviders")
            self.raise_material_errors(error_start)
            if self.news_providers_callback_failed or not self.news_providers_completed:
                raise PATWSError("news providers callback was malformed or incomplete")
        finally:
            self.news_providers_active = False
        return {
            "source": "native_tws", "mode": "paper", "account": ACCOUNT,
            "request_completed": True, "providers": list(self.news_providers),
            "request_started_at_utc": started, "retrieved_at_utc": utc_now(),
            "provider_scope": "API provider callback; individual history/article entitlements not proven",
            "broker_mutations": 0,
        }

    def get_historical_news(self, contract: Contract, timeout: float, providers: str,
                            start_utc: str, end_utc: str, limit: int = 300,
                            max_pages: int = 1) -> dict[str, Any]:
        """Retrieve bounded end-only pages, preserving ambiguous clocks and gaps.

        IBKR documents that supplying both range ends ignores endDateTime. Each
        native request therefore has an empty start; the start cutoff is applied
        locally. Pages overlap at the oldest second and deduplicate identities.
        An unprogressing timestamp page is incomplete, never silently skipped.
        """
        codes = validate_news_providers(providers)
        start = parse_utc_timestamp(start_utc, "start-utc")
        end = parse_utc_timestamp(end_utc, "end-utc")
        if (start >= end or start.microsecond or end.microsecond
                or type(limit) is not int or not 1 <= limit <= 300
                or type(max_pages) is not int or not 1 <= max_pages <= 10):
            raise PolicyError("news requires ordered whole-second UTC bounds, limit 1..300 and max-pages 1..10")
        resolved = self.qualify_exact_contract(contract, timeout)
        detail = dict(self.contract_details[0])
        if detail["primary_exchange"] not in US_PRIMARY_EXCHANGES:
            raise PolicyError("historical news requires an approved U.S. primary listing")
        before = self.get_status(timeout)
        if end.timestamp() > before["server_time_epoch"]:
            raise PolicyError("news end-utc is after the broker checkpoint")
        started = utc_now()
        page_end, pages, rows, seen = end, [], [], {}
        reason = "page_limit"
        coverage_complete = False
        self.news_page_limit = limit
        for page_number in range(1, max_pages + 1):
            request_id = 9200 + page_number
            self.news_request_id = request_id
            self.news_headlines.clear()
            self.news_ready.clear()
            self.news_completed = False
            self.news_callback_failed = False
            self.news_has_more = None
            self.news_completed_at_utc = None
            error_start = len(self.errors)
            native_end = page_end.strftime("%Y%m%d %H:%M:%S UTC")
            try:
                self.reqHistoricalNews(request_id, resolved.conId, "+".join(codes), "", native_end, limit, [])
                self._wait(self.news_ready, timeout, "historicalNewsEnd")
                self.raise_material_errors(error_start, {request_id})
                if self.news_callback_failed:
                    raise PATWSError("historical news callback was malformed; request is unusable")
                if not self.news_completed or type(self.news_has_more) is not bool:
                    raise PATWSError("historical news reception was not completed")
                page = list(self.news_headlines)
                pages.append({"page": page_number, "request_id": request_id,
                              "native_start": "", "native_end_utc": native_end,
                              "returned_count": len(page), "has_more": self.news_has_more,
                              "completed_at_utc": self.news_completed_at_utc})
                new_rows, times = 0, []
                for row in page:
                    if row["provider_code"] not in codes:
                        raise PATWSError("historical news returned an unrequested provider")
                    identity = (row["provider_code"], row["article_id"])
                    if identity in seen:
                        previous = seen[identity]
                        if any(previous[key] != row[key] for key in ("native_time", "headline")):
                            raise PATWSError("overlapping historical news changed an article's timestamp or headline")
                    else:
                        seen[identity] = row
                        rows.append(row)
                        new_rows += 1
                    if row["published_at_utc"] is not None:
                        published = parse_utc_timestamp(row["published_at_utc"], "callback time")
                        times.append(published)
                        if published > page_end:
                            raise PATWSError("historical news returned a publication after its explicit end cutoff")
                if any(row["published_at_utc"] is None for row in page):
                    reason = "callback_timezone_unknown"
                    break
                if not self.news_has_more:
                    reason, coverage_complete = "broker_has_more_false", True
                    break
                if times and min(times) <= start:
                    reason, coverage_complete = "start_cutoff_reached", True
                    break
                if not times or not new_rows:
                    reason = "pagination_did_not_progress"
                    break
                # Inclusive overlap retains boundary timestamps; never subtract a
                # second and potentially drop other articles sharing that second.
                oldest = min(times)
                # The wire end has whole-second precision. Round upwards so a
                # fractional oldest callback cannot exclude unseen articles in
                # that same second; dedup/no-progress guards bound the overlap.
                next_end = oldest.replace(microsecond=0)
                if oldest.microsecond:
                    next_end += timedelta(seconds=1)
                if page_number > 1 and next_end >= page_end:
                    reason = "pagination_did_not_progress"
                    break
                page_end = next_end
            finally:
                # reqHistoricalNews is one-shot; no cancel API exists. A timed
                # out request is not retried and the dispatcher disconnects.
                self.news_request_id = None
        headlines, ambiguous, outside = [], [], 0
        for row in rows:
            if row["published_at_utc"] is None:
                ambiguous.append(row)
            elif start < parse_utc_timestamp(row["published_at_utc"], "callback time") <= end:
                headlines.append({**row, "published_before_requested_cutoff": True})
            else:
                outside += 1
        headlines.sort(key=lambda row: (row["published_at_utc"], row["provider_code"], row["article_id"]))
        return {
            "source": "native_tws", "mode": "paper",
            "boundary": {"host": HOST, "port": PORT, "account": ACCOUNT, "client_id": DEFAULT_CLIENT_ID},
            "contract": detail, "request_completed": True, "coverage_complete": coverage_complete,
            "completion_reason": reason, "broker_has_more": pages[-1]["has_more"],
            "request": {"providers": list(codes), "start_utc_exclusive": start.isoformat(),
                        "end_utc_inclusive": end.isoformat(), "limit_per_page": limit,
                        "max_pages": max_pages, "native_direction": "end_only_backward_with_inclusive_overlap"},
            "pages": pages, "headlines": headlines, "ambiguous_timestamp_headlines": ambiguous,
            "unique_returned_count": len(rows), "outside_window_count": outside,
            "request_started_at_utc": started, "retrieved_at_utc": utc_now(),
            "server_time_epoch": before["server_time_epoch"], "broker_checkpoint_scope": "before_request",
            "provenance": {"history_vintage": "retrieval_time", "original_article_version_verified": False,
                           "historical_arrival_time_verified": False, "historical_point_in_time_eligible": False,
                           "data_scope": "cached historical headlines from requested API providers"},
            "broker_mutations": 0,
        }

    def get_news_article(self, timeout: float, provider: str, article_id: str) -> dict[str, Any]:
        validate_news_providers(provider)
        if "+" in provider or not isinstance(article_id, str) or not NEWS_ARTICLE_RE.fullmatch(article_id):
            raise PolicyError("article retrieval requires one provider and a valid exact article ID")
        request_id = 9301
        self.news_article_request_id = request_id
        self.news_article_value = None
        self.news_article_callback_failed = False
        self.news_article_ready.clear()
        error_start = len(self.errors)
        started = utc_now()
        try:
            self.reqNewsArticle(request_id, provider, article_id, [])
            self._wait(self.news_article_ready, timeout, "newsArticle")
            self.raise_material_errors(error_start, {request_id})
            if self.news_article_callback_failed or self.news_article_value is None:
                raise PATWSError("news article callback was malformed or incomplete")
            return {
                "source": "native_tws", "mode": "paper", "account": ACCOUNT,
                "provider_code": provider, "article_id": article_id, "request_completed": True,
                **self.news_article_value, "request_started_at_utc": started, "retrieved_at_utc": utc_now(),
                "provenance": {"history_vintage": "retrieval_time", "publication_time_verified": False,
                               "original_article_version_verified": False, "historical_point_in_time_eligible": False},
                "broker_mutations": 0,
            }
        finally:
            self.news_article_request_id = None

    def get_live_news(self, contract: Contract, timeout: float, providers: str,
                      seconds: float = 10, max_headlines: int = 100,
                      *, broadtape: bool = False) -> dict[str, Any]:
        """Bounded news-only subscription; empty observation is not no-news proof."""
        validate_news_capture(seconds, max_headlines)
        codes = validate_news_providers(providers)
        if broadtape:
            candidate = make_news_contract(providers, contract.symbol)
            if contract.secType != "NEWS" or contract.exchange != providers:
                raise PolicyError("BroadTape requires an exact provider NEWS contract")
            details = self.get_contracts(candidate, timeout)
            if (len(details) != 1 or details[0]["sec_type"] != "NEWS"
                    or details[0]["symbol"] != candidate.symbol
                    or details[0]["exchange"] != candidate.exchange):
                raise PolicyError("BroadTape news contract did not resolve uniquely and exactly")
            detail = dict(details[0])
            resolved = candidate
            resolved.conId = int(detail["conid"] or 0)
            ticks = "mdoff,292"
        else:
            if (contract.secType != "STK" or contract.currency != "USD"
                    or not contract.conId or not contract.symbol
                    or contract.primaryExchange not in US_PRIMARY_EXCHANGES):
                raise PolicyError("live news requires an exact approved U.S. USD stock")
            resolved = self.qualify_exact_contract(contract, timeout)
            detail = dict(self.contract_details[0])
            if detail["primary_exchange"] not in US_PRIMARY_EXCHANGES:
                raise PolicyError("live news requires an approved U.S. primary listing")
            ticks = "mdoff,292:" + "+".join(codes)
        # A provider callback is not proof of article/history rights, but never
        # subscribe to a code absent from the current API provider inventory.
        inventory = self.get_news_providers(timeout)
        available = {item["code"] for item in inventory["providers"]}
        missing = set(codes) - available
        if missing:
            raise PolicyError("news providers absent from current API inventory: " + "+".join(sorted(missing)))
        started = utc_now()
        error_start = len(self.errors)
        with self.live_news_lock:
            if self.live_news_request_id is not None:
                raise GateError("another live news capture is already active")
            request_id = self.live_news_next_id
            self.live_news_next_id += 1
            self.live_news_request_id = request_id
            self.live_news_providers = codes
            self.live_news_headlines.clear()
            self.live_news_seen.clear()
            self.live_news_max_headlines = max_headlines
            self.live_news_callback_count = 0
            self.live_news_callback_failed = False
            self.live_news_transport_closed = False
            self.live_news_stop_reason = None
            self.live_news_ready.clear()
            self.live_news_deadline = time.monotonic() + seconds
        try:
            self.reqMktData(request_id, resolved, ticks, False, False, [])
            self.live_news_ready.wait(max(0.0, self.live_news_deadline - time.monotonic()))
            self.raise_material_errors(error_start, {request_id})
            with self.live_news_lock:
                if self.live_news_callback_failed:
                    raise PATWSError("live news callback was malformed; capture is unusable")
                if self.live_news_transport_closed or not self.isConnected():
                    raise PATWSError("connection closed during live news capture; no retry")
                rows = list(self.live_news_headlines)
                count = self.live_news_callback_count
                reason = self.live_news_stop_reason or "observation_window_elapsed"
                # Freeze the bounded capture before sending cancellation; late
                # callbacks cannot enter this result or the next request.
                self.live_news_request_id = None
        finally:
            with self.live_news_lock:
                self.live_news_request_id = None
            self.cancelMktData(request_id)
        self.raise_material_errors(error_start, {request_id})
        return {
            "source": "native_tws", "mode": "paper",
            "boundary": {"host": HOST, "port": PORT, "account": ACCOUNT, "client_id": DEFAULT_CLIENT_ID},
            "contract": detail, "news_scope": "broadtape" if broadtape else "contract_specific",
            "request": {"request_id": request_id, "providers": list(codes), "generic_tick_list": ticks,
                        "snapshot": False, "regulatory_snapshot": False,
                        "seconds": seconds, "max_headlines": max_headlines},
            "provider_inventory": inventory["providers"], "request_completed": True,
            "completion_scope": "local_bounded_observation_not_broker_history_completion",
            "completion_reason": reason, "headlines": rows, "callback_count": count,
            "duplicate_count": count - len(rows), "subscription_callback_observed": bool(rows),
            "cancellation_sent": True, "coverage_complete": False, "absence_of_news_verified": False,
            "request_started_at_utc": started, "retrieved_at_utc": utc_now(),
            "provenance": {"data_scope": "bounded live subscription; may include cached headlines",
                           "publication_timestamp_unit_verified": False,
                           "original_article_version_verified": False,
                           "historical_arrival_time_verified": False, "historical_point_in_time_eligible": False},
            "broker_mutations": 0,
        }

    def allocate_order_id(self) -> int:
        if self.next_order_id is None:
            raise PATWSError("no nextValidId received")
        order_id = self.next_order_id
        self.next_order_id += 1
        return order_id

    def wait_for_order_callback(self, order_id: int, timeout: float) -> dict[str, Any] | None:
        event = self.order_events.setdefault(order_id, threading.Event())
        if not event.wait(timeout):
            return None
        return self.order_updates.get(order_id)


def _position_for_conid(positions: Iterable[Mapping[str, Any]], conid: int) -> Decimal:
    for item in positions:
        if item.get("conid") == conid:
            return Decimal(str(item.get("position", 0)))
    return Decimal(0)


def active_sell_commitment(
    open_orders: Iterable[Mapping[str, Any]], conid: int
) -> Decimal:
    terminal = {"cancelled", "apicancelled", "filled", "inactive", "rejected"}
    committed = Decimal(0)
    for item in open_orders:
        if (
            item.get("account") != ACCOUNT
            or item.get("conid") != conid
            or str(item.get("side") or "").upper() != "SELL"
            or str(item.get("status") or "").lower() in terminal
        ):
            continue
        remaining = _decimal_value(item.get("remaining_quantity", item.get("quantity", 0)))
        if remaining > 0:
            committed += remaining
    return committed


def available_funds(account_values: Iterable[Mapping[str, Any]]) -> Decimal:
    candidates: list[Decimal] = []
    for item in account_values:
        if (
            item.get("account") == ACCOUNT
            and item.get("tag") == "AvailableFunds"
            and item.get("currency") in {"USD", "BASE"}
        ):
            candidates.append(_decimal_value(item.get("value")))
    if not candidates:
        raise PolicyError("BUY requires a positive current AvailableFunds account-summary value")
    result = min(candidates)
    if result <= 0:
        raise PolicyError("BUY requires positive AvailableFunds")
    return result


@dataclass(frozen=True)
class StockOrderSpec:
    order_type: str
    side: str
    quantity: int
    tif: str
    order_ref: str
    limit_price: Decimal | None
    stop_price: Decimal | None
    trail_amount: Decimal | None
    trail_percent: Decimal | None
    trail_stop_price: Decimal | None
    limit_offset: Decimal | None
    trigger_method: int
    good_till_date: str | None
    reference_price: Decimal | None


def stock_order_spec_from_args(args: argparse.Namespace) -> StockOrderSpec:
    order_type = str(args.order_type).upper()
    limit_price = parse_money(args.limit) if args.limit is not None else None
    stop_price = parse_money(args.stop) if args.stop is not None else None
    trail_amount = parse_money(args.trail_amount) if args.trail_amount is not None else None
    trail_percent = parse_percent(args.trail_percent) if args.trail_percent is not None else None
    trail_stop_price = (
        parse_money(args.trail_stop_price) if args.trail_stop_price is not None else None
    )
    limit_offset = parse_money(args.limit_offset) if args.limit_offset is not None else None
    good_till_date = args.good_till_date if args.good_till_date else None
    reference_price = (
        parse_money(args.reference_price) if args.reference_price is not None else None
    )
    if reference_price is None:
        reference_price = limit_price or stop_price or trail_stop_price
    if order_type == "LMT" and limit_price is not None and reference_price is not None and reference_price < limit_price:
        raise PolicyError("LMT reference price cannot understate its exact limit price")
    return StockOrderSpec(
        order_type=order_type,
        side=str(args.side).upper(),
        quantity=args.quantity,
        tif=str(args.tif).upper(),
        order_ref=args.order_ref,
        limit_price=limit_price,
        stop_price=stop_price,
        trail_amount=trail_amount,
        trail_percent=trail_percent,
        trail_stop_price=trail_stop_price,
        limit_offset=limit_offset,
        trigger_method=args.trigger_method,
        good_till_date=good_till_date,
        reference_price=reference_price,
    )


def order_from_spec(spec: StockOrderSpec, *, client_id: int) -> Order:
    return make_stock_order(
        order_type=spec.order_type,
        side=spec.side,
        quantity=spec.quantity,
        tif=spec.tif,
        order_ref=spec.order_ref,
        limit_price=spec.limit_price,
        stop_price=spec.stop_price,
        trail_amount=spec.trail_amount,
        trail_percent=spec.trail_percent,
        trail_stop_price=spec.trail_stop_price,
        limit_offset=spec.limit_offset,
        trigger_method=spec.trigger_method,
        good_till_date=spec.good_till_date,
        client_id=client_id,
    )


def _reconcile_order(
    *,
    order_id: int,
    order_ref: str,
    conid: int,
    client_id: int,
    requested_quantity: int,
    open_orders: Iterable[Mapping[str, Any]],
    completed_orders: Iterable[Mapping[str, Any]],
    executions: Iterable[Mapping[str, Any]],
    update: Mapping[str, Any] | None,
) -> dict[str, Any]:
    def matches(item: Mapping[str, Any]) -> bool:
        return (
            item.get("order_id") == order_id
            and item.get("order_ref") == order_ref
            and item.get("account") == ACCOUNT
            and item.get("client_id") == client_id
            and item.get("conid") == conid
        )

    matching_open = [dict(item) for item in open_orders if matches(item)]
    matching_completed = [dict(item) for item in completed_orders if matches(item)]
    matching_exec = [dict(item) for item in executions if matches(item)]
    unique_exec: dict[str, dict[str, Any]] = {}
    for item in matching_exec:
        unique_exec[str(item.get("exec_id") or json.dumps(item, sort_keys=True))] = item
    executed = sum(
        (_decimal_value(item.get("shares", 0)) for item in unique_exec.values()),
        Decimal(0),
    )
    callback_filled = Decimal(0)
    if update and update.get("order_id") == order_id and update.get("filled") is not None:
        callback_filled = max(_decimal_value(update["filled"]), Decimal(0))
    filled = max(executed, callback_filled)
    statuses = {
        str(item.get("status") or "").lower()
        for item in [*matching_open, *matching_completed]
    }
    callback_status = ""
    fatal_error_rejection = False
    if update and update.get("order_id") == order_id:
        callback_status = str(update.get("status") or "").lower()
        if callback_status != "broker_error_reported":
            statuses.add(callback_status)
        fatal_error_rejection = bool(update.get("fatal_rejection"))
    if filled >= Decimal(requested_quantity) or "filled" in statuses:
        outcome = "FILLED"
    elif filled > 0:
        outcome = "PARTIALLY_FILLED"
    elif statuses & {"cancelled", "apicancelled"}:
        outcome = "CANCELLED"
    elif matching_open:
        outcome = "WORKING"
    elif statuses & {"rejected", "inactive"} or fatal_error_rejection:
        outcome = "REJECTED"
    else:
        outcome = "SUBMISSION_UNCERTAIN_DO_NOT_RETRY"
    return {
        "outcome": outcome,
        "order_id": order_id,
        "order_ref": order_ref,
        "account": ACCOUNT,
        "client_id": client_id,
        "conid": conid,
        "requested_quantity": requested_quantity,
        "reconciled_filled_quantity": str(filled),
        "last_callback": dict(update) if update else None,
        "open_orders": matching_open,
        "completed_orders": matching_completed,
        "executions": list(unique_exec.values()),
    }


def _uncertain_reconciliation(
    *, order_id: int, order_ref: str, conid: int, client_id: int, error: Exception
) -> dict[str, Any]:
    return {
        "outcome": "SUBMISSION_UNCERTAIN_DO_NOT_RETRY",
        "order_id": order_id,
        "order_ref": order_ref,
        "account": ACCOUNT,
        "client_id": client_id,
        "conid": conid,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def is_liquid_session(server_epoch: int, details: Mapping[str, Any]) -> bool:
    """Use broker-provided exchange regular sessions, including early closes."""
    try:
        zone = ZoneInfo(str(details["time_zone_id"]))
        now = datetime.fromtimestamp(server_epoch, timezone.utc).astimezone(zone)
        schedule = str(details["liquid_hours"])
        for segment in schedule.split(";"):
            if ":CLOSED" in segment:
                continue
            date_prefix, intervals = segment.split(":", 1)
            for interval in intervals.split(","):
                start, end = interval.split("-", 1)
                start_text = start if ":" in start else date_prefix + ":" + start
                end_text = end if ":" in end else date_prefix + ":" + end
                begin = datetime.strptime(start_text, "%Y%m%d:%H%M").replace(tzinfo=zone)
                finish = datetime.strptime(end_text, "%Y%m%d:%H%M").replace(tzinfo=zone)
                if begin <= now < finish:
                    return True
    except (KeyError, ValueError, TypeError, ZoneInfoNotFoundError, OverflowError):
        return False
    return False


def paper_trial_execution_profile(policy: Any) -> str:
    return _paper_trial.execution_profile(sys.modules[__name__], policy)


def require_paper_delayed_quote_freshness(
    quote: Mapping[str, Any], max_age_seconds: float, now: datetime | None = None,
) -> dict[str, Any]:
    return _paper_trial.quote_freshness(sys.modules[__name__], quote, max_age_seconds, now)


def require_paper_delayed_stock_preflight(
    app: TWSApp, contract: Contract, timeout: float, max_age_seconds: float,
) -> dict[str, Any]:
    return _paper_trial.stock_preflight(sys.modules[__name__], app, contract, timeout, max_age_seconds)


def require_live_stock_preflight(
    app: TWSApp, contract: Contract, timeout: float, max_age_seconds: float,
) -> dict[str, Any]:
    """Opt-in autonomous LMT preflight performed inside the canonical lock."""
    if not math.isfinite(max_age_seconds) or max_age_seconds <= 0 or max_age_seconds > 300:
        raise PolicyError("live quote maximum age must be in (0, 300] seconds")
    if contract.primaryExchange not in US_PRIMARY_EXCHANGES:
        raise PolicyError("live preflight requires an approved U.S. primary listing")
    details = app.get_contracts(contract, timeout)
    if len(details) != 1 or details[0]["primary_exchange"] != contract.primaryExchange:
        raise PolicyError("live preflight contract listing is unverified")
    quote = app.get_quote(contract, timeout, 1)
    status = app.get_status(timeout)
    server_epoch = int(status["server_time_epoch"])
    if not is_liquid_session(server_epoch, details[0]):
        raise PolicyError("broker schedule does not prove an open regular session")
    if quote.get("callback_market_data_type") != 1 or quote.get("quality") != "BBO":
        raise PolicyError("live preflight requires a broker-proven live BBO")
    try:
        source_epoch = float(quote["quote"]["last_timestamp_epoch"])
        received = datetime.fromisoformat(quote["quote"]["received_at_utc"].replace("Z", "+00:00"))
        bbo_received = [datetime.fromisoformat(quote["quote"][name + "_received_at_utc"].replace("Z", "+00:00"))
                        for name in ("bid", "ask")]
        source_age = server_epoch - source_epoch
        receipt_age = datetime.now(timezone.utc).timestamp() - received.timestamp()
        bbo_ages = [datetime.now(timezone.utc).timestamp() - observed.timestamp() for observed in bbo_received]
    except (KeyError, ValueError, TypeError, OverflowError) as exc:
        raise PolicyError("live quote lacks valid exchange and receipt timestamps") from exc
    if not math.isfinite(source_age) or not -5 <= source_age <= max_age_seconds:
        raise PolicyError("live quote last-trade source timestamp is stale or from the future")
    if not math.isfinite(receipt_age) or not -5 <= receipt_age <= max_age_seconds:
        raise PolicyError("live quote receipt timestamp is stale or from the future")
    if any(not math.isfinite(age) or not -5 <= age <= max_age_seconds for age in bbo_ages):
        raise PolicyError("live BBO bid or ask receipt is stale or from the future")
    return {"contract_details": details[0], "quote": quote, "broker_status": status}


def validate_technical_trial_controls(
    controls: Any, *, mutation: str, client_id: int, conid: int,
    side: str | None = None, quantity: int | None = None,
    mandate_id: str | None = None, target: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate optional trial restrictions, never authenticate an ownership label.

    The root must independently verify its exact fill ledger/release before
    invoking the CLI. Hashes bind supplied custody; they are not permissions.
    Original/final closing clocks remain duties even when an owned exit is late.
    """
    clocks = {"released_at", "release_expires_at", "checkpoint_expires_at",
              "trial_expires_at", "decision_not_after_at", "entry_cutoff_at",
              "closing_deadline_at", "session_open_at", "session_close_at"}
    fields = clocks | {"schema", "purpose", "trial_id", "daily_loss_policy",
                       "root_release_sha256", "mandate_sha256", "owned_exit"}
    if mutation == "CANCEL":
        fields |= {"target"}
    if (mutation not in {"SUBMIT", "CANCEL"} or type(controls) is not dict
            or set(controls) != fields or type(client_id) is not int
            or client_id != DEFAULT_CLIENT_ID or type(conid) is not int or conid <= 0
            or controls.get("schema") != "v8_bounded_technical_paper_trial_v1"
            or controls.get("purpose") != "TECHNICAL_PAPER_TRIAL"
            or controls.get("daily_loss_policy") != "ENTRY_ONLY"):
        raise PolicyError("exact fixed-paper technical trial controls required")
    def text(value: Any) -> bool:
        return (type(value) is str and 0 < len(value) <= 256 and value == value.strip()
                and not any(ord(char) < 32 for char in value))
    def pin(value: Any) -> bool:
        return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    if (not text(controls["trial_id"])
            or any(not pin(controls[key]) for key in ("root_release_sha256", "mandate_sha256"))):
        raise PolicyError("bounded trial identity and exact custody hashes required")
    try:
        encoded = json.dumps(controls, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        raise PolicyError("finite technical trial controls required") from None
    if len(encoded.encode("utf-8")) > 64 * 1024:
        raise PolicyError("technical trial controls exceed their bounded payload")
    times = {key: parse_utc_timestamp(controls[key], key) for key in clocks}
    current = datetime.now(timezone.utc) if now is None else now
    if (not isinstance(current, datetime) or current.tzinfo is None
            or current.utcoffset() != timedelta(0)):
        raise PolicyError("technical trial current clock must be explicit UTC")
    if (times["session_open_at"] >= times["session_close_at"]
            or times["released_at"] >= min(times[key] for key in
                    ("release_expires_at", "checkpoint_expires_at", "trial_expires_at"))
            or not times["entry_cutoff_at"] <= times["closing_deadline_at"] < times["trial_expires_at"]):
        raise PolicyError("technical trial clock ordering is invalid")
    end = min(times[key] for key in ("session_close_at", "release_expires_at",
                                    "checkpoint_expires_at", "trial_expires_at"))
    if not max(times["released_at"], times["session_open_at"]) <= current < end:
        raise PolicyError("technical trial release or current RTH lease is not valid")
    original_overdue = False
    late_fill_breach = False
    if mutation == "SUBMIT":
        if (mandate_id != controls["trial_id"] or side not in {"BUY", "SELL"}
                or type(quantity) is not int or not 0 < quantity <= 100
                or current >= times["decision_not_after_at"]):
            raise PolicyError("technical trial submission identity, quantity or decision clock is invalid")
        if side == "BUY":
            if controls["owned_exit"] is not None or current >= min(
                    times["entry_cutoff_at"], times["closing_deadline_at"]):
                raise PolicyError("technical trial BUY entry cutoff reached or exit attestation supplied")
        else:
            owned = controls["owned_exit"]
            owned_fields = {"source", "trial_id", "account", "client_id", "entry_intent_id",
                            "conid", "exit_quantity", "unreserved_quantity", "original_opened_at",
                            "original_deadline_at", "original_terms_sha256", "native_exec_ids"}
            if (type(owned) is not dict or set(owned) != owned_fields
                    or owned.get("source") != "ROOT_VERIFIED_EXACT_TRIAL_FILL_LEDGER"
                    or owned.get("trial_id") != controls["trial_id"] or owned.get("account") != ACCOUNT
                    or type(owned.get("client_id")) is not int or owned["client_id"] != DEFAULT_CLIENT_ID
                    or type(owned.get("conid")) is not int or owned["conid"] != conid
                    or type(owned.get("exit_quantity")) is not int or owned["exit_quantity"] != quantity
                    or type(owned.get("unreserved_quantity")) is not int
                    or owned["unreserved_quantity"] < quantity
                    or not text(owned.get("entry_intent_id")) or not pin(owned.get("original_terms_sha256"))):
                raise PolicyError("exact root-reviewed unreserved trial-fill exit attestation required")
            exec_ids = owned["native_exec_ids"]
            if (type(exec_ids) is not list or not 1 <= len(exec_ids) <= 128
                    or any(not text(value) for value in exec_ids) or len(set(exec_ids)) != len(exec_ids)):
                raise PolicyError("bounded unique original native execution identities required")
            opened = parse_utc_timestamp(owned["original_opened_at"], "original_opened_at")
            due = parse_utc_timestamp(owned["original_deadline_at"], "original_deadline_at")
            if (due > times["closing_deadline_at"] or opened > current
                    or due - opened > timedelta(days=14)):
                raise PolicyError("original owned-lot clocks are invalid or renewed beyond their bound")
            original_overdue = current >= due
            # A DAY entry can fill at/after its immutable exit duty in a
            # cancellation race. Never strand the verified reducing exit or
            # renew the missed duty: preserve both clocks and audit the breach.
            late_fill_breach = opened >= due
    else:
        observed = controls["target"]
        if (controls["owned_exit"] is not None or type(observed) is not dict
                or set(observed) != {"account", "client_id", "conid", "order_id", "order_ref"}
                or observed != target or observed.get("account") != ACCOUNT
                or type(observed.get("client_id")) is not int or observed["client_id"] != DEFAULT_CLIENT_ID
                or type(observed.get("conid")) is not int or observed["conid"] != conid
                or type(observed.get("order_id")) is not int or observed["order_id"] <= 0
                or type(observed.get("order_ref")) is not str
                or re.fullmatch(r"pa:rt-[0-9a-f]{40}", observed["order_ref"]) is None):
            raise PolicyError("technical cancellation requires its exact own target, not an exit-policy shortcut")
    return {"controls": dict(controls), "checked_at_utc": current.isoformat(),
            "original_deadline_overdue": original_overdue,
            "late_fill_deadline_breach": late_fill_breach,
            "closing_deadline_overdue": current >= times["closing_deadline_at"],
            "deadline_duties_preserved": True,
            "ownership_scope": "SUPPLIED_ROOT_VERIFIED_LEDGER_REQUIRES_INDEPENDENT_CALLER_VERIFICATION",
            "custody_hashes_are_authority": False}


def require_technical_trial_placement(
    controls: Mapping[str, Any], *, app: TWSApp, spec: StockOrderSpec,
    contract: Contract, policy: Mapping[str, Any], live: Mapping[str, Any],
    max_quote_age_seconds: float = 30.0,
) -> dict[str, Any]:
    """Final no-network gate after connected reads and again after audit fsync."""
    current = datetime.now(timezone.utc)
    checked = validate_technical_trial_controls(
        controls, mutation="SUBMIT", client_id=app.clientId, conid=contract.conId,
        side=spec.side, quantity=spec.quantity, mandate_id=policy["mandate_id"], now=current,
    )
    risk = live["portfolio_risk"]
    begin = parse_utc_timestamp(risk["batch_started_at_utc"], "risk batch start")
    finish = parse_utc_timestamp(risk["batch_finished_at_utc"], "risk batch finish")
    if not begin <= finish <= current or (current - begin).total_seconds() > policy["max_account_age_seconds"]:
        raise PolicyError("technical trial account collection expired before placement")
    if not is_liquid_session(int(current.timestamp()), live["contract_details"]):
        raise PolicyError("technical trial native regular session closed before placement")
    quote = live["quote"]
    if paper_trial_execution_profile(dict(policy)) == PAPER_DELAYED_TYPE3:
        if live.get("execution_profile") != PAPER_DELAYED_TYPE3:
            raise PolicyError("paper delayed placement profile provenance mismatch")
        mono_begin = risk.get("batch_started_monotonic_ns")
        mono_finish = risk.get("batch_finished_monotonic_ns")
        mono_current = time.monotonic_ns()
        if (type(mono_begin) is not int or type(mono_finish) is not int
                or not 0 < mono_begin <= mono_finish <= mono_current
                or (mono_current - mono_begin) / 1e9 > policy["max_account_age_seconds"]):
            raise PolicyError("paper account monotonic collection lease expired")
        freshness = require_paper_delayed_quote_freshness(
            quote, min(max_quote_age_seconds, policy["max_quote_age_seconds"]), current)
        loss = risk.get("native_paper_daily_loss")
        if not isinstance(loss, dict) or loss.get("session_key") != _paper_trial.session_key(sys.modules[__name__], current):
            raise PolicyError("paper native PnL session guard is missing")
        if spec.side == "BUY" and (not loss.get("numeric_pnl_observed") or loss.get("entry_halt_active")):
            raise PolicyError("paper session native-PnL entry halt active")
        # Observe late original callbacks before the final placement as well;
        # a breach cannot be overwritten by a recovered latest value.
        with app._paper_pnl_lock:
            current_min = app._paper_pnl_minimum
            latest = app._paper_pnl_origins[-1] if app._paper_pnl_origins else None
        loss_limit = parse_money(policy["max_daily_loss_usd"])
        if current_min is not None and current_min <= -loss_limit:
            _paper_trial.observe_native_pnl(sys.modules[__name__], app, app._paper_trial_audit,
                                            allow_missing=True, loss_limit_usd=loss_limit)
        if spec.side == "BUY" and current_min is not None and current_min <= -loss_limit:
            raise PolicyError("paper session native-PnL breach before placement")
        if spec.side == "BUY" and (latest is None or latest["daily_pnl"] is None or app._paper_pnl_overflow):
            raise PolicyError("paper session native-PnL latest numeric value unavailable")
        if current_min is not None and current_min <= -loss_limit:
            # The reducing SELL exception may have persisted a breach with
            # fsync. Recheck the original leases after that local I/O, too.
            current = datetime.now(timezone.utc)
            checked = validate_technical_trial_controls(
                controls, mutation="SUBMIT", client_id=app.clientId, conid=contract.conId,
                side=spec.side, quantity=spec.quantity, mandate_id=policy["mandate_id"], now=current)
            if ((current - begin).total_seconds() > policy["max_account_age_seconds"]
                    or (time.monotonic_ns() - mono_begin) / 1e9 > policy["max_account_age_seconds"]
                    or not finish <= current
                    or not is_liquid_session(int(current.timestamp()), live["contract_details"]) 
                    or loss["session_key"] != _paper_trial.session_key(sys.modules[__name__], current)):
                raise PolicyError("paper reducing exit evidence expired during loss persistence")
            freshness = require_paper_delayed_quote_freshness(
                quote, min(max_quote_age_seconds, policy["max_quote_age_seconds"]), current)
        checked.update(freshness)
        return checked
    if quote.get("callback_market_data_type") != 1 or quote.get("quality") != "BBO":
        raise PolicyError("technical trial requires its fresh native live BBO")
    if (type(max_quote_age_seconds) not in (int, float) or not math.isfinite(max_quote_age_seconds)
            or max_quote_age_seconds <= 0):
        raise PolicyError("technical trial CLI quote-age bound must be finite and positive")
    effective_quote_age = min(max_quote_age_seconds, policy["max_quote_age_seconds"])
    try:
        q = quote["quote"]
        source = float(q["last_timestamp_epoch"])
        ages = [current.timestamp() - source] + [(current - parse_utc_timestamp(q[key], key)).total_seconds()
                for key in ("received_at_utc", "bid_received_at_utc", "ask_received_at_utc")]
    except (KeyError, TypeError, ValueError, OverflowError):
        raise PolicyError("technical trial quote clocks are invalid before placement") from None
    if any(not math.isfinite(age) or not -5 <= age <= effective_quote_age for age in ages):
        raise PolicyError("technical trial quote expired before placement")
    checked["effective_max_quote_age_seconds"] = effective_quote_age
    return checked


def require_portfolio_risk(
    app: TWSApp, spec: StockOrderSpec, contract: Contract,
    policy: Mapping[str, Any], timeout: float, live: Mapping[str, Any],
    max_quote_age_seconds: float = 30.0,
    audit: AuditLog | None = None,
) -> dict[str, Any]:
    """Refresh and enforce the autonomous desk mandate under the mutation lock."""
    required = {
        "mandate_id", "allowed_contracts", "max_order_notional_usd", "max_symbol_exposure_usd",
        "max_gross_exposure_usd", "max_daily_loss_usd", "max_account_age_seconds", "max_quote_age_seconds",
    }
    if not isinstance(policy, dict) or not required <= policy.keys():
        raise PolicyError("autonomous portfolio risk profile is incomplete")
    if not isinstance(policy["mandate_id"], str) or not policy["mandate_id"].strip():
        raise PolicyError("autonomous portfolio mandate identity is required")
    age_limit = policy["max_account_age_seconds"]
    if type(age_limit) is not int or not 0 < age_limit <= 300:
        raise PolicyError("autonomous account evidence age must be in 1..300 seconds")
    quote_age = policy["max_quote_age_seconds"]
    if type(quote_age) is not int or not 0 < quote_age <= 300:
        raise PolicyError("autonomous quote evidence age must be in 1..300 seconds")
    approved = policy["allowed_contracts"]
    identity = (contract.conId, contract.symbol, contract.primaryExchange)
    if not isinstance(approved, list) or not any(
        isinstance(item, dict) and (item.get("conid"), item.get("symbol"), item.get("primary_exchange")) == identity
        for item in approved
    ):
        raise PolicyError("exact submitted stock is outside the autonomous mandate")
    limits = {key: parse_money(policy[key]) for key in required if key.endswith("_usd")}
    trial = None
    profile = paper_trial_execution_profile(dict(policy))
    if "technical_paper_trial" in policy:
        trial = validate_technical_trial_controls(
            policy["technical_paper_trial"], mutation="SUBMIT", client_id=app.clientId,
            conid=contract.conId, side=spec.side, quantity=spec.quantity, mandate_id=policy["mandate_id"],
        )
        ceilings = {"max_order_notional_usd": Decimal("1000.00"),
                    "max_symbol_exposure_usd": Decimal("2000.00"),
                    "max_gross_exposure_usd": Decimal("5000.00"),
                    "max_daily_loss_usd": Decimal("100.00")}
        if any(limits[key] > maximum for key, maximum in ceilings.items()) or age_limit > 60 or quote_age > 30:
            raise PolicyError("technical trial independent risk or freshness ceiling exceeded")
        # order_from_spec always sets outsideRth=False; it is not a spec field.
        if spec.order_type != "LMT" or spec.tif != "DAY":
            raise PolicyError("technical trial permits exact regular-hours LMT DAY only")
    batch_started = datetime.now(timezone.utc)
    batch_started_mono = time.monotonic_ns()
    if profile == PAPER_DELAYED_TYPE3:
        validate_boundary(client_id=app.clientId)
        if ACCOUNT not in app.managed_accounts:
            raise PolicyError("exact native managed paper account is unverified")
        _paper_trial.arm_native_pnl(sys.modules[__name__], app)
        app._paper_trial_audit = audit
        app._paper_trial_loss_limit = limits["max_daily_loss_usd"]
    positions = app.get_positions(timeout)
    orders = app.get_open_orders(timeout, all_clients=True)
    portfolio = app.get_portfolio(timeout)
    try:
        pnl = app.get_pnl(timeout)
    except PATWSError as exc:
        if profile != PAPER_DELAYED_TYPE3 or spec.side != "SELL" or trial is None:
            if profile == PAPER_DELAYED_TYPE3:
                _paper_trial.observe_native_pnl(sys.modules[__name__], app, audit, allow_missing=True,
                                                loss_limit_usd=limits["max_daily_loss_usd"])
            raise
        pnl = {"account": ACCOUNT, "daily_pnl": None, "received_at_utc": utc_now(),
               "source": "reqPnL", "missing_numeric_pnl": True, "failure": str(exc)}
    loss_observation = None
    if profile == PAPER_DELAYED_TYPE3:
        loss_observation = _paper_trial.observe_native_pnl(
            sys.modules[__name__], app, audit, allow_missing=spec.side == "SELL",
            loss_limit_usd=limits["max_daily_loss_usd"])
    account = app.get_account(timeout)
    # Price, exchange schedule and receipt/source freshness are assessed after
    # the full account-risk batch, using the tighter mandate/CLI quote limit.
    preflight = require_paper_delayed_stock_preflight if profile == PAPER_DELAYED_TYPE3 else require_live_stock_preflight
    live = preflight(app, contract, timeout, min(max_quote_age_seconds, quote_age))
    batch_finished = datetime.now(timezone.utc)
    batch_finished_mono = time.monotonic_ns()
    if (batch_finished - batch_started).total_seconds() > age_limit:
        raise PolicyError("autonomous portfolio evidence expired during connected collection")
    if profile == PAPER_DELAYED_TYPE3 and (batch_finished_mono - batch_started_mono) / 1e9 > age_limit:
        raise PolicyError("paper portfolio monotonic evidence lease expired during collection")
    for row in [*portfolio, pnl]:
        if row.get("account") != ACCOUNT:
            raise PolicyError("autonomous portfolio evidence is outside the paper account")
        try:
            observed = datetime.fromisoformat(row["received_at_utc"].replace("Z", "+00:00"))
            age = (batch_finished - observed).total_seconds()
        except (KeyError, TypeError, ValueError) as exc:
            raise PolicyError("native portfolio risk evidence lacks a valid timestamp") from exc
        if not -5 <= age <= age_limit:
            raise PolicyError("native portfolio risk evidence is stale or from the future")
    account_values = {
        row["tag"]: _decimal_value(row["value"]) for row in account
        if row.get("account") == ACCOUNT and row.get("currency") == "USD"
    }
    if account_values.get("NetLiquidation", Decimal(0)) <= 0 or "TotalCashValue" not in account_values:
        raise PolicyError("USD base-account valuation and cash are unverified")
    daily = (_decimal_value(pnl["daily_pnl"]) if pnl.get("daily_pnl") is not None else None)
    if daily is None and not (profile == PAPER_DELAYED_TYPE3 and spec.side == "SELL"):
        raise PolicyError("native numeric dailyPnL unavailable; entries paused")
    if trial is not None:
        trial = validate_technical_trial_controls(
            policy["technical_paper_trial"], mutation="SUBMIT", client_id=app.clientId,
            conid=contract.conId, side=spec.side, quantity=spec.quantity, mandate_id=policy["mandate_id"],
            now=batch_finished,
        )
    breached = (daily is not None and daily <= -limits["max_daily_loss_usd"]) or (
        loss_observation is not None and loss_observation.entry_halt_active)
    if breached and not (trial is not None and spec.side == "SELL"):
        raise PolicyError("autonomous daily loss limit reached")
    marks = {item["conid"]: item for item in portfolio}
    gross = Decimal(0)
    own_quantity = Decimal(0)
    own_mark = Decimal(0)
    for position in positions:
        amount = _decimal_value(position["position"])
        if not amount:
            continue
        mark = marks.get(position["conid"], {})
        price = _decimal_value(mark.get("market_price"))
        if (position.get("account") != ACCOUNT or position.get("sec_type") != "STK"
                or position.get("currency") != "USD" or amount < 0 or amount != amount.to_integral_value()
                or _decimal_value(mark.get("position")) != amount or price <= 0):
            raise PolicyError("autonomous risk requires verified long whole-share USD portfolio marks")
        gross += amount * price
        if position["conid"] == contract.conId:
            own_quantity, own_mark = amount, price
    price = spec.limit_price
    if price is None:
        raise PolicyError("autonomous risk requires an immutable limit price")
    notional = Decimal(spec.quantity) * price
    if notional > limits["max_order_notional_usd"]:
        raise PolicyError("autonomous order notional limit exceeded")
    terminal = {"cancelled", "apicancelled", "filled", "inactive", "rejected"}
    for item in orders:
        if item.get("account") != ACCOUNT or str(item.get("status", "")).lower() in terminal:
            continue
        remaining = _decimal_value(item.get("remaining_quantity", item.get("quantity")))
        if (item.get("sec_type") != "STK" or item.get("currency") != "USD"
                or item.get("side") not in {"BUY", "SELL"} or remaining < 0
                or remaining != remaining.to_integral_value()
                or item.get("order_type") != "LMT" or _decimal_value(item.get("limit_price")) <= 0):
            raise PolicyError("autonomous risk cannot value a non-USD-stock or fractional commitment")
    buys = [item for item in orders if item.get("account") == ACCOUNT and item.get("side") == "BUY"
            and str(item.get("status", "")).lower() not in terminal]
    reserved = Decimal(0)
    symbol_reserved = Decimal(0)
    conservative = max(price, _decimal_value(live["quote"]["quote"]["ask"]), own_mark)
    for item in buys:
        remaining = _decimal_value(item.get("remaining_quantity", item.get("quantity")))
        limit = _decimal_value(item.get("limit_price"))
        if not remaining:
            continue
        if item.get("order_type") != "LMT" or remaining < 0 or limit <= 0:
            raise PolicyError("autonomous risk cannot value an outstanding BUY commitment")
        mark_record = marks.get(item["conid"])
        commitment_price = limit
        if mark_record is not None:
            mark = _decimal_value(mark_record.get("market_price"))
            if mark <= 0:
                raise PolicyError("outstanding BUY held-stock mark is unverified")
            commitment_price = max(limit, mark)
        reserved += remaining * commitment_price
        if item["conid"] == contract.conId:
            symbol_reserved += remaining * max(limit, conservative)
    if spec.side == "SELL":
        available = own_quantity - active_sell_commitment(orders, contract.conId)
        if Decimal(spec.quantity) > available:
            raise PolicyError("autonomous SELL exceeds freshly reconciled uncommitted long position")
    elif spec.side == "BUY":
        if reserved + notional > account_values["TotalCashValue"]:
            raise PolicyError("autonomous BUY exceeds USD cash after outstanding BUY commitments")
        if gross + reserved + Decimal(spec.quantity) * conservative > limits["max_gross_exposure_usd"]:
            raise PolicyError("autonomous aggregate gross exposure limit exceeded")
        if (own_quantity + spec.quantity) * conservative + symbol_reserved > limits["max_symbol_exposure_usd"]:
            raise PolicyError("autonomous aggregate symbol exposure limit exceeded")
    result = {"mandate": dict(policy), "positions": positions, "open_orders": orders,
            "portfolio": portfolio, "pnl": pnl, "account": account,
            "gross_exposure_usd": str(gross), "reserved_buy_exposure_usd": str(reserved),
            "batch_started_at_utc": batch_started.isoformat(), "batch_finished_at_utc": batch_finished.isoformat(),
            "live_preflight": live}
    if trial is not None:
        result["technical_trial_check"] = {**trial, "daily_pnl": str(daily) if daily is not None else None,
            "daily_loss_limit": str(limits["max_daily_loss_usd"]),
            "entry_halt_active": breached,
            "reducing_owned_sell_exception": spec.side == "SELL",
            "fresh_native_uncommitted_quantity": str(own_quantity - active_sell_commitment(orders, contract.conId))}
    if loss_observation is not None:
        result["native_paper_daily_loss"] = loss_observation.to_record()
        result["execution_profile"] = profile
        result["batch_started_monotonic_ns"] = batch_started_mono
        result["batch_finished_monotonic_ns"] = batch_finished_mono
    return result


def submit_stock_order(
    app: TWSApp,
    args: argparse.Namespace,
    audit: AuditLog,
    lock: MutationLock | None = None,
) -> dict[str, Any]:
    require_mutation_context(audit, lock)
    require_mutation_gate("submit", args.confirm)
    validate_order_ref(args.order_ref)
    if args.conid <= 0:
        raise PolicyError("mutation requires a positive exact conId")
    if audit.has_attempt("paper_order_submission_attempt", args.order_ref):
        raise PolicyError(
            "this orderRef already has a local submission attempt; reconcile manually and never retry it"
        )
    require_live = getattr(args, "require_live_preflight", False)
    requested_paper_profile = getattr(args, "paper_trial_profile", None)
    if requested_paper_profile not in (None, PAPER_DELAYED_TYPE3):
        raise PolicyError("unsupported paper execution profile")
    require_connected_preflight = require_live or requested_paper_profile is not None
    portfolio_policy = None
    duplicate_profile_keys = []
    def profile_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                duplicate_profile_keys.append(key)
            result[key] = value
        return result
    raw_profile = getattr(args, "portfolio_risk_policy_json", None)
    if raw_profile is not None or require_connected_preflight:
        try:
            portfolio_policy = json.loads(raw_profile, object_pairs_hook=profile_object)
        except (TypeError, ValueError):
            if require_connected_preflight:
                raise PolicyError("autonomous live preflight requires a valid portfolio risk profile") from None
        if isinstance(portfolio_policy, dict) and "technical_paper_trial" in portfolio_policy:
            if not require_connected_preflight or duplicate_profile_keys:
                raise PolicyError("technical trial requires connected preflight and unambiguous portfolio JSON")
            if type(raw_profile) is not str or len(raw_profile.encode("utf-8")) > 64 * 1024:
                raise PolicyError("bounded technical trial portfolio JSON required")
        profile = paper_trial_execution_profile(portfolio_policy)
        if (profile == PAPER_DELAYED_TYPE3) != (requested_paper_profile == PAPER_DELAYED_TYPE3):
            raise PolicyError("paper delayed execution requires matching CLI and portfolio profile opt-ins")
    policy = RiskPolicy.load(Path(args.risk_policy))
    spec = stock_order_spec_from_args(args)
    order = order_from_spec(spec, client_id=args.client_id)
    contract = app.qualify_exact_contract(
        make_stock_contract(
            conid=args.conid, symbol=args.symbol, primary_exchange=args.primary_exchange
        ),
        args.timeout,
    )
    conid = int(contract.conId)
    positions = app.get_positions(args.timeout)
    open_orders = app.get_open_orders(args.timeout, all_clients=True)
    gross_position = _position_for_conid(positions, conid)
    sell_commitment = active_sell_commitment(open_orders, conid)
    sell_available = gross_position - sell_commitment
    policy.validate_order(
        side=args.side,
        quantity=args.quantity,
        order_type=spec.order_type,
        reference_price=spec.reference_price,
        tif=args.tif,
        current_position=sell_available,
    )
    funds: Decimal | None = None
    if args.side == "BUY":
        funds = available_funds(app.get_account(args.timeout))
        if spec.reference_price is None:
            raise PolicyError("BUY requires a positive price reference for the notional check")
        notional = Decimal(args.quantity) * spec.reference_price
        if funds < notional:
            raise PolicyError("BUY notional exceeds positively observed AvailableFunds")
    completed = app.get_completed_orders(args.timeout)
    executions = app.get_executions(args.timeout)
    if duplicate_order_ref(args.order_ref, [*open_orders, *completed, *executions]):
        raise PolicyError("orderRef already appears in broker evidence; no submission attempted")
    live_preflight = None
    if require_connected_preflight:
        if spec.order_type != "LMT" or spec.tif != "DAY":
            raise PolicyError("autonomous live preflight permits LMT DAY only")
        preflight = require_paper_delayed_stock_preflight if requested_paper_profile else require_live_stock_preflight
        live_preflight = preflight(
            app, contract, args.timeout, args.max_quote_age_seconds,
        )
        portfolio_risk = require_portfolio_risk(
            app, spec, contract, portfolio_policy, args.timeout, live_preflight, args.max_quote_age_seconds,
            audit=audit,
        )
        live_preflight = portfolio_risk.pop("live_preflight")
        live_preflight["portfolio_risk"] = portfolio_risk
        if "technical_paper_trial" in portfolio_policy:
            if re.fullmatch(r"pa:rt-[0-9a-f]{40}", args.order_ref) is None:
                raise PolicyError("technical trial submission requires its exact runtime order namespace")
            require_technical_trial_placement(portfolio_policy["technical_paper_trial"],
                app=app, spec=spec, contract=contract, policy=portfolio_policy, live=live_preflight,
                max_quote_age_seconds=args.max_quote_age_seconds)
    order_id = app.allocate_order_id()
    terms = order_record(order_id, contract, order, "ATTEMPT_PENDING")
    audit.append(
        "paper_order_submission_attempt",
        {
            "boundary": {"host": HOST, "port": PORT, "account": ACCOUNT, "client_id": args.client_id},
            "terms": terms,
            "gross_position": str(gross_position),
            "active_sell_commitment": str(sell_commitment),
            "sell_available": str(sell_available),
            "available_funds": str(funds) if funds is not None else None,
            "gate": "dual_gate_passed",
            "live_preflight": live_preflight,
        },
    )
    # Exactly one call.  Any timeout/exception after this line is uncertain and
    # must be reconciled manually; the utility never retries or reprices.
    app.order_events[order_id] = threading.Event()
    wire_attempted = False
    reconciled = None
    wire_error_start = len(app.errors)
    try:
        placement = None
        if live_preflight and "technical_paper_trial" in portfolio_policy:
            placement = require_technical_trial_placement(portfolio_policy["technical_paper_trial"],
                app=app, spec=spec, contract=contract, policy=portfolio_policy, live=live_preflight,
                max_quote_age_seconds=args.max_quote_age_seconds)
            audit.append("paper_order_technical_trial_pre_wire_check", placement)
            # Recheck immediately after that audit fsync, immediately before
            # placeOrder. A late duty never changes its original deadline.
            placement = require_technical_trial_placement(portfolio_policy["technical_paper_trial"],
                app=app, spec=spec, contract=contract, policy=portfolio_policy, live=live_preflight,
                max_quote_age_seconds=args.max_quote_age_seconds)
        wire_attempted = True
        app.placeOrder(order_id, contract, order)
        update = app.wait_for_order_callback(order_id, args.timeout)
        with _submission_notice_scope(app, order_id):
            reconciled = _reconcile_order(
                order_id=order_id,
                order_ref=args.order_ref,
                conid=conid,
                client_id=args.client_id,
                requested_quantity=args.quantity,
                open_orders=app.get_open_orders(args.timeout, all_clients=False),
                completed_orders=app.get_completed_orders(args.timeout),
                executions=app.get_executions(args.timeout),
                update=update,
            )
            reconciled["positions_after"] = app.get_positions(args.timeout)
            # A target-order rejection can arrive during a differently scoped
            # read (for example executions). Recheck the full wire interval.
            app.raise_material_errors(wire_error_start)
        notices = [item for item in app.errors[wire_error_start:] if _known_held_order_notice(item, order_id)]
        _verify_held_order_notice(reconciled, terms, notices)
        reconciled["errors"] = app.errors
        if placement is not None:
            reconciled["technical_trial_placement_check"] = placement
        audit.append("paper_order_submission_reconciliation", reconciled)
        return reconciled
    except Exception as exc:
        uncertain = _uncertain_reconciliation(
            order_id=order_id,
            order_ref=args.order_ref,
            conid=conid,
            client_id=args.client_id,
            error=exc,
        )
        if reconciled is not None:
            uncertain["retained_post_wire_reconciliation"] = reconciled
        uncertain["errors"] = list(app.errors)
        if live_preflight and "technical_paper_trial" in portfolio_policy:
            uncertain["broker_mutation_attempted"] = wire_attempted
            uncertain["failure_phase"] = "AFTER_PLACE_ORDER_CALL" if wire_attempted else "PRE_WIRE_TRIAL_GATE"
        audit.append("paper_order_submission_reconciliation", uncertain)
        raise PATWSError(
            f"post-submission failure; order is uncertain and must not be retried: {exc}"
        ) from exc


def submit_stock_limit(
    app: TWSApp,
    args: argparse.Namespace,
    audit: AuditLog,
    lock: MutationLock | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper for the original stock-limit command."""

    return submit_stock_order(app, args, audit, lock)


def cancel_pa_order(
    app: TWSApp,
    args: argparse.Namespace,
    audit: AuditLog,
    lock: MutationLock | None = None,
) -> dict[str, Any]:
    require_mutation_context(audit, lock)
    require_mutation_gate("cancel", args.confirm)
    validate_order_ref(args.order_ref)
    if args.conid <= 0:
        raise PolicyError("cancellation requires a positive exact conId")
    if audit.has_attempt("paper_order_cancel_attempt", args.order_ref, args.order_id):
        raise PolicyError(
            "this exact cancellation already has a local attempt; reconcile manually and never retry it"
        )
    target = {"account": ACCOUNT, "client_id": args.client_id, "conid": args.conid,
              "order_id": args.order_id, "order_ref": args.order_ref}
    controls = None
    controls_raw = getattr(args, "technical_trial_controls_json", None)
    if controls_raw is not None:
        if type(controls_raw) is not str or len(controls_raw.encode("utf-8")) > 64 * 1024:
            raise PolicyError("bounded technical trial cancellation JSON required")
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise PolicyError("duplicate technical trial control key")
                result[key] = value
            return result
        try:
            controls = json.loads(controls_raw, object_pairs_hook=unique_object)
        except (TypeError, ValueError):
            raise PolicyError("valid technical trial cancellation JSON required") from None
        if type(app.clientId) is not int or app.clientId != args.client_id:
            raise PolicyError("technical cancellation requires the actual fixed PA connection identity")
        validate_technical_trial_controls(controls, mutation="CANCEL", client_id=app.clientId,
                                         conid=args.conid, target=target)
    collected_at = datetime.now(timezone.utc) if controls is not None else None
    open_orders = app.get_open_orders(args.timeout, all_clients=False)
    exact = [
        item
        for item in open_orders
        if eligible_cancel_record(
            item,
            order_id=args.order_id,
            order_ref=args.order_ref,
            conid=args.conid,
            client_id=args.client_id,
        )
    ]
    if len(exact) != 1:
        raise PolicyError(
            "cancel requires exactly one open order owned by this PA client, account, conId, and orderRef"
        )
    cancel_check = None
    if controls is not None:
        record = exact[0]
        if (record.get("sec_type") != "STK" or record.get("currency") != "USD"
                or record.get("primary_exchange") not in US_PRIMARY_EXCHANGES
                or record.get("order_type") != "LMT" or record.get("tif") != "DAY"
                or record.get("outside_rth") is not False):
            raise PolicyError("technical trial cancellation requires the exact regular-hours USD-stock LMT DAY order")
        resolved = app.qualify_exact_contract(make_stock_contract(
            conid=args.conid, symbol=record["symbol"], primary_exchange=record["primary_exchange"]), args.timeout)
        if (resolved.conId != args.conid or resolved.symbol != record["symbol"]
                or resolved.primaryExchange != record["primary_exchange"]):
            raise PolicyError("technical cancellation native listing identity changed")
        detail = dict(app.contract_details[0])
        status = app.get_status(args.timeout)
        def check_cancel():
            current = datetime.now(timezone.utc)
            result = validate_technical_trial_controls(controls, mutation="CANCEL",
                client_id=app.clientId, conid=args.conid, target=target, now=current)
            epoch = status.get("server_time_epoch")
            if (type(epoch) is not int or epoch <= 0 or not -5 <= current.timestamp() - epoch <= 60
                    or current < collected_at or (current - collected_at).total_seconds() > 60
                    or not is_liquid_session(epoch, detail)
                    or not is_liquid_session(int(current.timestamp()), detail)):
                raise PolicyError("technical cancellation evidence or native regular session expired")
            return result
        cancel_check = check_cancel()
    audit.append(
        "paper_order_cancel_attempt",
        {
            "boundary": {"host": HOST, "port": PORT, "account": ACCOUNT, "client_id": args.client_id},
            "order": exact[0],
            "gate": "dual_gate_passed",
            **({"technical_trial_check": cancel_check} if controls is not None else {}),
        },
    )
    app.order_events[args.order_id] = threading.Event()
    wire_attempted = False
    try:
        if controls is not None:
            cancel_check = check_cancel()
            audit.append("paper_order_cancel_technical_trial_pre_wire_check", cancel_check)
            cancel_check = check_cancel()  # AFTER fsync, immediately before cancelOrder
        wire_attempted = True
        app.cancelOrder(args.order_id)
        update = app.wait_for_order_callback(args.order_id, args.timeout)
        reconciled = _reconcile_order(
            order_id=args.order_id,
            order_ref=args.order_ref,
            conid=args.conid,
            client_id=args.client_id,
            requested_quantity=int(_decimal_value(exact[0]["quantity"])),
            open_orders=app.get_open_orders(args.timeout, all_clients=False),
            completed_orders=app.get_completed_orders(args.timeout),
            executions=app.get_executions(args.timeout),
            update=update,
        )
        reconciled["errors"] = app.errors
        if controls is not None:
            reconciled["technical_trial_cancel_check"] = cancel_check
        audit.append("paper_order_cancel_reconciliation", reconciled)
        return reconciled
    except Exception as exc:
        uncertain = _uncertain_reconciliation(
            order_id=args.order_id,
            order_ref=args.order_ref,
            conid=args.conid,
            client_id=args.client_id,
            error=exc,
        )
        if controls is not None:
            uncertain["broker_mutation_attempted"] = wire_attempted
            uncertain["failure_phase"] = "AFTER_CANCEL_ORDER_CALL" if wire_attempted else "PRE_WIRE_TRIAL_GATE"
        audit.append("paper_order_cancel_reconciliation", uncertain)
        raise PATWSError(
            f"post-cancel failure; cancellation is uncertain and must not be retried: {exc}"
        ) from exc


def _add_contract_args(parser: argparse.ArgumentParser, *, require_conid: bool = False) -> None:
    parser.add_argument("--conid", type=int, required=require_conid, default=0)
    parser.add_argument("--symbol", default="")
    parser.add_argument("--primary-exchange", default="")


def _add_order_args(
    parser: argparse.ArgumentParser, *, fixed_order_type: str | None = None
) -> None:
    _add_contract_args(parser, require_conid=True)
    if fixed_order_type is None:
        parser.add_argument("--order-type", required=True, choices=tuple(sorted(SUPPORTED_STOCK_ORDER_TYPES)))
    else:
        parser.set_defaults(order_type=fixed_order_type)
    parser.add_argument("--side", required=True, choices=("BUY", "SELL"))
    parser.add_argument("--quantity", required=True, type=int)
    parser.add_argument("--limit", required=fixed_order_type == "LMT")
    parser.add_argument("--stop")
    parser.add_argument("--trail-amount")
    parser.add_argument("--trail-percent")
    parser.add_argument("--trail-stop-price")
    parser.add_argument("--limit-offset")
    parser.add_argument("--reference-price")
    parser.add_argument("--trigger-method", type=int, choices=tuple(sorted(SUPPORTED_TRIGGER_METHODS)), default=0)
    parser.add_argument("--good-till-date")
    parser.add_argument("--tif", required=True, choices=tuple(sorted(SUPPORTED_TIFS)))
    parser.add_argument("--order-ref", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(client_id=DEFAULT_CLIENT_ID)
    parser.add_argument("--timeout", type=float, default=15.0)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("account")
    commands.add_parser("positions")
    commands.add_parser("portfolio")
    commands.add_parser("pnl")
    commands.add_parser("open-orders")
    commands.add_parser("completed-orders")
    commands.add_parser("executions")

    contract = commands.add_parser("contract")
    _add_contract_args(contract)
    quote = commands.add_parser("quote")
    _add_contract_args(quote)
    quote.add_argument("--market-data-type", type=int, choices=(1, 3), default=1)
    history = commands.add_parser("historical-daily")
    _add_contract_args(history, require_conid=True)
    history.add_argument("--duration", choices=("6 M", "1 Y", "2 Y"), default="2 Y")
    history.add_argument("--as-of")
    commands.add_parser("news-providers")
    news = commands.add_parser("historical-news")
    _add_contract_args(news, require_conid=True)
    news.add_argument("--providers", required=True)
    news.add_argument("--start-utc", required=True)
    news.add_argument("--end-utc", required=True)
    news.add_argument("--limit", type=int, default=300)
    news.add_argument("--max-pages", type=int, default=1)
    article = commands.add_parser("news-article")
    article.add_argument("--provider", required=True)
    article.add_argument("--article-id", required=True)
    live_news = commands.add_parser("live-news", help="bounded news-only capture for an exact U.S. stock")
    _add_contract_args(live_news, require_conid=True)
    live_news.add_argument("--providers", required=True)
    broadtape = commands.add_parser("broadtape-news", help="bounded capture for an exact provider NEWS contract")
    broadtape.add_argument("--provider", required=True)
    broadtape.add_argument("--news-symbol", required=True)
    for capture in (live_news, broadtape):
        capture.add_argument("--seconds", type=float, default=10.0)
        capture.add_argument("--max-headlines", type=int, default=100)

    preview = commands.add_parser("preview-stock-limit")
    _add_order_args(preview, fixed_order_type="LMT")
    preview.add_argument("--risk-policy", required=True)
    preview.add_argument("--paper-trial-profile", choices=(PAPER_DELAYED_TYPE3,))
    preview.add_argument("--portfolio-risk-policy-json")

    submit = commands.add_parser("submit-stock-limit")
    _add_order_args(submit, fixed_order_type="LMT")
    submit.add_argument("--risk-policy", required=True)
    submit.add_argument("--confirm", required=True)
    submit.add_argument("--require-live-preflight", action="store_true")
    submit.add_argument("--paper-trial-profile", choices=(PAPER_DELAYED_TYPE3,))
    submit.add_argument("--max-quote-age-seconds", type=float, default=30.0)
    submit.add_argument("--portfolio-risk-policy-json")

    generic_preview = commands.add_parser("preview-stock-order")
    _add_order_args(generic_preview)
    generic_preview.add_argument("--risk-policy", required=True)
    generic_preview.add_argument("--paper-trial-profile", choices=(PAPER_DELAYED_TYPE3,))
    generic_preview.add_argument("--portfolio-risk-policy-json")

    generic_submit = commands.add_parser("submit-stock-order")
    _add_order_args(generic_submit)
    generic_submit.add_argument("--risk-policy", required=True)
    generic_submit.add_argument("--confirm", required=True)
    generic_submit.add_argument("--require-live-preflight", action="store_true")
    generic_submit.add_argument("--paper-trial-profile", choices=(PAPER_DELAYED_TYPE3,))
    generic_submit.add_argument("--max-quote-age-seconds", type=float, default=30.0)
    generic_submit.add_argument("--portfolio-risk-policy-json")

    cancel = commands.add_parser("cancel-pa-order")
    cancel.add_argument("--order-id", type=int, required=True)
    cancel.add_argument("--order-ref", required=True)
    cancel.add_argument("--conid", type=int, required=True)
    cancel.add_argument("--confirm", required=True)
    cancel.add_argument("--technical-trial-controls-json",
                        help="optional exact root-reviewed trial restrictions; no ownership authority is inferred")
    return parser


def _preview(args: argparse.Namespace) -> dict[str, Any]:
    validate_boundary(client_id=args.client_id)
    requested = getattr(args, "paper_trial_profile", None)
    portfolio = None
    raw = getattr(args, "portfolio_risk_policy_json", None)
    if requested is not None or raw is not None:
        if type(raw) is not str or len(raw.encode("utf-8")) > 64 * 1024:
            raise PolicyError("bounded portfolio JSON required for paper trial preview")
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise PolicyError("ambiguous portfolio JSON in preview")
                result[key] = value
            return result
        try:
            portfolio = json.loads(raw, object_pairs_hook=unique_object)
        except (TypeError, ValueError) as exc:
            raise PolicyError("invalid paper trial preview portfolio JSON") from exc
        if (paper_trial_execution_profile(portfolio) == PAPER_DELAYED_TYPE3) != (requested == PAPER_DELAYED_TYPE3):
            raise PolicyError("preview requires matching CLI and portfolio profile opt-ins")
    policy = RiskPolicy.load(Path(args.risk_policy))
    spec = stock_order_spec_from_args(args)
    if requested == PAPER_DELAYED_TYPE3:
        if spec.order_type != "LMT" or spec.tif != "DAY":
            raise PolicyError("paper trial preview permits regular-hours LMT DAY only")
        validate_technical_trial_controls(
            portfolio["technical_paper_trial"], mutation="SUBMIT", client_id=args.client_id,
            conid=args.conid, side=spec.side, quantity=spec.quantity, mandate_id=portfolio.get("mandate_id"))
    # A SELL cannot be fully approved offline because the current long position
    # must come from TWS.  Preview reports that unresolved fact explicitly.
    policy.validate_order(
        side=args.side,
        quantity=args.quantity,
        order_type=spec.order_type,
        reference_price=spec.reference_price,
        tif=args.tif,
        current_position=Decimal(args.quantity) if args.side == "SELL" else None,
    )
    contract = make_stock_contract(
        conid=args.conid, symbol=args.symbol, primary_exchange=args.primary_exchange
    )
    order = order_from_spec(spec, client_id=args.client_id)
    result = {
        "preview_only": True,
        "broker_connection": False,
        "broker_mutation": False,
        "sell_position_check": "required_at_connected_preflight" if args.side == "SELL" else "not_applicable",
        "contract": contract_record(contract),
        "order": order_record(0, contract, order, "PREVIEW_ONLY"),
    }
    if requested == PAPER_DELAYED_TYPE3:
        result.update({"execution_profile": PAPER_DELAYED_TYPE3,
                       "requested_market_data_type": 3,
                       "connected_preflight": "REQUIRED_NOT_PERFORMED",
                       "native_pnl_guard_checked": False,
                       "execution_eligible": False})
    return result


def run_command(args: argparse.Namespace) -> Any:
    validate_boundary(client_id=args.client_id)
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.timeout > 60:
        raise PolicyError("timeout must be in (0, 60] seconds")
    if args.command in {"preview-stock-limit", "preview-stock-order"}:
        return _preview(args)
    if args.command in LOCKED_RESEARCH_COMMANDS:
        if (args.command in {"historical-daily", "historical-news", "live-news"}
                and (not args.symbol or args.primary_exchange not in US_PRIMARY_EXCHANGES
                     or type(args.conid) is not int or args.conid <= 0)):
            raise PolicyError("native history/news requires an exact approved stock conId and U.S. primary listing")
        if args.command == "historical-news":
            validate_news_providers(args.providers)
            start = parse_utc_timestamp(args.start_utc, "start-utc")
            end = parse_utc_timestamp(args.end_utc, "end-utc")
            if (start >= end or start.microsecond or end.microsecond
                    or not 1 <= args.limit <= 300 or not 1 <= args.max_pages <= 10):
                raise PolicyError("news requires ordered whole-second UTC bounds, limit 1..300 and max-pages 1..10")
        if args.command == "news-article":
            codes = validate_news_providers(args.provider)
            if len(codes) != 1 or not NEWS_ARTICLE_RE.fullmatch(args.article_id):
                raise PolicyError("article retrieval requires one provider and a valid exact article ID")
        if args.command in {"live-news", "broadtape-news"}:
            validate_news_capture(args.seconds, args.max_headlines)
            if args.command == "live-news":
                validate_news_providers(args.providers)
            else:
                make_news_contract(args.provider, args.news_symbol)
        # Read-only native history/news shares the writer lock and never steals it.
        # No audit mutation, submission environment or alternate client is used.
        with MutationLock() as lock:
            return _run_connected(args, None, lock)

    # Mutation gates are checked before any network connection.  A missing gate
    # therefore cannot even observe broker state through a mutation command.
    if args.command in {"submit-stock-limit", "submit-stock-order"}:
        require_mutation_gate("submit", args.confirm)
    elif args.command == "cancel-pa-order":
        require_mutation_gate("cancel", args.confirm)

    if args.command in {"submit-stock-limit", "submit-stock-order", "cancel-pa-order"}:
        with MutationLock() as lock:
            audit = AuditLog(CANONICAL_AUDIT_PATH)
            audit.verify()
            return _run_connected(args, audit, lock)
    return _run_connected(args, None, None)


def _run_connected(
    args: argparse.Namespace,
    audit: AuditLog | None,
    lock: MutationLock | None,
) -> Any:
    """Run one connection; the caller owns the mutation lock when required."""
    if args.command in LOCKED_RESEARCH_COMMANDS and (lock is None or not lock.owns(CANONICAL_MUTATION_LOCK_PATH)):
        raise GateError("native history/news requires the canonical PA serialization lock")
    app = TWSApp()
    try:
        app.connect_checked(args.client_id, args.timeout)
        if args.command == "status":
            return app.get_status(args.timeout)
        if args.command == "account":
            return app.get_account(args.timeout)
        if args.command == "positions":
            return app.get_positions(args.timeout)
        if args.command == "portfolio":
            return app.get_portfolio(args.timeout)
        if args.command == "pnl":
            return app.get_pnl(args.timeout)
        if args.command == "open-orders":
            return app.get_open_orders(args.timeout, all_clients=True)
        if args.command == "completed-orders":
            return app.get_completed_orders(args.timeout)
        if args.command == "executions":
            return app.get_execution_report(args.timeout)
        if args.command == "contract":
            return app.get_contracts(
                make_stock_contract(
                    conid=args.conid,
                    symbol=args.symbol,
                    primary_exchange=args.primary_exchange,
                ),
                args.timeout,
            )
        if args.command == "quote":
            candidate = make_stock_contract(
                conid=args.conid,
                symbol=args.symbol,
                primary_exchange=args.primary_exchange,
            )
            resolved = app.qualify_exact_contract(candidate, args.timeout)
            return app.get_quote(resolved, args.timeout, args.market_data_type)
        if args.command == "historical-daily":
            if lock is None or not lock.owns(CANONICAL_MUTATION_LOCK_PATH):
                raise GateError("daily history requires the canonical PA serialization lock")
            return app.get_historical_daily(make_stock_contract(
                conid=args.conid, symbol=args.symbol, primary_exchange=args.primary_exchange,
            ), args.timeout, args.duration, args.as_of)
        if args.command == "news-providers":
            return app.get_news_providers(args.timeout)
        if args.command == "historical-news":
            return app.get_historical_news(make_stock_contract(
                conid=args.conid, symbol=args.symbol, primary_exchange=args.primary_exchange,
            ), args.timeout, args.providers, args.start_utc, args.end_utc, args.limit, args.max_pages)
        if args.command == "news-article":
            return app.get_news_article(args.timeout, args.provider, args.article_id)
        if args.command == "live-news":
            return app.get_live_news(make_stock_contract(
                conid=args.conid, symbol=args.symbol, primary_exchange=args.primary_exchange,
            ), args.timeout, args.providers, args.seconds, args.max_headlines)
        if args.command == "broadtape-news":
            return app.get_live_news(make_news_contract(args.provider, args.news_symbol),
                                     args.timeout, args.provider, args.seconds, args.max_headlines,
                                     broadtape=True)
        if args.command in {"submit-stock-limit", "submit-stock-order"}:
            if audit is None:
                raise GateError("canonical mutation audit is unavailable")
            return submit_stock_order(app, args, audit, lock)
        if args.command == "cancel-pa-order":
            if audit is None:
                raise GateError("canonical mutation audit is unavailable")
            return cancel_pa_order(app, args, audit, lock)
        raise PATWSError(f"unknown command: {args.command}")
    finally:
        app.disconnect_clean()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run_command(build_parser().parse_args(argv))
    except (PATWSError, OSError) as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)}))
        return 2
    print(json.dumps({"ok": True, "result": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
