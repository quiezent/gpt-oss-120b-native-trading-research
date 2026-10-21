"""Bounded daily stock history through the canonical native TWS read surface.

Daily bar dates are labels, not precise close/publication timestamps. TRADES
prices are split-adjusted, not dividend-adjusted. This capture is a retrieval
vintage; it cannot establish the adjustments or publications available at a
past decision. The conservative cutoff excludes the current exchange-local day.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .tws_broker import US_PRIMARY_EXCHANGES, PaperTWSBroker
from .validation import PAPER_BOUNDARY, contract_key, money, timestamp

DURATIONS = frozenset({"6 M", "1 Y", "2 Y"})
MAX_BARS = 1000
TRADES_SOURCE = "https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-bar-what-to-show/trades"
VOLUME_SOURCE = "https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-data-limitations/historical-volume-scaling"


class HistoricalDataError(RuntimeError):
    """A controlled native history validation failure."""


def normalize_daily_history(raw: Any, contract: dict[str, Any]) -> dict[str, Any]:
    """Verify a completed native request without inventing bar availability."""
    try:
        if (not isinstance(raw, dict) or raw.get("source") != "native_tws"
                or raw.get("mode") != "paper" or raw.get("boundary") != PAPER_BOUNDARY
                or raw.get("request_completed") is not True):
            raise HistoricalDataError("daily history did not prove native request completion and boundary")
        detail = raw["contract"]
        if (contract_key(detail) != contract_key(contract) or detail.get("sec_type") != "STK"
                or detail.get("currency") != "USD" or detail["primary_exchange"] not in US_PRIMARY_EXCHANGES):
            raise HistoricalDataError("daily history contract or U.S. listing mismatch")
        request = raw["request"]
        if (request.get("duration") not in DURATIONS or request.get("bar_size") != "1 day"
                or request.get("what_to_show") != "TRADES" or request.get("use_rth") is not True
                or request.get("format_date") != 1 or request.get("keep_up_to_date") is not False):
            raise HistoricalDataError("unsupported native daily history request")
        broker_epoch = raw["server_time_epoch"]
        if (type(broker_epoch) is not int or broker_epoch <= 0
                or raw.get("broker_checkpoint_scope") != "before_request"):
            raise HistoricalDataError("native history lacks broker time")
        as_of = timestamp(raw["as_of_utc"]).astimezone(timezone.utc)
        if as_of.timestamp() > broker_epoch:
            raise HistoricalDataError("history cutoff is ahead of the broker checkpoint")
        retrieved = timestamp(raw["retrieved_at_utc"]).astimezone(timezone.utc)
        started = timestamp(raw["request_started_at_utc"]).astimezone(timezone.utc)
        completed = timestamp(raw["request_completed_at_utc"]).astimezone(timezone.utc)
        if not started <= completed <= retrieved:
            raise HistoricalDataError("history receipt moved backwards in time")
        zone = ZoneInfo(detail["time_zone_id"])
        local_day = as_of.astimezone(zone).date()
        midnight = datetime.combine(local_day, datetime.min.time(), tzinfo=zone).astimezone(timezone.utc)
        if request.get("end_date_time_utc") != midnight.strftime("%Y%m%d-%H:%M:%S"):
            raise HistoricalDataError("history request did not apply the conservative local-day cutoff")
        rows = raw["bars"]
        if not isinstance(rows, list) or not rows or len(rows) > MAX_BARS:
            raise HistoricalDataError("native daily history is empty or exceeds its bound")
        bars, seen = [], set()
        excluded = raw.get("excluded_current_or_future_date_count", 0)
        if type(excluded) is not int or excluded < 0:
            raise HistoricalDataError("invalid native excluded-bar count")
        for row in rows:
            native_date = row["native_date"]
            if not isinstance(native_date, str) or len(native_date) != 8 or not native_date.isdigit():
                raise HistoricalDataError("native daily bar date must be YYYYMMDD")
            day = date.fromisoformat(native_date)
            if day in seen:
                raise HistoricalDataError("native history contains duplicate daily bars")
            seen.add(day)
            if day >= local_day:
                excluded += 1
                continue
            prices = {field: money(row[field]) for field in ("open", "high", "low", "close")}
            if (min(prices.values()) <= 0 or prices["low"] > min(prices["open"], prices["close"])
                    or prices["high"] < max(prices["open"], prices["close"]) or prices["high"] < prices["low"]):
                raise HistoricalDataError("native daily OHLC is nonpositive or incoherent")
            volume = money(row["volume"])
            if volume < 0:
                raise HistoricalDataError("native daily volume is negative")
            received = timestamp(row["received_at_utc"]).astimezone(timezone.utc)
            if not started <= received <= completed:
                raise HistoricalDataError("daily bar callback timestamp is outside the native request")
            bars.append({"session": day.isoformat(), "native_date": native_date,
                         **{key: str(value) for key, value in prices.items()}, "volume": str(volume),
                         "received_at_utc": received.isoformat(), "available_at": None,
                         "bar_count": row.get("bar_count"), "average": row.get("average")})
        if not bars:
            raise HistoricalDataError("native history has no prior-day bars after the completion cutoff")
        bars.sort(key=lambda bar: bar["session"])
        value = {
            "source": "native_tws", "mode": "paper", "boundary": dict(PAPER_BOUNDARY),
            "contract": deepcopy(detail), "symbol": detail["symbol"], "currency": "USD",
            "request": deepcopy(request), "request_completed": True, "bars": bars,
            "as_of_utc": as_of.isoformat(), "retrieved_at_utc": retrieved.isoformat(),
            "request_started_at_utc": started.isoformat(), "server_time_epoch": broker_epoch,
            "request_completed_at_utc": completed.isoformat(), "broker_checkpoint_scope": "before_request",
            "excluded_current_or_future_date_count": excluded,
            "provenance": {
                "source_kind": "native_tws", "retrieved_at": retrieved.isoformat(),
                "history_vintage": "retrieval_time", "publication_times_verified": False,
                "correction_status": "unknown", "price_basis": "split_adjusted",
                "dividend_adjusted": False, "corporate_action_coverage": "unknown",
                "corporate_action_sessions": [], "calendar_status": "unknown",
                "daily_date_scope": "broker date label; precise close time unknown",
                "completion_convention": "request ends at exchange-local calendar midnight; current and future date labels excluded",
                "volume_units": "native_configured_units_unverified", "volume_conversion_applied": False,
                "price_basis_source": TRADES_SOURCE, "volume_units_source": VOLUME_SOURCE,
                "historical_point_in_time_eligible": False,
            },
        }
        value["history_id"] = "native-history:" + hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return value
    except HistoricalDataError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError, ZoneInfoNotFoundError):
        raise HistoricalDataError("native daily history failed validation") from None


class NativeHistoricalData:
    """Serial, read-only API; it reuses a PaperTWSBroker controlled command runner."""

    def __init__(self, broker: PaperTWSBroker):
        if not isinstance(broker, PaperTWSBroker):
            raise HistoricalDataError("history requires the canonical native broker adapter")
        self.broker = broker

    def daily(self, contract: dict[str, Any], *, duration: str = "2 Y",
              as_of: str | None = None) -> dict[str, Any]:
        if duration not in DURATIONS:
            raise HistoricalDataError("duration must be 6 M, 1 Y or 2 Y")
        arguments = self.broker._contract_arguments(contract) + ["--duration", duration]
        if as_of is not None:
            arguments.extend(["--as-of", timestamp(as_of).astimezone(timezone.utc).isoformat()])
        try:
            raw = self.broker._invoke("historical-daily", arguments)
            return normalize_daily_history(raw, contract)
        except HistoricalDataError:
            raise
        except Exception:  # noqa: BLE001 - never expose process environments, raw error bodies or credentials
            raise HistoricalDataError("native daily history unavailable; no retry") from None


def default_reader() -> NativeHistoricalData:
    return NativeHistoricalData(PaperTWSBroker(Path(__file__).parent / "paper_risk_policy.example.json"))
