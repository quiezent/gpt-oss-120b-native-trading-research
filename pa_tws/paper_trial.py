"""Explicit delayed-data PAPER trial checks, not research qualification.

The canonical module passes its own API to these helpers so script and package
invocations use the same PolicyError, boundary and audit classes. No I/O occurs
on import. Native PnL means the actual reqPnL account-base metric, with its
configuration/initialization/economic-coverage limitations left explicit.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

PROFILE = "PAPER_DELAYED_TYPE3"
LIVE_PROFILE = "LIVE_TYPE1"
LOSS_EVENT = "paper_trial_native_pnl_observation"
MAX_PNL_ORIGINS = 256


@dataclass(frozen=True)
class NativePaperDailyLossObservation:
    account: str
    client_id: int
    session_key: str
    minimum_daily_pnl: str | None
    latest_daily_pnl: str | None
    callback_count: int
    observed_at_utc: str
    entry_halt_active: bool
    numeric_pnl_observed: bool
    source: str = "NATIVE_REQ_PNL_ACCOUNT_BASE"
    applied_reset_configuration: str = "UNKNOWN_RESEARCH_QUALIFICATION"
    costed_nav_reconstruction: str = "UNKNOWN_RESEARCH_QUALIFICATION"
    connector_selection_attributed: bool = False

    def to_record(self) -> dict:
        return asdict(self)


def execution_profile(pa, policy: Any) -> str:
    if not isinstance(policy, dict) or "execution_profile" not in policy:
        return LIVE_PROFILE
    if (policy["execution_profile"] != PROFILE
            or type(policy.get("technical_paper_trial")) is not dict):
        raise pa.PolicyError("PAPER_DELAYED_TYPE3 requires exact technical paper trial controls")
    return PROFILE


def _finite(pa, value):
    if isinstance(value, bool):
        raise pa.PolicyError("native numeric PnL/price cannot be boolean")
    parsed = pa._decimal_value(value)
    if abs(parsed) >= Decimal(str(pa.UNSET_DOUBLE)):
        raise pa.PolicyError("native numeric PnL/price is unset")
    return parsed


def quote_freshness(pa, quote, max_age_seconds, now=None):
    if (type(max_age_seconds) not in (int, float) or not math.isfinite(max_age_seconds)
            or not 0 < max_age_seconds <= 30):
        raise pa.PolicyError("paper delayed quote receipt bound must be in (0,30] seconds")
    current = datetime.now(timezone.utc) if now is None else now
    if (not isinstance(current, datetime) or current.tzinfo is None
            or current.utcoffset().total_seconds() != 0):
        raise pa.PolicyError("paper quote clock must be explicit UTC")
    try:
        mode = quote["callback_market_data_type"]
        if (type(quote["requested_market_data_type"]) is not int
                or quote["requested_market_data_type"] != 3 or type(mode) is not int
                or mode not in {1, 3} or quote["quality"] != "BBO"
                or quote["snapshot_complete"] is not True):
            raise pa.PolicyError("completed requested-Type3 actual-Type1/3 BBO required")
        q = quote["quote"]
        bid, ask = _finite(pa, q["bid"]), _finite(pa, q["ask"])
        expected_ticks = (66, 67) if mode == 3 else (1, 2)
        if (bid <= 0 or ask <= 0 or bid > ask
                or any(type(q[key]) is not int for key in ("bid_tick_type", "ask_tick_type"))
                or (q["bid_tick_type"], q["ask_tick_type"]) != expected_ticks):
            raise pa.PolicyError("paper BBO prices or original tick classification invalid")
        start = pa.parse_utc_timestamp(quote["snapshot_request_started_at_utc"], "quote request start")
        end = pa.parse_utc_timestamp(quote["snapshot_completed_at_utc"], "quote snapshot end")
        receipts = [pa.parse_utc_timestamp(q[key], key)
                    for key in ("bid_received_at_utc", "ask_received_at_utc")]
        if not start <= min(receipts) <= max(receipts) <= end <= current:
            raise pa.PolicyError("paper snapshot request/receipt/end clocks are unordered")
        ages = [(current - value).total_seconds() for value in [start, end, *receipts]]
        mono_start = quote["snapshot_request_started_monotonic_ns"]
        mono_end = quote["snapshot_completed_monotonic_ns"]
        mono_now = time.monotonic_ns()
        if (type(mono_start) is not int or type(mono_end) is not int
                or not 0 < mono_start <= mono_end <= mono_now
                or (mono_now - mono_start) / 1e9 > max_age_seconds):
            raise pa.PolicyError("paper snapshot monotonic request lease expired")
        if any(not math.isfinite(age) or not 0 <= age <= max_age_seconds for age in ages):
            raise pa.PolicyError("paper snapshot request/receipt lease expired")
    except (KeyError, ValueError, TypeError, OverflowError) as exc:
        raise pa.PolicyError("paper quote lacks valid completed native snapshot provenance") from exc
    return {"execution_profile": PROFILE, "requested_market_data_type": 3,
            "actual_market_data_type": mode, "receipt_age_seconds": max(ages),
            "market_event_age_seconds": "UNKNOWN_NOT_A_LIVE_FRESHNESS_CLAIM",
            "reference_kind": "PAPER_DELAYED_QUOTE_REFERENCE" if mode == 3 else "NATIVE_LIVE_BBO"}


def stock_preflight(pa, app, contract, timeout, max_age_seconds):
    if type(app.clientId) is not int or app.clientId != 9901:
        raise pa.PolicyError("native client identity must be exact integer9901")
    pa.validate_boundary(client_id=app.clientId)
    if pa.ACCOUNT not in app.managed_accounts:
        raise pa.PolicyError("native managedAccounts does not expose the exact paper account")
    if contract.primaryExchange not in pa.US_PRIMARY_EXCHANGES:
        raise pa.PolicyError("paper delayed profile requires an approved U.S. primary listing")
    details = app.get_contracts(contract, timeout)
    if (len(details) != 1 or details[0].get("conid") != contract.conId
            or details[0].get("primary_exchange") != contract.primaryExchange
            or details[0].get("sec_type") != "STK" or details[0].get("currency") != "USD"):
        raise pa.PolicyError("paper delayed exact USD stock identity unresolved")
    quote = app.get_quote(contract, timeout, 3)
    status = app.get_status(timeout)
    if (status.get("account") != pa.ACCOUNT or type(status.get("client_id")) is not int
            or status.get("client_id") != 9901
            or not pa.is_liquid_session(int(status["server_time_epoch"]), details[0])):
        raise pa.PolicyError("native paper identity/RTH is unverified")
    freshness = quote_freshness(pa, quote, max_age_seconds)
    return {"contract_details": details[0], "quote": quote, "broker_status": status,
            "execution_profile": PROFILE, "quote_freshness": freshness}


def session_key(pa, now=None):
    current = datetime.now(timezone.utc) if now is None else now
    return f"{pa.ACCOUNT}:America/New_York:{current.astimezone(ZoneInfo('America/New_York')).date().isoformat()}"


def arm_native_pnl(pa, app):
    key = session_key(pa)
    with app._paper_pnl_lock:
        if app._paper_pnl_session_key != key:
            app._paper_pnl_minimum = None
            app._paper_pnl_origins = []
            app._paper_pnl_callback_count = 0
            app._paper_pnl_overflow = False
        app._paper_pnl_session_key = key
        app._paper_pnl_window_start_count = app._paper_pnl_callback_count


def record_native_pnl(pa, app, req_id, daily, unrealized, realized):
    with app._paper_pnl_lock:
        if app._paper_pnl_session_key is None or req_id != 9105:
            return
        normalized = pa._optional_order_number(daily)
        if isinstance(daily, bool):
            normalized = None
        app._paper_pnl_callback_count += 1
        row = {"ordinal": app._paper_pnl_callback_count, "request_id": req_id,
               "requested_account": pa.ACCOUNT, "account_is_request_binding_not_callback_field": True,
               "daily_pnl": normalized, "sdk_daily_pnl_repr": repr(daily),
               "sdk_daily_pnl_hex": daily.hex() if type(daily) is float else None,
               "unrealized_pnl_repr": repr(unrealized), "realized_pnl_repr": repr(realized),
               "received_at_utc": pa.utc_now(), "receipt_monotonic_ns": time.monotonic_ns()}
        if normalized is not None:
            parsed = _finite(pa, normalized)
            app._paper_pnl_minimum = parsed if app._paper_pnl_minimum is None else min(app._paper_pnl_minimum, parsed)
        if len(app._paper_pnl_origins) < MAX_PNL_ORIGINS:
            app._paper_pnl_origins.append(row)
        else:
            app._paper_pnl_overflow = True


def observe_native_pnl(pa, app, audit, *, allow_missing=False, loss_limit_usd=Decimal(100), finalized_session=False):
    from copy import deepcopy

    if audit is None or audit.path.resolve() != pa.CANONICAL_AUDIT_PATH.resolve():
        raise pa.PolicyError("canonical audit required for the paper session loss guard")
    threshold = _finite(pa, loss_limit_usd)
    if not 0 < threshold <= Decimal(100):
        raise pa.PolicyError("paper session daily loss cap must be positive and at most100USD")
    key = app._paper_pnl_session_key if finalized_session else session_key(pa)
    # Verification can take time while the reader still ingests callbacks.
    # Snapshot the current minimum only after it, never before it.
    previous_rows = audit.verify()
    with app._paper_pnl_lock:
        if app._paper_pnl_session_key != key:
            raise pa.PolicyError("native paper PnL session scope is unarmed or changed")
        origins = deepcopy(app._paper_pnl_origins)
        minimum = app._paper_pnl_minimum
        count = app._paper_pnl_callback_count
        window_start = app._paper_pnl_window_start_count
        overflow = app._paper_pnl_overflow
    sticky = False
    for row in previous_rows:
        if row.get("event") == LOSS_EVENT:
            previous = row["details"]
            if previous.get("session_key") == key and previous.get("account") == pa.ACCOUNT:
                prior = previous.get("minimum_daily_pnl")
                sticky |= previous.get("entry_halt_active") is True
                sticky |= prior is not None and _finite(pa, prior) <= -threshold
    sticky |= minimum is not None and minimum <= -threshold
    sticky |= getattr(app, "_paper_pnl_final_reader_unsettled", False)
    latest = origins[-1].get("daily_pnl") if origins else None
    numeric = latest is not None and minimum is not None and count > window_start and not overflow
    observation = NativePaperDailyLossObservation(
        pa.ACCOUNT, 9901, key, str(minimum) if minimum is not None else None,
        str(latest) if latest is not None else None, count, pa.utc_now(), sticky, numeric)
    # Always retain observed breaches before refusal, including bounded-prefix
    # overflow; never manufacture an all-callbacks or initialization claim.
    audit.append(LOSS_EVENT, {**observation.to_record(), "execution_profile": PROFILE,
                             "callback_origins": origins, "callback_retention_overflow": overflow,
                             "all_callback_origins_retained": not overflow,
                             "owned_reader_unsettled_at_finalization": getattr(app, "_paper_pnl_final_reader_unsettled", False),
                             "daily_loss_threshold_usd": str(threshold)})
    if not numeric and not allow_missing:
        raise pa.PolicyError("native numeric paper dailyPnL unavailable; new entries paused")
    return observation


def persist_final_native_pnl(pa, app):
    """Retain the final armed callback minimum after the owned reader stops.

    No alternative audit path; read-only observers without a mutation audit do
    nothing. Preserve the original armed session even across local midnight.
    """
    audit = getattr(app, "_paper_trial_audit", None)
    if audit is None or app._paper_pnl_session_key is None:
        return
    observe_native_pnl(pa, app, audit, allow_missing=True,
                       loss_limit_usd=getattr(app, "_paper_trial_loss_limit", Decimal(100)), finalized_session=True)
