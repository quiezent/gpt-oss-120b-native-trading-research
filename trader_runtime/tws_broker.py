"""Fixed paper-account native TWS adapter; models never own the order surface.

Every command delegates to pa_tws with argv, a scoped environment and its fixed
canonical audit/lock. Account positions and all-open-order snapshots are current
account-wide evidence. Executions remain the evidence visible to client 9901;
neither a current-day completed-order query nor an ownership assertion makes
them a complete account history. Another account writer can change state between
snapshots. The canonical mutation lock serializes PA-controlled submissions and
their connected preflights, but does not lock unrelated broker clients.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import threading
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .validation import PAPER_BOUNDARY

PA_SCRIPT = Path(__file__).resolve().parents[1] / "pa_tws" / "pa_tws.py"
PA_PYTHON = Path(os.environ.get("PA_PYTHON", __import__("sys").executable))
CANONICAL_AUDIT = PA_SCRIPT.parent / "audit" / "pa_tws_audit.jsonl"
US_PRIMARY_EXCHANGES = frozenset({"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "IEX"})
_INTENT_FIELDS = {
    "intent_id", "account", "conid", "symbol", "primary_exchange", "side",
    "quantity", "limit_price", "order_type", "tif", "outside_rth",
}
_COMMAND_LOCK = threading.RLock()


class TWSBrokerError(RuntimeError):
    """Broker evidence or a controlled preflight could not be verified."""


def _execution_observation(value: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Accept legacy rows or the additive fee envelope without losing evidence.

    This parses observation structure only. Missing fees remain unknown; it
    computes no total, return, FX value, trading decision or owned inventory.
    Every original envelope field, unmatched report and query fact survives.
    """
    if type(value) is list:
        rows = value
        representation = "LEGACY_EXECUTION_LIST"
        fees = "UNKNOWN_LEGACY_RESPONSE"
    elif type(value) is dict and type(value.get("executions")) is list:
        if (type(value.get("commission_reports")) is not list
                or type(value.get("query")) is not dict):
            raise TWSBrokerError("execution fee envelope lacks report/query evidence")
        rows = value["executions"]
        representation = "EXECUTION_FEE_EVIDENCE_ENVELOPE"
        fees = "PER_EXECUTION_OBSERVED_OR_UNKNOWN; no account total inferred"
    else:
        raise TWSBrokerError("native execution observation must be rows or a fee evidence envelope")
    if any(type(row) is not dict for row in rows):
        raise TWSBrokerError("native execution rows must be objects")
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise TWSBrokerError("native execution observation must be finite JSON") from None
    return deepcopy(rows), {"representation": representation, "fee_evidence_scope": fees,
                            "original_result": deepcopy(value)}


def _money(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise TWSBrokerError("invalid native monetary value")
    try:
        amount = Decimal(str(value))
    except (ValueError, InvalidOperation):
        raise TWSBrokerError("invalid native monetary value") from None
    if not amount.is_finite():
        raise TWSBrokerError("non-finite native monetary value")
    return amount


def _utc(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError("missing timezone")
        return result.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError):
        raise TWSBrokerError("invalid native evidence timestamp") from None


def _session_open(epoch: int, detail: dict[str, Any]) -> bool:
    try:
        zone = ZoneInfo(detail["time_zone_id"])
        now = datetime.fromtimestamp(epoch, timezone.utc).astimezone(zone)
        for segment in detail["liquid_hours"].split(";"):
            if ":CLOSED" in segment:
                continue
            date_prefix, intervals = segment.split(":", 1)
            for interval in intervals.split(","):
                start, end = interval.split("-", 1)
                start = start if ":" in start else date_prefix + ":" + start
                end = end if ":" in end else date_prefix + ":" + end
                begin = datetime.strptime(start, "%Y%m%d:%H%M").replace(tzinfo=zone)
                finish = datetime.strptime(end, "%Y%m%d:%H%M").replace(tzinfo=zone)
                if begin <= now < finish:
                    return True
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError, ZoneInfoNotFoundError):
        return False
    return False


class PaperTWSBroker:
    """Observe and submit exact whole-share U.S. stock LMT DAY paper intents.

    runner follows subprocess.run's interface; clock returns an aware datetime.
    Both injection points are for deterministic offline tests. Neither changes
    the fixed paper boundary or canonical native audit/lock paths.
    """

    def __init__(
        self,
        risk_policy_path: str | Path,
        timeout_seconds: float = 15.0,
        max_quote_age_seconds: float = 30.0,
        exclusive_order_owner: bool = False,
        *,
        runner: Callable[..., Any] | None = None,
        clock: Callable[[], datetime] | None = None,
        max_account_age_seconds: float = 60.0,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
            raise TWSBrokerError("native timeout must be in (0, 60] seconds")
        if isinstance(max_quote_age_seconds, bool) or not math.isfinite(max_quote_age_seconds) or not 0 < max_quote_age_seconds <= 300:
            raise TWSBrokerError("quote age limit must be in (0, 300] seconds")
        self.risk_policy_path = Path(risk_policy_path).resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.max_quote_age_seconds = float(max_quote_age_seconds)
        if isinstance(max_account_age_seconds, bool) or not math.isfinite(max_account_age_seconds) or not 0 < max_account_age_seconds <= 300:
            raise TWSBrokerError("account evidence age limit must be in (0, 300] seconds")
        self.max_account_age_seconds = float(max_account_age_seconds)
        # Optional operator information only. It never expands evidence coverage.
        self.exclusive_order_owner = exclusive_order_owner
        self._runner = runner or subprocess.run
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._attempted: set[str] = set()
        self._receipts: dict[str, dict[str, Any]] = {}
        self._intents: dict[str, dict[str, Any]] = {}

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise TWSBrokerError("clock must return an aware datetime")
        return value.astimezone(timezone.utc)

    def _invoke(self, command: str, arguments: list[str] | None = None, *, mutation: bool = False) -> Any:
        if mutation != (command == "submit-stock-limit"):
            raise TWSBrokerError("unsupported native command authority")
        env = dict(os.environ)
        env.pop("IBKR_PA_ALLOW_PAPER_ORDER", None)
        for name in list(env):
            if name.startswith("TINKER_") or name in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}:
                env.pop(name)
        if mutation:
            env["IBKR_PA_ALLOW_PAPER_ORDER"] = "YES"
        if not PA_PYTHON.is_file():
            raise TWSBrokerError("the canonical project broker interpreter is unavailable")
        argv = [str(PA_PYTHON), str(PA_SCRIPT), "--timeout", str(self.timeout_seconds), command]
        argv.extend(arguments or [])
        try:
            with _COMMAND_LOCK:
                completed = self._runner(
                    argv, env=env, capture_output=True, text=True, encoding="utf-8",
                    timeout=self.timeout_seconds * 20 + 10, shell=False, check=False,
                )
            if len(completed.stdout.encode("utf-8")) > 8 * 1024 * 1024:
                raise TWSBrokerError("native broker response is too large")
            envelope = json.loads(completed.stdout)
            if completed.returncode != 0 or not isinstance(envelope, dict) or envelope.get("ok") is not True:
                raise TWSBrokerError("native TWS command failed; inspect the canonical broker audit")
            return envelope["result"]
        except TWSBrokerError:
            raise
        except Exception:  # noqa: BLE001 - redact process/environment details at the broker boundary
            # Never expose inherited environment, stderr, or raw process errors.
            raise TWSBrokerError("native TWS command failed or timed out") from None

    @staticmethod
    def _coverage(complete: bool, mutation_allowed: bool = False) -> dict[str, Any]:
        return {
            "risk_snapshot_complete": complete, "positions_scope": "account_wide",
            "open_orders_scope": "account_wide_snapshot", "executions_scope": "client_9901",
            "account_wide_executions_complete": False,
            "completed_orders_scope": "broker_current_day_including_TWS",
            "mutation_allowed": mutation_allowed,
        }

    def status(self) -> dict[str, Any]:
        raw = self._invoke("status")
        if (
            not isinstance(raw, dict) or raw.get("connected") is not True
            or raw.get("host") != PAPER_BOUNDARY["host"]
            or raw.get("port") != PAPER_BOUNDARY["port"]
            or raw.get("account") != PAPER_BOUNDARY["account"]
            or raw.get("client_id") != PAPER_BOUNDARY["client_id"]
            or PAPER_BOUNDARY["account"] not in raw.get("managed_accounts", [])
            or type(raw.get("server_time_epoch")) is not int or raw["server_time_epoch"] <= 0
        ):
            raise TWSBrokerError("native broker did not prove the fixed paper boundary")
        return {
            **raw, "mode": "paper", "source": "native_tws", "boundary": dict(PAPER_BOUNDARY),
            "observed_at_utc": self._now().isoformat(), "coverage": self._coverage(False),
        }

    @staticmethod
    def _contract_arguments(contract: dict[str, Any]) -> list[str]:
        if (
            type(contract.get("conid")) is not int or contract["conid"] <= 0
            or not isinstance(contract.get("symbol"), str) or not contract["symbol"]
            or contract.get("primary_exchange") not in US_PRIMARY_EXCHANGES
        ):
            raise TWSBrokerError("an exact approved U.S. stock identity is required")
        return ["--conid", str(contract["conid"]), "--symbol", contract["symbol"],
                "--primary-exchange", contract["primary_exchange"]]

    def _verified_quote(self, contract: dict[str, Any], detail: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
        identity = (contract["conid"], contract["symbol"], contract["primary_exchange"])
        for actual in (detail, raw.get("contract", {})):
            if (
                (actual.get("conid"), actual.get("symbol"), actual.get("primary_exchange")) != identity
                or actual.get("sec_type") != "STK" or actual.get("currency") != "USD"
            ):
                raise TWSBrokerError("native stock contract identity or listing mismatch")
        quote = raw["quote"]
        if raw.get("callback_market_data_type") != 1 or raw.get("quality") != "BBO":
            raise TWSBrokerError("a callback-proven live BBO is required")
        bid, ask = _money(quote["bid"]), _money(quote["ask"])
        if not 0 < bid <= ask:
            raise TWSBrokerError("native quote has an invalid BBO")
        try:
            source = datetime.fromtimestamp(float(quote["last_timestamp_epoch"]), timezone.utc)
        except (KeyError, ValueError, TypeError, OverflowError, OSError):
            raise TWSBrokerError("native quote has no valid last-trade source timestamp") from None
        received = _utc(quote["received_at_utc"])
        bbo_receipts = [_utc(quote[name + "_received_at_utc"]) for name in ("bid", "ask")]
        now = self._now()
        if not -5 <= (now - source).total_seconds() <= self.max_quote_age_seconds:
            raise TWSBrokerError("native quote source timestamp is stale or from the future")
        if not -5 <= (now - received).total_seconds() <= self.max_quote_age_seconds:
            raise TWSBrokerError("native quote receipt is stale or from the future")
        if any(not -5 <= (now - value).total_seconds() <= self.max_quote_age_seconds for value in bbo_receipts):
            raise TWSBrokerError("native BBO bid or ask receipt is stale or from the future")
        return {
            "conid": identity[0], "symbol": identity[1], "primary_exchange": identity[2],
            "us_listing_verified": True, "market_data_type": "LIVE", "bid": str(bid), "ask": str(ask),
            "as_of": source.isoformat(), "source_timestamp_scope": "last_trade_exchange_timestamp",
            "received_at_utc": received.isoformat(), "receipt_timestamp_scope": "local_callback_receipt",
            "bid_received_at_utc": bbo_receipts[0].isoformat(), "ask_received_at_utc": bbo_receipts[1].isoformat(),
        }

    def snapshot(self, contracts: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(contracts, list):
            raise TWSBrokerError("approved contracts must be a list")
        collection_started = self._now()
        self.status()
        raw = {command: self._invoke(command) for command in
               ("account", "positions", "open-orders", "completed-orders", "executions")}
        reasons: list[str] = []
        for command in ("portfolio", "pnl"):
            try:
                raw[command] = self._invoke(command)
            except TWSBrokerError:
                raw[command] = None
                reasons.append("native " + command + " evidence unavailable")
        values = raw["account"]
        def account_value(tag: str) -> str | None:
            candidates = [row for row in values if row.get("account") == PAPER_BOUNDARY["account"]
                          and row.get("tag") == tag and row.get("currency") == "USD"]
            return str(_money(candidates[-1]["value"])) if candidates else None
        account = {"account": PAPER_BOUNDARY["account"], "cash_usd": account_value("TotalCashValue"),
                   "nav_usd": account_value("NetLiquidation"), "daily_pnl_usd": None}
        pnl = raw["pnl"]
        if isinstance(pnl, dict) and pnl.get("account") == PAPER_BOUNDARY["account"] and account["nav_usd"] is not None:
            try:
                account["daily_pnl_usd"] = str(_money(pnl["daily_pnl"]))
            except (TWSBrokerError, KeyError):
                reasons.append("native account daily PnL is unset")
        else:
            reasons.append("USD base-account daily PnL is unverified")
        if account["cash_usd"] is None or account["nav_usd"] is None:
            reasons.append("native USD account balances are unverified")
        marks = {row["conid"]: row for row in (raw["portfolio"] or []) if row.get("account") == PAPER_BOUNDARY["account"]}
        positions = []
        for row in raw["positions"]:
            if row.get("account") != PAPER_BOUNDARY["account"]:
                raise TWSBrokerError("position evidence is outside the paper account")
            amount = _money(row["position"])
            if not amount:
                continue
            mark = marks.get(row["conid"], {})
            price = mark.get("market_price") if row.get("currency") == "USD" and row.get("sec_type") == "STK" else None
            if price is None or _money(price) <= 0 or _money(mark.get("position")) != amount:
                price = None
                reasons.append("native USD position mark is unverified")
            positions.append({**row, "quantity": int(amount) if amount == amount.to_integral_value() else str(amount),
                              "mark_usd": None if price is None else str(_money(price)),
                              "mark_timestamp_scope": "broker_portfolio_mark"})
        open_orders = []
        terminal = {"cancelled", "apicancelled", "filled", "inactive", "rejected"}
        for row in raw["open-orders"]:
            if row.get("account") != PAPER_BOUNDARY["account"]:
                raise TWSBrokerError("open-order evidence is outside the paper account")
            if str(row.get("status", "")).lower() not in terminal:
                open_orders.append(dict(row))
                try:
                    remaining = _money(row["remaining_quantity"])
                    if (row.get("sec_type") != "STK" or row.get("currency") != "USD"
                            or row.get("side") not in {"BUY", "SELL"} or remaining < 0
                            or remaining != remaining.to_integral_value() or row.get("order_type") != "LMT"
                            or _money(row.get("limit_price")) <= 0):
                        reasons.append("native open commitment cannot be valued as whole-share USD stock LMT")
                except (TWSBrokerError, KeyError):
                    reasons.append("native open commitment valuation is unverified")
        details, quotes = [], []
        for contract in contracts:
            arguments = self._contract_arguments(contract)
            resolved = self._invoke("contract", arguments)
            if not isinstance(resolved, list) or len(resolved) != 1:
                raise TWSBrokerError("native contract does not resolve uniquely")
            detail = resolved[0]
            if any(detail.get(field) != contract[field] for field in ("conid", "symbol", "primary_exchange")):
                raise TWSBrokerError("native primary listing does not match the approved contract")
            if detail.get("sec_type") != "STK" or detail.get("currency") != "USD":
                raise TWSBrokerError("native approved contract is not a USD stock")
            details.append(detail)
            try:
                quote = self._invoke("quote", arguments + ["--market-data-type", "1"])
                quotes.append(self._verified_quote(contract, detail, quote))
            except (TWSBrokerError, KeyError, TypeError):
                reasons.append("fresh native live BBO unavailable for approved contract")
        status = self.status()
        epoch = status["server_time_epoch"]
        decision_at = status["observed_at_utc"]
        final_receipt = _utc(decision_at)
        if (final_receipt - collection_started).total_seconds() > self.max_account_age_seconds:
            reasons.append("native account state expired during sequential collection")
        for row in [*(raw["portfolio"] or []), *([pnl] if isinstance(pnl, dict) else [])]:
            try:
                age = (final_receipt - _utc(row["received_at_utc"])).total_seconds()
                if not -5 <= age <= self.max_account_age_seconds:
                    reasons.append("native portfolio or PnL callback evidence is stale or from the future")
            except (TWSBrokerError, KeyError):
                reasons.append("native portfolio or PnL callback freshness is unverified")
        market_open = bool(details) and all(_session_open(epoch, detail) for detail in details)
        if not market_open:
            reasons.append("broker contract schedule does not prove an open regular session")
        # Recheck earlier quotes against the final broker checkpoint after all reads.
        usable = []
        for quote in quotes:
            age = epoch - _utc(quote["as_of"]).timestamp()
            bbo_ages = [(final_receipt - _utc(quote[name + "_received_at_utc"])).total_seconds() for name in ("bid", "ask")]
            if -5 <= age <= self.max_quote_age_seconds and all(-5 <= value <= self.max_quote_age_seconds for value in bbo_ages):
                usable.append(quote)
            else:
                reasons.append("quote expired while collecting account state")
        complete = not reasons
        identifier = hashlib.sha256(json.dumps({"epoch": epoch, "raw": raw, "quotes": usable}, sort_keys=True).encode()).hexdigest()
        return {
            "snapshot_id": "native:" + identifier, "mode": "paper", "source": "native_tws",
            "boundary": dict(PAPER_BOUNDARY), "observed_at_utc": status["observed_at_utc"],
            "decision_at": decision_at, "server_time_epoch": epoch, "account": account,
            "positions": positions, "open_orders": open_orders, "quotes": usable, "evidence": [],
            "market_session_open": market_open, "contracts": details, "validation_reasons": reasons,
            "account_wide_evidence": {"complete": True, "as_of": collection_started.isoformat(),
                                      "scope": "current positions and open orders; not historical executions"},
            "coverage": self._coverage(complete, complete), "broker_evidence": raw,
            "pnl_reset_scope": pnl.get("reset_scope", "unknown") if isinstance(pnl, dict) else "unknown",
        }

    @staticmethod
    def order_ref(intent_id: str) -> str:
        if not isinstance(intent_id, str) or not intent_id.strip():
            raise TWSBrokerError("immutable intent identity is required")
        return "pa:rt-" + hashlib.sha256(intent_id.encode("utf-8")).hexdigest()[:40]

    def _intent_arguments(self, intent: dict[str, Any]) -> list[str]:
        if not isinstance(intent, dict) or set(intent) != _INTENT_FIELDS:
            raise TWSBrokerError("native intent has missing or unsupported fields")
        if (intent["account"] != PAPER_BOUNDARY["account"] or intent["order_type"] != "LMT"
                or intent["tif"] != "DAY" or intent["outside_rth"] is not False
                or intent["side"] not in {"BUY", "SELL"} or type(intent["quantity"]) is not int
                or intent["quantity"] <= 0):
            raise TWSBrokerError("native paper intent must be an exact whole-share LMT DAY order")
        price = _money(intent["limit_price"])
        if price <= 0 or price % Decimal("0.01"):
            raise TWSBrokerError("native limit must be positive and cent-denominated")
        return self._contract_arguments(intent) + [
            "--side", intent["side"], "--quantity", str(intent["quantity"]), "--limit", str(price),
            "--tif", "DAY", "--order-ref", self.order_ref(intent["intent_id"]),
            "--risk-policy", str(self.risk_policy_path),
        ]

    def preview(self, intent: dict[str, Any]) -> dict[str, Any]:
        native = self._invoke("preview-stock-limit", self._intent_arguments(intent))
        return {"allowed": True, "intent": deepcopy(intent), "source": "native_tws",
                "broker_connection": False, "native_preview": native}

    def submit(self, intent: dict[str, Any], *, confirm: str, mandate: dict[str, Any] | None = None) -> dict[str, Any]:
        arguments = self._intent_arguments(intent)
        if confirm != "SUBMIT_PAPER_ORDER":
            raise TWSBrokerError("exact SUBMIT_PAPER_ORDER confirmation is required")
        if not isinstance(mandate, dict):
            raise TWSBrokerError("an explicit autonomous portfolio mandate is required")
        try:
            profile = json.dumps(mandate, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (ValueError, TypeError, OverflowError, RecursionError):
            raise TWSBrokerError("portfolio mandate must be finite JSON") from None
        if len(profile.encode("utf-8")) > 65536:
            raise TWSBrokerError("portfolio mandate exceeds the bounded native risk profile size")
        age = mandate.get("max_quote_age_seconds")
        if type(age) is not int or not 0 < age <= 300:
            raise TWSBrokerError("mandate quote age limit must be a positive bounded integer")
        connected_quote_age = min(self.max_quote_age_seconds, age)
        reference = self.order_ref(intent["intent_id"])
        with _COMMAND_LOCK:
            if reference in self._attempted:
                raise TWSBrokerError("intent already attempted; reconcile and never retry")
            self._attempted.add(reference)
            self._intents[reference] = deepcopy(intent)
            try:
                receipt = self._invoke("submit-stock-limit", arguments + [
                    "--confirm", confirm, "--require-live-preflight",
                    "--max-quote-age-seconds", str(connected_quote_age),
                    "--portfolio-risk-policy-json", profile,
                ], mutation=True)
                if (not isinstance(receipt, dict) or receipt.get("order_ref") != reference
                        or receipt.get("account") != PAPER_BOUNDARY["account"]
                        or receipt.get("client_id") != PAPER_BOUNDARY["client_id"]
                        or receipt.get("conid") != intent["conid"]
                        or receipt.get("requested_quantity") != intent["quantity"]):
                    raise TWSBrokerError("broker receipt identity does not match the immutable intent")
                outcome = receipt.get("outcome")
                if outcome not in {"WORKING", "FILLED", "PARTIALLY_FILLED", "CANCELLED", "REJECTED", "SUBMISSION_UNCERTAIN_DO_NOT_RETRY"}:
                    raise TWSBrokerError("broker receipt outcome is unrecognized")
                receipt = {**receipt, "outcome": "UNCERTAIN_DO_NOT_RETRY" if outcome == "SUBMISSION_UNCERTAIN_DO_NOT_RETRY" else outcome,
                           "source": "native_tws", "intent_id": intent["intent_id"], "evidence_verified": outcome != "SUBMISSION_UNCERTAIN_DO_NOT_RETRY"}
            except TWSBrokerError:
                receipt = {"outcome": "UNCERTAIN_DO_NOT_RETRY", "source": "native_tws", "evidence_verified": False,
                           "intent_id": intent["intent_id"],
                           "account": PAPER_BOUNDARY["account"], "client_id": PAPER_BOUNDARY["client_id"],
                           "conid": intent["conid"], "order_ref": reference, "requested_quantity": intent["quantity"]}
            self._receipts[reference] = receipt
            return deepcopy(receipt)

    @staticmethod
    def _audit_rows() -> list[dict[str, Any]]:
        if not CANONICAL_AUDIT.exists():
            return []
        rows, previous = [], "0" * 64
        try:
            for line in CANONICAL_AUDIT.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                observed = record.pop("hash")
                digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if record["prev_hash"] != previous or observed != digest:
                    raise TWSBrokerError("canonical native audit chain is invalid")
                record["hash"] = observed
                rows.append(record)
                previous = digest
        except (ValueError, KeyError, TypeError, OSError):
            raise TWSBrokerError("canonical native audit cannot be verified") from None
        return rows

    def reconcile(self, intent: dict[str, Any]) -> dict[str, Any]:
        self._intent_arguments(intent)
        reference = self.order_ref(intent["intent_id"])
        attempts, historical = [], None
        for row in self._audit_rows():
            detail = row.get("details", {})
            terms = detail.get("terms", {})
            if row.get("event") == "paper_order_submission_attempt" and terms.get("order_ref") == reference:
                attempts.append(terms)
            if row.get("event") == "paper_order_submission_reconciliation" and detail.get("order_ref") == reference:
                historical = detail
        base = {"source": "native_tws", "intent_id": intent["intent_id"], "account": PAPER_BOUNDARY["account"], "client_id": PAPER_BOUNDARY["client_id"],
                "conid": intent["conid"], "order_ref": reference, "requested_quantity": intent["quantity"]}
        uncertain = {**base, "outcome": "UNCERTAIN_DO_NOT_RETRY", "evidence_verified": False}
        cached_intent = self._intents.get(reference)
        if cached_intent is not None and cached_intent != intent:
            return uncertain
        expected = attempts[-1] if attempts else self._receipts.get(reference)
        if expected is None:
            return uncertain
        def positive_id(value: Any) -> bool:
            return type(value) is int and value > 0
        def exact_terms(terms: dict[str, Any]) -> bool:
            try:
                return (
                    terms.get("account") == base["account"] and terms.get("client_id") == 9901
                    and terms.get("conid") == intent["conid"] and terms.get("symbol") == intent["symbol"]
                    and terms.get("primary_exchange") in {None, "", intent["primary_exchange"]}
                    and terms.get("currency") in {None, "USD"} and terms.get("sec_type") in {None, "STK"}
                    and terms.get("order_ref") == reference and terms.get("side") == intent["side"]
                    and _money(terms.get("quantity")) == intent["quantity"]
                    and terms.get("order_type") == "LMT" and terms.get("tif") == "DAY"
                    and terms.get("outside_rth") is False
                    and _money(terms.get("limit_price")) == _money(intent["limit_price"])
                )
            except TWSBrokerError:
                return False
        if attempts and (len(attempts) != 1 or not exact_terms(expected)):
            return uncertain
        if not attempts and cached_intent != intent:
            return uncertain
        if not positive_id(expected.get("order_id")) and not positive_id(expected.get("perm_id")):
            return uncertain
        def valid_receipt(receipt: dict[str, Any] | None) -> bool:
            return bool(receipt and receipt.get("account") == base["account"]
                        and receipt.get("client_id") == 9901 and receipt.get("conid") == intent["conid"]
                        and receipt.get("order_ref") == reference and receipt.get("requested_quantity") == intent["quantity"]
                        and positive_id(receipt.get("order_id")) and receipt["order_id"] == expected.get("order_id"))
        if historical is not None and not valid_receipt(historical):
            # A canonical post-submission exception intentionally omits quantity.
            # Its exact original attempt remains available for later recovery;
            # it must never be used as terminal fill/cancel/rejection proof.
            if (historical.get("outcome") in {"FILLED", "CANCELLED", "REJECTED"}
                    or historical.get("account") != base["account"]
                    or historical.get("client_id") != 9901
                    or historical.get("conid") != intent["conid"]
                    or historical.get("order_id") != expected.get("order_id")):
                return uncertain
            historical = None
        expected = dict(expected)
        for candidate in (historical, self._receipts.get(reference)):
            if valid_receipt(candidate):
                permanent = candidate.get("perm_id") or (candidate.get("last_callback") or {}).get("perm_id")
                if positive_id(permanent):
                    expected["perm_id"] = permanent
        raw = {command: self._invoke(command) for command in ("open-orders", "completed-orders", "executions")}
        # Preserve the complete new envelope before joining its execution rows.
        # Legacy list callers retain the same matching/quantity/outcome behavior.
        try:
            execution_rows, execution_observation = _execution_observation(raw["executions"])
        except TWSBrokerError as error:
            return {**uncertain, "execution_observation": {
                "representation": "INVALID_EXECUTION_OBSERVATION", "original_result": deepcopy(raw["executions"]),
                "parser_error": str(error), "fees_unknown": True}}
        observed_uncertain = {**uncertain, "execution_observation": execution_observation}
        def matches(row: dict[str, Any]) -> bool:
            return (row.get("account") == base["account"] and row.get("conid") == intent["conid"]
                    and row.get("order_ref") == reference
                    and ((positive_id(expected.get("order_id")) and row.get("order_id") == expected["order_id"])
                         or (positive_id(expected.get("perm_id")) and row.get("perm_id") == expected["perm_id"])))
        matching_opens = [row for row in raw["open-orders"] if matches(row) and row.get("client_id") == 9901]
        if any(not exact_terms(row) for row in matching_opens):
            return observed_uncertain
        opens = matching_opens
        completed = [row for row in raw["completed-orders"] if matches(row)
                     and (row.get("client_id") == 9901 or
                          (positive_id(expected.get("perm_id")) and row.get("perm_id") == expected["perm_id"]))]
        verified_completed = []
        for row in completed:
            comparable = dict(row)
            if row.get("client_id") != 9901 and positive_id(expected.get("perm_id")) and row.get("perm_id") == expected["perm_id"]:
                comparable["client_id"] = 9901
            if exact_terms(comparable):
                verified_completed.append(row)
            elif row.get("quantity") not in {None, 0, "0"}:
                # Positive original-looking terms that contradict the canonical
                # immutable attempt cannot establish any accepted outcome.
                return observed_uncertain
        matching_exec = [row for row in execution_rows if matches(row) and row.get("client_id") == 9901]
        expected_side = "BOT" if intent["side"] == "BUY" else "SLD"
        if any(row.get("side") not in {expected_side, intent["side"]}
               or not isinstance(row.get("exec_id"), str) or not row["exec_id"]
               or _money(row.get("shares")) <= 0 for row in matching_exec):
            return observed_uncertain
        executions = {row["exec_id"]: row for row in matching_exec}
        filled = sum((_money(row["shares"]) for row in executions.values()), Decimal(0))
        if filled > intent["quantity"]:
            return observed_uncertain
        statuses = {str(row.get("status", "")).lower() for row in [*opens, *verified_completed]}
        if filled == intent["quantity"]:
            outcome = "FILLED"
        elif filled > 0:
            outcome = "PARTIALLY_FILLED"
        elif statuses & {"cancelled", "apicancelled"}:
            outcome = "CANCELLED"
        elif statuses & {"inactive", "rejected"}:
            outcome = "REJECTED"
        elif "filled" in statuses:
            filled, outcome = Decimal(intent["quantity"]), "FILLED"
        elif opens:
            outcome = "WORKING"
        elif historical and valid_receipt(historical) and historical.get("outcome") in {"FILLED", "CANCELLED", "REJECTED"}:
            observed_filled = _money(historical.get("reconciled_filled_quantity", 0))
            if observed_filled > intent["quantity"] or (historical["outcome"] == "FILLED" and observed_filled != intent["quantity"]):
                return observed_uncertain
            filled, outcome = observed_filled, historical["outcome"]
        else:
            outcome = "UNCERTAIN_DO_NOT_RETRY"
        return {**base, "order_id": expected.get("order_id"), "outcome": outcome,
                "evidence_verified": outcome != "UNCERTAIN_DO_NOT_RETRY",
                "reconciled_filled_quantity": str(filled), "open_orders": opens,
                "completed_orders": completed, "executions": list(executions.values()),
                "execution_observation": execution_observation}
