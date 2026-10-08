"""Offline framework ABI over the unchanged retrospective portfolio ledger.

Importing or constructing this host opens no broker, provider or Budget. The
model supplies complete framework requests. Only the existing mechanical
PaperTWSBroker.order_ref rule derives a native field. No generated economic
term, contract, policy, clock target or failed request is repaired.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal

from training.nav_learning import (PortfolioSimulationHost, SimulationConfig,
    canonical, digest, episode_decision_at, money, number)
from training.native_agent import framework_candidate as component
from training.native_agent.clock_host import ClockWaitHost
from training.framework import model_mandate
from trader_runtime.tws_broker import PaperTWSBroker

INTERFACE = "python_framework_simulator"
ENVIRONMENT = "RETROSPECTIVE_SIMULATION_ONLY"
TIME_SOURCE = "SIMULATED_HISTORICAL_UTC"
MAX_QUANTITY = 3
MAX_NOTIONAL_USD = "2500"


def tool_definition():
    """Reuse the actual eleven-intent/nine-policy schema without mutation."""
    tool = model_mandate.tool_definition(component)
    function = tool[0]["function"]
    function["description"] = (
        "RETROSPECTIVE SIMULATION ONLY. The following production ABI is retained, "
        "but all economic observations and clock/wait results here are simulated historical evidence. "
        "There is no TWS connection, live/delayed BBO, actual broker preflight, wall-clock sleep or provider operation. "
        "Production freshness/profile wording below is qualified by these explicit simulation semantics. "
        "" + function["description"] +
        " In this simulator framework_snapshot returns requested contracts' completed daily history and close marks "
        "only through the current historical session. Positive limits must be exact cents; invalid terms are refused. "
        "framework_preview never creates an order. framework_submit queues unchanged economic fields for the same "
        "PortfolioSimulationHost next-session adverse-adjusted-open LMT DAY execution and modeled fees as the native simulator. "
        "Both interfaces retain the same declared ledger constraints, with an explicit root maximum of three shares "
        "and USD2500 modeled absolute limit notional per order. Model monetary mandate limits are additional visible "
        "admission limits, never substituted for ledger configuration; execution rechecks the unchanged ledger limits. "
        "Mandate symbol/gross exposure admission uses max(model limit,current completed close) conservatively, "
        "without inventing a BBO. Prior same-symbol pending BUY commitments use max(their limit,new intent limit, "
        "current completed close), preserving canonical native mandate admission; aggregate reserved exposure values "
        "each pending commitment at max(its limit,its symbol's completed close). "
        "Daily loss is simulated marked NAV change since the preceding observed session; "
        "a reached model daily-loss limit refuses either side. Account receipt age is zero on fresh local observation; "
        "quote age is simulated UTC elapsed since the completed daily bar became available. Both original execution_profile "
        "labels are retained but have the same disclosed historical fill semantics here. "
        "clock returns SIMULATED_HISTORICAL_UTC. wait_until preserves your exact complete UTC target and immediately "
        "advances the simulated clock and ledger only through completed-session availability gates no later than it. "
        "Each daily session becomes observable at the next New York calendar midnight, the existing simulation assumption; "
        "its open fills and close are disclosed together at that gate. Targets beyond the disclosed terminal gate are "
        "refused unchanged. Intra-session waiting reveals no new price or execution. Past targets return the current "
        "simulated time immediately. No actual production UTC, broker observation, sampling or sleeping is implied. "
        "Reconciliation reads the recorded immutable intent and never retransmits. Unknown identity/fee coverage stays "
        "unknown. Terminal NAV uses the core evaluator's disclosed remaining-time advancement and never liquidates inventory.")
    function["parameters"]["properties"]["until_utc"]["description"] = (
        "Complete model-selected UTC timestamp for retrospective wait_until; SIMULATED_HISTORICAL_UTC, never wall-clock sleep.")
    return tool


class FrameworkSimulationHost:
    """Replayable operational state, with PortfolioSimulationHost owning money."""
    def __init__(self, histories, descriptor, config=None):
        if config is not None and type(config) is not SimulationConfig:
            raise ValueError("the unchanged SimulationConfig object is required")
        self.ledger = PortfolioSimulationHost(histories, descriptor, config)
        self.config = self.ledger.config
        self.max_quantity = MAX_QUANTITY
        self.max_notional_usd = MAX_NOTIONAL_USD
        self.contracts = {}
        for symbol, row in histories.items():
            exchange = row.get("primary_exchange")
            if (row.get("symbol") != symbol or row.get("currency") != "USD"
                    or type(exchange) is not str or not exchange or len(exchange) > 256):
                raise ValueError("dataset-declared USD symbol and primary_exchange required; no synthetic listing repair")
            provenance = row.get("provenance", {})
            source_kind = provenance.get("source_kind", row.get("source_kind", "DATASET_DECLARED_ORIGIN_UNVERIFIED"))
            synthetic = row.get("synthetic_contract_identity", False)
            if type(synthetic) is not bool:
                raise ValueError("synthetic identity annotation must be explicit boolean")
            self.contracts[symbol] = {
                "conid": self.ledger.contract_ids[symbol], "symbol": symbol, "primary_exchange": exchange,
                "currency": "USD", "dataset_asset_type": row.get("asset_type"),
                "source_kind": source_kind, "synthetic_identity": synthetic,
                "qualification": "EXACT_DATASET_IDENTITY_ONLY_NOT_FRESH_BROKER_QUALIFICATION"}
        if len({row["conid"] for row in self.contracts.values()}) != len(self.contracts):
            raise ValueError("unique dataset contract identities required")
        self.gates = [datetime.fromisoformat(episode_decision_at({"start_session": session})).astimezone(timezone.utc)
                      for session in self.ledger.sessions]
        self.current_utc = self.gates[self.ledger.start]
        self.deadline_utc = self.gates[self.ledger.end]
        self.current_policy = None
        self.configuration_receipts = []
        self.intents = {}
        self.once_identities = set()
        self.receipts = []
        self.session_start_nav = self.ledger.initial_nav
        self._terminal = None

    def declaration(self):
        core = self.ledger.declaration()
        return {"tool_address": "pa_tws", "interface": INTERFACE, "environment": ENVIRONMENT,
            "boundary": deepcopy(component.BOUNDARY),
            "operations": sorted(set(component.KEYS) | {model_mandate.CONFIGURE, "clock", "wait_until"}),
            "maximum_quantity": self.max_quantity, "modeled_absolute_limit_notional_usd": self.max_notional_usd,
            "positive_cent_denominated_limits_required": True,
            "ledger_class": "training.nav_learning.PortfolioSimulationHost",
            "limits_and_costs": asdict(self.config), "verified_contracts": deepcopy(list(self.contracts.values())),
            "dataset_universe_is_observation_coverage_not_trade_shortlist": True,
            "fresh_native_broker_qualification": False,
            "model_portfolio_mandate": {"configuration_operation": model_mandate.CONFIGURE,
                "required_fields": sorted(model_mandate.MANDATE_FIELDS), "all_fields_selected_by_model": True,
                "current_policy": deepcopy(self.current_policy), "host_contract_shortlist": None,
                "maximum_order_notional_usd": self.max_notional_usd, "supported_profiles": sorted(model_mandate.PROFILES),
                "policy_revisions_preserve_old_receipts": True, "per_intent_policy_snapshot_immutable": True,
                "monetary_limits_are_visible_admission_constraints": True,
                "quote_age_source": "simulated elapsed UTC since completed-bar availability gate",
                "account_age_source": "fresh local simulated state receipt, age zero",
                "daily_loss_source": "simulated marked NAV change since preceding observed session",
                "symbol_pending_commitment_price": "max(pending limit,new intent limit,current completed close), canonical native formula",
                "gross_pending_commitment_price": "max(pending limit,its symbol's current completed close)",
                "execution_profile_semantics": "both retained labels use the same declared historical ledger; no BBO claimed"},
            "local_temporal_operations": {"clock_source": TIME_SOURCE,
                "current_utc": self.current_utc.isoformat(), "session_deadline_utc": self.deadline_utc.isoformat(),
                "completion_gate": "NEXT_NEW_YORK_CALENDAR_MIDNIGHT_SIMULATION_ASSUMPTION",
                "wait_until_exact_target_retained": True, "target_after_deadline_refused": True,
                "wall_clock_sleep": False, "intra_session_prices_available": False},
            "fills": core["fills"], "terminal": core["terminal"], "qualification": core["qualification"],
            "economic_routing": False, "teacher_trades": False, "forced_exit": False,
            "model_arguments_preserved": True, "final_is_terminal": True,
            "provider_calls": 0, "broker_calls": 0}

    def tool_definition(self):
        return tool_definition()

    def account(self):
        # The runner hashes identical initial economics across both interfaces.
        # Interface and temporal qualifications belong to declaration/results.
        return self.ledger.account()

    def _qualified(self, value):
        symbol = value["symbol"]
        row = self.contracts.get(symbol)
        if row is None or any(type(value[key]) is not type(row[key]) or value[key] != row[key]
                              for key in ("conid", "symbol", "primary_exchange")):
            raise ValueError("exact dataset conid/symbol/primary_exchange identity required; no listing or identity repair")
        return deepcopy(row)

    def _validate(self, request):
        validator = object.__new__(component.FrameworkHost)
        validator.max_quantity, validator.max_notional_usd = self.max_quantity, self.max_notional_usd
        command = validator._validate(request)
        if "intent" in request:
            intent = request["intent"]
            self._qualified(intent)
            if not intent["intent_id"].strip():
                raise ValueError("immutable intent identity is required")
            if number(intent["limit_price"]) % Decimal("0.01"):
                raise ValueError("native framework limit must be positive and cent-denominated; no rounding")
        if command == "framework_snapshot":
            for value in request["contracts"]:
                self._qualified(value)
        return command

    @staticmethod
    def _native_order(intent):
        return {"command": "submit-stock-order", "conid": intent["conid"], "symbol": intent["symbol"],
            "side": intent["side"], "quantity": intent["quantity"], "limit": deepcopy(intent["limit_price"]),
            "order_type": intent["order_type"], "tif": intent["tif"],
            "order_ref": PaperTWSBroker.order_ref(intent["intent_id"])}

    def _policy_check(self, intent):
        if self.current_policy is None:
            raise ValueError("complete model-selected portfolio mandate not configured")
        policy = deepcopy(self.current_policy)
        contract = {key: intent[key] for key in ("conid", "symbol", "primary_exchange")}
        if not any(canonical(contract) == canonical(row) for row in policy["allowed_contracts"]):
            raise ValueError("intent contract is absent from the unchanged model mandate")
        if (self.current_utc - self.gates[self.ledger.cursor]).total_seconds() > policy["max_quote_age_seconds"]:
            raise ValueError("simulated completed-bar source exceeds model-selected quote age limit")
        if self.ledger._nav() - self.session_start_nav <= -number(policy["max_daily_loss_usd"]):
            raise ValueError("model-selected simulated daily loss limit reached")
        s, q, price = intent["symbol"], intent["quantity"], number(intent["limit_price"])
        if q * price > number(policy["max_order_notional_usd"]):
            raise ValueError("model-selected order notional limit exceeded")
        if intent["side"] == "BUY":
            conservative = max(price, self.ledger._mark(s))
            gross = sum((number(amount) * self.ledger._mark(symbol)
                         for symbol, amount in self.ledger.inventory.items()), Decimal(0))
            reserved = sum((number(order["quantity"]) * max(number(order["limit"]), self.ledger._mark(order["symbol"]))
                            for order in self.ledger.orders if order["side"] == "BUY"), Decimal(0))
            symbol_reserved = sum((number(order["quantity"]) * max(number(order["limit"]), conservative)
                                   for order in self.ledger.orders if order["side"] == "BUY" and order["symbol"] == s), Decimal(0))
            if gross + reserved + q * conservative > number(policy["max_gross_exposure_usd"]):
                raise ValueError("model-selected aggregate gross exposure limit exceeded")
            if (number(self.ledger.inventory[s]) + q) * conservative + symbol_reserved > number(policy["max_symbol_exposure_usd"]):
                raise ValueError("model-selected aggregate symbol exposure limit exceeded")
        return policy

    def _ledger_preflight(self, intent):
        if self.ledger.done or self.ledger.cursor == self.ledger.end:
            raise ValueError("no future execution session remains")
        if self.ledger.orders_this_session >= self.config.max_orders_per_session:
            raise ValueError("MAX_ORDER_RATE")
        error = self.ledger._capacity_error(intent["symbol"], intent["side"], intent["quantity"],
            number(intent["limit_price"]), include_pending=True)
        if error:
            raise ValueError(error)

    def _configure(self, request):
        if set(request) != {"command", "mandate"}:
            raise ValueError("exact command and complete model mandate required")
        policy = model_mandate.validate_mandate(request["mandate"], maximum_order_notional_usd=self.max_notional_usd)
        for row in policy["allowed_contracts"]:
            self._qualified(row)
        self.current_policy = deepcopy(policy)
        self.configuration_receipts.append(deepcopy(policy))
        return {"ok": True, "status": "MODEL_MANDATE_CONFIGURED", "policy": deepcopy(policy),
            "configuration_ordinal": len(self.configuration_receipts), "broker_invoked": False}

    def _clock(self, request):
        expected = {"command"} if request["command"] == "clock" else {"command", "until_utc"}
        if set(request) != expected:
            raise ValueError("exact local temporal fields required")
        started, target, advanced = self.current_utc, None, 0
        if request["command"] == "wait_until":
            target = ClockWaitHost._target(request["until_utc"])
            if target > self.deadline_utc:
                raise ValueError("exact model-selected target exceeds simulated historical horizon deadline")
            if target > self.current_utc:
                while self.ledger.cursor < self.ledger.end and self.gates[self.ledger.cursor + 1] <= target:
                    self.session_start_nav = self.ledger._nav()
                    observed = self.ledger.execute({"command": "advance", "sessions": 1})
                    if observed["result"].get("ok") is not True:
                        raise RuntimeError("unchanged ledger advance refused: " + canonical(observed))
                    advanced += 1
                self.current_utc = target
        return {"ok": True, "status": "SIMULATED_CLOCK_OBSERVED" if target is None else "SIMULATED_WAIT_REACHED",
            "clock_source": TIME_SOURCE, "started_at_utc": started.isoformat(),
            "current_utc": self.current_utc.isoformat(), "requested_until_utc": request.get("until_utc"),
            "simulated_elapsed_seconds": (self.current_utc - started).total_seconds(), "sessions_advanced": advanced,
            "completed_session": self.ledger.sessions[self.ledger.cursor], "wall_clock_sleep": False,
            "broker_invoked": False, "provider_invoked": False}

    def _order_observation(self, reference):
        record = self.intents.get(reference)
        if record is None:
            return {"ok": False, "outcome": "OUTCOME_UNKNOWN", "order_ref": reference,
                "evidence_verified": False, "fee_coverage": "UNKNOWN_NO_MATCHING_SIMULATED_ATTEMPT", "commission_usd": None,
                "filled_quantity": None, "remaining_quantity": None, "retransmission": False}
        intent = record["intent"]
        base = {"intent": deepcopy(intent), "intent_sha256": record["intent_sha256"],
            "mandate_at_attempt": deepcopy(record["policy"]), "native_arguments": deepcopy(record["native_arguments"]),
            "order_ref": reference, "order_id": record["order_id"], "account": intent["account"],
            "client_id": component.BOUNDARY["client_id"], "conid": intent["conid"], "requested_quantity": intent["quantity"],
            "simulated": True, "retransmission": False}
        fills = [deepcopy(row) for row in self.ledger.fills if row["order_ref"] == reference]
        if fills:
            exact = all(row.get("symbol") == intent["symbol"] and row.get("side") == intent["side"]
                        and type(row.get("quantity")) is int and row["quantity"] > 0 for row in fills)
            quantity = sum(row["quantity"] for row in fills) if exact else None
            if quantity != intent["quantity"]:
                return {**base, "ok": False, "outcome": "OUTCOME_UNKNOWN", "evidence_verified": False,
                    "filled_quantity": quantity, "remaining_quantity": None, "fills": fills, "commission_usd": None,
                    "fee_coverage": "UNKNOWN_INCOMPLETE_SIMULATED_FILL_EVIDENCE"}
            try:
                commissions = [number(row["commission_usd"]) for row in fills]
                if any(value < 0 for value in commissions):
                    raise ValueError("negative simulated fee")
            except (ValueError, TypeError, KeyError):
                commissions = None
            return {**base, "ok": True, "outcome": "FILLED", "evidence_verified": True,
                "filled_quantity": quantity, "remaining_quantity": 0, "fills": fills,
                "commission_usd": None if commissions is None else money(sum(commissions, Decimal(0))),
                "fee_coverage": "UNKNOWN_MISSING_SIMULATED_FEE" if commissions is None else "COMPLETE_DECLARED_SIMULATION_FEES"}
        queued = [row for row in self.ledger.orders if row["order_ref"] == reference]
        if queued:
            return {**base, "ok": True, "outcome": "WORKING", "evidence_verified": True,
                "filled_quantity": 0, "remaining_quantity": intent["quantity"], "fills": [],
                "commission_usd": None, "fee_coverage": "NOT_EXECUTED_NO_FILL_FEES_YET"}
        if record["cancelled"]:
            status = "CANCELLED"
        elif record["accepted"]:
            expired = [row for row in self.ledger.receipts if row.get("order_ref") == reference
                       and row.get("status") == "EXPIRED_DAY_UNFILLED"]
            if len(expired) != 1:
                return {**base, "ok": False, "outcome": "OUTCOME_UNKNOWN", "evidence_verified": False,
                    "filled_quantity": None, "remaining_quantity": None, "fills": [], "commission_usd": None,
                    "fee_coverage": "UNKNOWN_NO_TERMINAL_SIMULATED_RECEIPT"}
            status = "EXPIRED_DAY_UNFILLED"
        else:
            status = "REJECTED"
        return {**base, "ok": True, "outcome": status, "evidence_verified": True,
            "filled_quantity": 0, "remaining_quantity": 0, "fills": [], "commission_usd": "0",
            "fee_coverage": "COMPLETE_DECLARED_SIMULATION_NO_FILL", "attempt_result": deepcopy(record["attempt_result"])}

    def _execute(self, request):
        if type(request) is not dict or type(request.get("command")) is not str:
            raise ValueError("one exact declared framework command object required")
        canonical(request)
        command = request["command"]
        if command in {"clock", "wait_until"}:
            return self._clock(request)
        if command == model_mandate.CONFIGURE:
            return self._configure(request)
        self._validate(request)
        if command == "framework_status":
            return {"ok": True, "connected": False, "environment": ENVIRONMENT, "clock_source": TIME_SOURCE,
                "current_utc": self.current_utc.isoformat(), "boundary": deepcopy(component.BOUNDARY),
                "simulation_ready": True, "broker_observation": False}
        if command == "framework_snapshot":
            account = self.account()
            history = {row["symbol"]: self.ledger.execute({"command": "historical-daily", "symbol": row["symbol"],
                "count": self.config.history_window})["result"] for row in request["contracts"]}
            return {"ok": True, "environment": ENVIRONMENT, "clock_source": TIME_SOURCE,
                "observed_at_utc": self.current_utc.isoformat(), "account": account,
                "simulated_daily_pnl_usd": money(self.ledger._nav() - self.session_start_nav),
                "daily_pnl_source": "modeled marked NAV change since preceding observed session; broker daily PnL unknown",
                "positions": deepcopy(account["positions"]),
                "open_orders": [self._order_observation(row["order_ref"]) for row in self.ledger.orders],
                "executions": deepcopy(self.ledger.fills), "contracts": [self._qualified(row) for row in request["contracts"]],
                "history": history, "quotes": [{**self._qualified(row), "quality": "SIMULATED_COMPLETED_DAILY_CLOSE",
                    "close_usd": money(self.ledger._mark(row["symbol"])),
                    "price_source": "past completed TRADES daily bar; original publication vintage unverified",
                    "available_at_utc": self.gates[self.ledger.cursor].isoformat(), "live_bbo": False} for row in request["contracts"]],
                "complete_simulation_ledger": True, "complete_broker_history": False}
        if command == "framework_cancel":
            reference = request["order_ref"]
            identity = digest({"kind": "cancel", **{key: request[key] for key in ("conid", "order_id", "order_ref")}})
            if identity in self.once_identities:
                raise ValueError("exact cancellation identity already attempted; no retransmission")
            self.once_identities.add(identity)
            record = self.intents.get(reference)
            if record is None or record["order_id"] != request["order_id"] or record["intent"]["conid"] != request["conid"]:
                raise ValueError("exact simulated PA-owned cancellation identity required")
            observed = self.ledger.execute({"command": "cancel-order", "order_ref": reference})
            if observed["result"].get("ok") is not True:
                raise ValueError(observed["result"].get("error", "unchanged ledger cancellation refused"))
            record["cancelled"] = True
            return self._order_observation(reference)
        intent = request["intent"]
        reference = PaperTWSBroker.order_ref(intent["intent_id"])
        record = self.intents.get(reference)
        if record is not None and canonical(intent) != canonical(record["intent"]):
            raise ValueError("immutable intent identity conflicts with exact earlier terms; no term replacement")
        if command == "framework_reconcile":
            return self._order_observation(reference)
        if command == "framework_preview":
            policy = self._policy_check(intent)
            self._ledger_preflight(intent)
            return {"ok": True, "allowed": True, "status": "SIMULATED_PREVIEW_ONLY", "intent": deepcopy(intent),
                "policy": policy, "order_ref": reference, "native_arguments": self._native_order(intent),
                "economic_order_created": False, "fill_or_fee_implied": False}
        if reference in self.intents:
            raise ValueError("immutable submission identity already attempted; reconcile and never retry")
        # Reserve the exact identity even for a policy/native refusal, matching
        # the production framework's once receipt before invoking its callback.
        native = self._native_order(intent)
        record = {"intent": deepcopy(intent), "intent_sha256": digest(intent), "policy": deepcopy(self.current_policy),
            "native_arguments": deepcopy(native), "order_id": len(self.intents) + 1,
            "accepted": False, "cancelled": False, "attempt_result": None}
        self.intents[reference] = record
        try:
            record["policy"] = self._policy_check(intent)
            observed = self.ledger.execute(native)
            record["attempt_result"] = deepcopy(observed)
            if observed["result"].get("ok") is not True:
                raise ValueError(observed["result"].get("error", "unchanged ledger submission refused"))
            record["accepted"] = True
        except (ValueError, TypeError, KeyError) as error:
            if record["attempt_result"] is None:
                record["attempt_result"] = {"ok": False, "error": str(error), "ledger_invoked": False}
            raise
        return self._order_observation(reference)

    def execute(self, arguments):
        request = deepcopy(arguments)
        try:
            if self._terminal is not None:
                raise ValueError("episode is terminal; no automatic resume")
            result = self._execute(request)
            status = "FRAMEWORK_API_RESULT" if result.get("ok") is not False else "FRAMEWORK_OUTCOME_UNKNOWN"
        except (ValueError, KeyError, TypeError) as error:
            result = {"ok": False, "message": str(error), "economic_terms_repaired": False}
            status = "MODEL_MANDATE_REFUSAL" if type(request) is dict and request.get("command") == model_mandate.CONFIGURE else "FRAMEWORK_REFUSAL"
            if status == "MODEL_MANDATE_REFUSAL":
                result["current_policy_unchanged"] = True
        receipt = {"kind": "retrospective_framework_api_receipt_v1", "ok": result.get("ok") is not False,
            "status": status, "arguments": request, "result": result, "interface": INTERFACE,
            "environment": ENVIRONMENT, "clock_source": TIME_SOURCE, "session": self.ledger.sessions[self.ledger.cursor],
            "observed_at_utc": self.current_utc.isoformat(), "broker_invoked": False, "provider_invoked": False,
            "raw_transport_receipts": [], "transport_attempted": False, "retry_authorized": False}
        self.receipts.append(deepcopy(receipt))
        return receipt

    def terminal_result(self):
        if self._terminal is None:
            result = self.ledger.terminal_result()
            self.current_utc = self.deadline_utc
            self._terminal = {**result, "interface": INTERFACE, "clock_source": TIME_SOURCE,
                "framework_terminal_orders": [self._order_observation(reference) for reference in self.intents],
                "framework_receipt_count": len(self.receipts), "broker_lifecycle_established": False}
        return deepcopy(self._terminal)
